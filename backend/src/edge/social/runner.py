# backend/src/edge/social/runner.py
"""One social pass per worker tick: cheat sheet, value plays, blind plays, final recaps, receipts.

Every post goes: switch on (ledger.enabled) -> brake not tripped -> claim the (kind, key) ->
build (safety check, render) -> post -> finish. A key is claimed once, so a post never goes out
twice; a check that fails for good marks the key `skipped`, a check that may pass later (game not
final yet, ungraded play) claims nothing and waits. Any error in one item is logged and recorded
on its ledger row; social_step itself never raises.

Times are ET (zoneinfo, DST-safe). "Fires at 11:00" means the first tick at/after 11:00 ET that day.
"""
from __future__ import annotations

import asyncio
import logging
import time as time_mod
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from typing import Awaitable, Callable

import httpx
from psycopg2.extras import RealDictCursor

from src.config import settings
from src.edge import nfl_players, nightly, players, store, views
from src.edge.config import SPORTS
from src.edge.grading import profit_units
from src.edge.players import ET, et_date
from src.edge.social import blind, cards, content, headshots, ledger, safety
from src.edge.social.poster import BlotatoPoster, DryRunPoster, Poster, PostResult
from src.edge.teams import ABBR_BY_NAME, NFL_ABBR_BY_NAME
from src.utils.redact import redact

log = logging.getLogger(__name__)

CHEATSHEET_AT = {"nba": (time(11, 0), frozenset(range(7))),
                 "nfl": (time(10, 0), frozenset({3, 6, 0}))}  # Thu, Sun, Mon (MNF)
VALUE_SPORTS = frozenset({"nba"})
FINAL_WINDOW = timedelta(hours=36)
RECEIPT_DAYS = 3
BLIND_EVERY = timedelta(minutes=5)  # selection rebuilds the board and matchup panels: not every 30 s tick
BLIND_WINDOW = (time(10, 0), time(22, 0))  # ET: blind plays post from 10:00 up to (not at) 22:00
BLIND_MIN_GAP = timedelta(minutes=20)      # since the sport's last blind post (posted or dry run)


SOCIAL_PASS_BUDGET = 120  # seconds: past this a pass starts no new post or Blotato lookup
VALUE_MIN_LEAD = timedelta(minutes=10)       # every Top-5 leg must tip at least this far out when posted
CHEATSHEET_MIN_LEAD = timedelta(minutes=10)  # a cheat-sheet prop counts only if its game is this far out


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _monotonic() -> float:
    return time_mod.monotonic()


def _note_unknown(conn, now: datetime) -> None:
    """Worker state social_unknown: lost live posts Blotato can't be asked about (no submission id)."""
    store.set_state(conn, "social_unknown", str(ledger.unknown_without_submission(conn)), now)


def default_poster() -> Poster:
    """Blotato only when the master switch is on and dry-run is off; dry run otherwise."""
    if settings.twitter_posting_enabled and not settings.social_dry_run:
        return BlotatoPoster()
    return DryRunPoster()


def _http() -> httpx.AsyncClient:
    return httpx.AsyncClient()


def blind_window_open(now: datetime) -> bool:
    return BLIND_WINDOW[0] <= now.astimezone(ET).time() < BLIND_WINDOW[1]


def cheatsheet_due(sport: str, now: datetime) -> date | None:
    """The ET date whose cheat sheet is due at `now`, or None (wrong day or too early)."""
    at = CHEATSHEET_AT.get(sport)
    local = now.astimezone(ET)
    if at is None or local.weekday() not in at[1] or local.time() < at[0]:
        return None
    return local.date()


def _abbr_map(sport: str) -> dict[str, str]:
    return ABBR_BY_NAME if sport == "nba" else NFL_ABBR_BY_NAME


def _bet_key(r: dict) -> tuple:
    return (r["event_id"], r["market"], r["player"], r["side"], float(r["line"]))


def _matchup_teams(r: dict) -> tuple[str | None, str | None]:
    """(home, away) full names from home_team/away_team or 'Away @ Home'."""
    if r.get("home_team") and r.get("away_team"):
        return r["home_team"], r["away_team"]
    away, _, home = (r.get("matchup") or "").partition(" @ ")
    return (home or None), (away or None)


# ---- DB helpers (each one is a test seam) -------------------------------------------------------

def _live_rows(conn, sport: str, now: datetime) -> list[dict]:
    return views.ev_rows(conn, now, sport=SPORTS[sport])


def _gate(conn, sport: str) -> str | None:
    return store.get_state(conn).get(f"social_blind_gate:{sport}")


def _top_legs(conn, sport: str, day: date) -> list[dict]:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""SELECT * FROM edge_nightly_legs WHERE sport = %s AND slate_date = %s AND kind = 'top'
                       ORDER BY rank""", (sport, day))
        rows = [dict(r) for r in cur.fetchall()]
    conn.rollback()
    for r in rows:
        r["line"], r["price"] = float(r["line"]), int(r["price"])
        r["fair_prob"], r["ev"] = float(r["fair_prob"]), float(r["ev"])
    return rows


def _blind_to_post(conn, sport: str, now: datetime) -> list[dict]:
    """Recorded blind plays whose game has not started (the ledger decides which are still unposted)."""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT * FROM edge_blind_plays WHERE sport = %s AND commence_time > %s ORDER BY n",
                    (sport, now))
        rows = [dict(r) for r in cur.fetchall()]
    conn.rollback()
    return rows


def _slate_tips(conn, sport: str, day: date) -> dict[str, datetime]:
    lo = datetime(day.year, day.month, day.day, tzinfo=ET)
    with conn.cursor() as cur:
        cur.execute("""SELECT event_id, commence_time FROM edge_events WHERE sport = %s AND status <> 'unavailable'
                       AND commence_time >= %s AND commence_time < %s""",
                    (SPORTS[sport].key, lo, lo + timedelta(days=1)))
        rows = cur.fetchall()
    conn.rollback()
    return {e: t for e, t in rows}


def _today_teams(conn, sport: str, day: date) -> set[str]:
    lo = datetime(day.year, day.month, day.day, tzinfo=ET)
    with conn.cursor() as cur:
        cur.execute("""SELECT home_team, away_team FROM edge_events WHERE sport = %s AND status <> 'unavailable'
                       AND commence_time >= %s AND commence_time < %s""",
                    (SPORTS[sport].key, lo, lo + timedelta(days=1)))
        rows = cur.fetchall()
    conn.rollback()
    abbr = _abbr_map(sport)
    return {abbr[n] for pair in rows for n in pair if n in abbr}


def _headshots_due(conn, sport: str, now: datetime) -> bool:
    """True once per sport per ET day; the day is stored before the refresh runs, so a failing
    refresh is not retried every tick."""
    key, day = f"social_headshots:{sport}", et_date(now).isoformat()
    if store.get_state(conn).get(key) == day:
        return False
    store.set_state(conn, key, day, now)
    return True


def _blind_due(conn, sport: str, now: datetime) -> bool:
    key = f"social_blind_at:{sport}"
    last = store.get_state(conn).get(key)
    if last and now - datetime.fromisoformat(last) < BLIND_EVERY:
        return False
    store.set_state(conn, key, now.isoformat(), now)
    return True


def _last_blind_post(conn, sport: str) -> datetime | None:
    """When the sport's last blind play went out (posted or dry run), or None."""
    with conn.cursor() as cur:
        cur.execute("""SELECT MAX(updated_at) FROM edge_social_posts
                       WHERE kind = 'blind' AND sport = %s AND status IN ('posted', 'dry_run')""", (sport,))
        row = cur.fetchone()
    conn.rollback()
    return row[0] if row else None


_EVENT_COLS = "event_id, home_team, away_team, commence_time, status, completed, home_score, away_score"


def _final_events(conn, sport: str, now: datetime) -> list[dict]:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(f"""SELECT {_EVENT_COLS} FROM edge_events WHERE sport = %s AND completed
                        AND commence_time >= %s AND commence_time <= %s ORDER BY commence_time""",
                    (SPORTS[sport].key, now - FINAL_WINDOW, now))
        rows = [dict(r) for r in cur.fetchall()]
    conn.rollback()
    return rows


_WIN = """JOIN LATERAL (
    SELECT o.result FROM edge_opportunities o
    WHERE o.event_id = x.event_id AND o.market = x.market AND o.player = x.player
      AND o.side = x.side AND o.line = x.line AND o.result IS NOT NULL ORDER BY o.id LIMIT 1) res
    ON res.result = 'win'"""
_EV_JOIN = """JOIN edge_events e ON e.event_id = x.event_id"""
_EV_SELECT = ", ".join(f"e.{c} AS ev_{c}" for c in _EVENT_COLS.split(", "))


def _receipt_candidates(conn, sport: str, now: datetime) -> list[dict]:
    """Graded wins of blind plays and Top-5 legs from the last few slates, with their event."""
    since = et_date(now) - timedelta(days=RECEIPT_DAYS)
    out = []
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(f"""SELECT x.*, NULL::int AS rank, {_EV_SELECT}, res.result FROM edge_blind_plays x
                        {_EV_JOIN} {_WIN} WHERE x.sport = %s AND x.slate_date >= %s ORDER BY x.n""",
                    (sport, since))
        out += [{**dict(r), "kind": "blind"} for r in cur.fetchall()]
        cur.execute(f"""SELECT x.*, NULL::int AS n, NULL::jsonb AS panel, NULL::text AS position, {_EV_SELECT},
                               res.result FROM edge_nightly_legs x {_EV_JOIN} {_WIN}
                        WHERE x.sport = %s AND x.kind = 'top' AND x.slate_date >= %s
                        ORDER BY x.slate_date, x.rank""", (sport, since))
        out += [{**dict(r), "kind": "value"} for r in cur.fetchall()]
    conn.rollback()
    for r in out:
        r["event"] = {c: r.pop(f"ev_{c}") for c in _EVENT_COLS.split(", ")}
        r["sport"] = sport
        r["line"], r["price"] = float(r["line"]), int(r["price"])
        r["ev"] = float(r["ev"]) if r.get("ev") is not None else None
        r.setdefault("matchup", None)
        if not r.get("matchup"):
            r["matchup"] = f"{r['event']['away_team']} @ {r['event']['home_team']}"
    return out


def _second_source(conn, sport: str, event: dict) -> tuple[int, int] | None:
    return (content.second_source_nba if sport == "nba" else content.second_source_nfl)(conn, event)


def _assign_teams(conn, sport: str, rows: list[dict]) -> None:
    """Set row["team"] (abbr) to the player's roster team when exactly one of the game's two teams
    matches; None otherwise (no logo or face beats a wrong one)."""
    names = sorted({r["player"] for r in rows if r.get("player")})
    if not names:
        return
    table = "edge_players" if sport == "nba" else "edge_nfl_players"
    with conn.cursor() as cur:
        cur.execute(f"SELECT odds_name, team FROM {table} WHERE odds_name = ANY(%s)", (names,))
        rosters: dict[str, set[str]] = {}
        for name, team in cur.fetchall():
            if team:
                rosters.setdefault(name, set()).add(team)
    conn.rollback()
    abbr = _abbr_map(sport)
    for r in rows:
        home, away = _matchup_teams(r)
        cands = rosters.get(r.get("player"), set()) & {abbr.get(home or ""), abbr.get(away or "")}
        r["team"] = next(iter(cands)) if len(cands) == 1 else None


def _box_line(conn, sport: str, play: dict) -> tuple[float | None, str | None]:
    """(graded stat, 'pts · reb · ast' line) from the player's Final box score. NBA only."""
    if sport != "nba" or play["market"] not in players.STAT_PARTS:
        return None, None
    tip = safety.tip_of(play["event"])
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""SELECT g.* FROM edge_player_games g JOIN edge_players p ON p.bdl_player_id = g.bdl_player_id
                       WHERE p.odds_name = %s AND g.game_date = %s""", (play["player"], et_date(tip)))
        rows = cur.fetchall()
    conn.rollback()
    if len(rows) != 1:
        return None, None
    g = rows[0]
    return players.stat_for(play["market"], g), f"{g['pts']} pts · {g['reb']} reb · {g['ast']} ast"


def _blind_players(conn, sport: str, day: date) -> set[str]:
    """Players with a blind play recorded for the slate (posted or not): never named elsewhere."""
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT player FROM edge_blind_plays WHERE sport = %s AND slate_date = %s", (sport, day))
        out = {r[0] for r in cur.fetchall()}
    conn.rollback()
    return out


def _cheatsheet_rows(conn, sport: str, rows: list[dict], exclude: set[str] = frozenset()) -> list[dict]:
    """Rows ranked for the sheet, without the `exclude` players. NBA form from edge_player_games; NFL
    from nfl_players.load_games (the NFL player page's games), which needs the player's team: roster
    team ∩ the game's teams."""
    rows = [r for r in rows if r["player"] not in exclude]
    rows = [r for r in rows if r.get("bdl_player_id")]
    if sport == "nfl":
        _assign_teams(conn, sport, rows)
        games: dict[int, list[dict]] = {}
        for r in rows:
            if r.get("team") and r["bdl_player_id"] not in games:
                games[r["bdl_player_id"]] = nfl_players.load_games(conn, r["bdl_player_id"], r["team"], r["player"])
    else:
        ids = sorted({r["bdl_player_id"] for r in rows})
        games = players.load_recent_games(conn, ids) if ids else {}
    conn.rollback()
    return content.cheatsheet_rows(rows, games, sport, exclude_players=exclude)


# ---- the pass -----------------------------------------------------------------------------------

@dataclass
class Draft:
    texts: list[str]
    images: list[list[bytes]]
    detail: dict = field(default_factory=dict)
    # Called with the wall clock right before the post goes out; False -> skipped ("late at post time").
    recheck: Callable[[datetime], bool] | None = None


@dataclass
class Skip:
    reason: str


Build = Callable[[], Awaitable["Draft | Skip"]]


class _Pass:
    def __init__(self, conn, now: datetime, poster: Poster, http):
        self.conn, self.now, self.poster, self.http = conn, now, poster, http
        self.today = et_date(now)
        self.counted = ("posted", "dry_run") if poster.dry_run else ("posted",)
        self.braked: set[str] = set()
        self.out: dict = {}
        self.started = _monotonic()
        self.halt: str | None = None  # why this pass stopped posting (Blotato failure or budget)

    def _rollback(self) -> None:
        try:
            self.conn.rollback()
        except Exception:  # noqa: BLE001
            pass

    def _tally(self, sport: str, kind: str, status: str) -> None:
        self.out.setdefault(sport, {}).setdefault(kind, {}).setdefault(status, 0)
        self.out[sport][kind][status] += 1

    def was_posted(self, kind: str, key: str) -> dict | None:
        row = ledger.find(self.conn, kind, key)
        return row if row and row["status"] in self.counted else None

    def settled(self, kind: str, key: str) -> bool:
        row = ledger.find(self.conn, kind, key)
        return bool(row) and row["status"] in ("posted", "dry_run", "skipped", "unknown")

    def brake(self, sport: str) -> bool:
        """True (and logged loudly, once per pass) when the sport's runaway brake is tripped."""
        if sport in self.braked:
            return True
        if not ledger.brake_tripped(self.conn, sport, self.now):
            return False
        log.error("SOCIAL BRAKE TRIPPED for %s: posting stopped (caps %s/day, %s/hour)", sport,
                  settings.social_max_per_day, settings.social_max_per_hour)
        self.braked.add(sport)
        self.out.setdefault(sport, {})["brake"] = True
        return True

    def stopped(self) -> bool:
        """True once Blotato failed in this pass or the pass ran past SOCIAL_PASS_BUDGET: no new posts
        or lookups start, and nothing more is claimed (the next tick picks it up)."""
        if self.halt is None and _monotonic() - self.started > SOCIAL_PASS_BUDGET:
            self.halt = "budget"
            log.error("social pass ran past its %ss budget; no more posts this pass", SOCIAL_PASS_BUDGET)
        return self.halt is not None

    def trip(self, why: str | None) -> None:
        if self.halt is None:
            self.halt = f"blotato: {why}"
            log.error("social: Blotato failed (%s); no more posts or lookups this pass", why)

    async def publish(self, sport: str, kind: str, key: str, build: Build) -> str:
        if self.stopped():
            return "stopped"
        if self.brake(sport):
            return "brake"
        c = ledger.claim(self.conn, sport, kind, key, self.now, dry_run=self.poster.dry_run)
        if c is None:
            return "done"
        try:
            draft = await build()
        except Exception as e:  # noqa: BLE001 - one bad card never stops the rest
            log.exception("social %s %s: build failed", kind, key)
            self._rollback()
            ledger.finish(self.conn, c, "failed", self.now, text="",
                          detail={"error": redact(f"{type(e).__name__}: {e}")[:300]})
            self._tally(sport, kind, "failed")
            return "failed"
        if isinstance(draft, Skip):
            log.info("social %s %s skipped: %s", kind, key, draft.reason)
            ledger.finish(self.conn, c, "skipped", self.now, text="", detail={"reason": draft.reason})
            self._tally(sport, kind, "skipped")
            return "skipped"
        if draft.recheck is not None and not draft.recheck(_utcnow()):
            log.warning("social %s %s skipped: late at post time", kind, key)
            ledger.finish(self.conn, c, "skipped", self.now, text="", detail={"reason": "late at post time"})
            self._tally(sport, kind, "skipped")
            return "skipped"
        try:
            res = await asyncio.to_thread(self.poster.post, draft.texts, draft.images)
        except Exception as e:  # noqa: BLE001 - outcome unknown: never retried
            log.exception("social %s %s: poster raised", kind, key)
            res = PostResult(False, None, [], None, retryable=False, error=redact(f"{type(e).__name__}: {e}")[:300])
        if not res.ok:
            self.trip(res.error)
        if res.ok and not self.poster.dry_run:
            try:  # first, so a worker lost before finish() leaves a row Blotato can reconcile
                ledger.note_submission(self.conn, c, res.submission_id,
                                       res.media_urls[0] if res.media_urls else None)
            except Exception:  # noqa: BLE001
                log.exception("social %s %s: could not store the submission id", kind, key)
                self._rollback()
        if res.ok:
            status = "dry_run" if self.poster.dry_run else "posted"
        else:
            status = "failed" if res.retryable else "skipped"
            log.error("social %s %s not posted (%s): %s", kind, key, status, res.error)
        first = next((img for imgs in draft.images for img in imgs), None)
        detail = {**draft.detail, "texts": draft.texts, "media_urls": res.media_urls,
                  **({"error": res.error} if res.error else {})}
        ledger.finish(self.conn, c, status, self.now, text="\n\n---\n\n".join(draft.texts), image=first,
                      media_url=res.media_urls[0] if res.media_urls else None, submission_id=res.submission_id,
                      post_url=res.post_url, detail=detail)
        self._tally(sport, kind, status)
        return status

    async def guarded(self, what: str, fn, *args):
        """fn's result, or None when it raised (logged; one kind failing never stops the next)."""
        try:
            return await fn(*args)
        except Exception:  # noqa: BLE001
            log.exception("social %s failed", what)
            self._rollback()
            return None

    # ---- media ----

    async def faces(self, sport: str, rows: list[dict]) -> dict[str, bytes | None]:
        out: dict[str, bytes | None] = {}
        for r in rows:
            name, team = r.get("player"), r.get("team")
            if name and team and name not in out:
                out[name] = await headshots.fetch_png(self.conn, sport, name, team, self.http)
        return out

    async def logos(self, sport: str, abbrs) -> dict[str, bytes | None]:
        return {a: await headshots.fetch_logo(sport, a, self.http) for a in sorted({a for a in abbrs if a})}

    # ---- per sport ----

    async def run_sport(self, sport: str) -> None:
        if not self.stopped() and _headshots_due(self.conn, sport, self.now):  # inside the pass budget
            try:
                await headshots.refresh(self.conn, sport, _today_teams(self.conn, sport, self.today), self.now,
                                        self.http, stop=self.stopped)
            except Exception:  # noqa: BLE001
                log.exception("social headshots refresh failed for %s", sport)
                self._rollback()
        rows_cache: list[list[dict]] = []

        def live() -> list[dict]:
            if not rows_cache:
                rows_cache.append([r for r in _live_rows(self.conn, sport, self.now)
                                   if content.slate_for(safety.tip_of(r)) == self.today])
            return rows_cache[0]

        await self.guarded(f"{sport} cheatsheet", self.cheatsheet, sport, live)
        if sport in VALUE_SPORTS:
            await self.guarded(f"{sport} value", self.value, sport)
        await self.guarded(f"{sport} blind", self.blind, sport, live)
        await self.guarded(f"{sport} final", self.finals, sport)
        await self.guarded(f"{sport} receipt", self.receipts, sport)

    async def cheatsheet(self, sport: str, live) -> None:
        day = cheatsheet_due(sport, self.now)
        if day is None or not ledger.enabled("cheatsheet", sport):
            return
        key = f"cheatsheet:{sport}:{day.isoformat()}"
        if self.settled("cheatsheet", key):
            return

        def far_enough(rows: list[dict], t: datetime) -> list[dict]:
            return [r for r in rows if safety.tip_of(r) >= t + CHEATSHEET_MIN_LEAD]

        async def build():
            rows = far_enough(_cheatsheet_rows(self.conn, sport, far_enough(live(), self.now),
                                               _blind_players(self.conn, sport, day)), self.now)
            if len(rows) < content.CHEATSHEET_MIN:
                return Skip("fewer than 6 +EV props with a last-10 hit rate, 10+ minutes before tip")
            _assign_teams(self.conn, sport, rows)
            png = cards.render_cheatsheet(sport, day, rows, await self.faces(sport, rows),
                                          logos=await self.logos(sport, (r.get("team") for r in rows)))
            # every prop on the card must still be 10+ minutes from its tip when the post goes out
            return Draft([content.cheatsheet_text(sport, day)], [[png]], {"rows": len(rows)},
                         recheck=lambda t: len(far_enough(rows, t)) == len(rows))
        await self.publish(sport, "cheatsheet", key, build)

    async def render_value(self, sport: str, day: date, legs: list[dict]) -> bytes:
        legs = [{**l, "rank": int(l["rank"]) + 1} for l in legs]  # stored 0-based, shown 1-based
        _assign_teams(self.conn, sport, legs)
        return cards.render_value(sport, day, legs, await self.faces(sport, legs), legs[0]["locked_at"],
                                  logos=await self.logos(sport, (l.get("team") for l in legs)))

    async def value(self, sport: str) -> None:
        if not ledger.enabled("value", sport):
            return
        key = f"value:{sport}:{self.today.isoformat()}"
        if self.settled("value", key):
            return
        legs = _top_legs(self.conn, sport, self.today)
        if not legs:
            return  # not locked yet

        def lead_ok(t: datetime) -> bool:
            return safety.value_ok(legs, t) and all(safety.tip_of(l) >= t + VALUE_MIN_LEAD for l in legs)

        async def build():
            if not lead_ok(self.now):
                return Skip("a Top-5 game starts within 10 minutes or has started")
            png = await self.render_value(sport, self.today, legs)
            return Draft([content.value_text(sport, legs[0]["locked_at"])], [[png]], {"legs": len(legs)},
                         recheck=lead_ok)
        await self.publish(sport, "value", key, build)

    def blind_spaced(self, sport: str) -> bool:
        """True when the sport's last blind post (posted or dry run) is BLIND_MIN_GAP or more ago."""
        last = _last_blind_post(self.conn, sport)
        return last is None or self.now - last >= BLIND_MIN_GAP

    async def blind(self, sport: str, live) -> None:
        """At most one blind post per pass, BLIND_MIN_GAP apart, inside BLIND_WINDOW. Gate, switch,
        brake, window and spacing are checked before anything is numbered. A play recorded on an
        earlier pass but never settled (the worker died before posting) goes first; otherwise the
        best qualifying candidate is recorded, alone, right before it is posted."""
        if (not ledger.enabled("blind", sport) or _gate(self.conn, sport) != "pass" or self.stopped()
                or self.brake(sport)):
            return
        if not blind_window_open(self.now) or not self.blind_spaced(sport):
            return
        if not _blind_due(self.conn, sport, self.now):
            return
        rows = live()
        current = {(*_bet_key(r), r["book"]): r for r in rows}
        tips: dict[date, dict] = {}
        if await self.post_recorded(sport, current, tips) or self.stopped():
            return
        top5 = {l["player"] for l in _top_legs(self.conn, sport, self.today)}
        if sport in VALUE_SPORTS:  # not locked yet: today's likely 5 (no value post, nothing to keep off)
            top5 |= {r["player"] for r in nightly.top_picks(nightly.top5_eligible(rows))}
        panels = blind.build_panels(self.conn, sport, rows, self.now)
        for pick in blind.select(rows, top5, panels, self.now):
            if blind.record(self.conn, sport, [pick], panels, self.now):  # [] = refused (slate limits)
                await self.post_recorded(sport, current, tips)
                return

    async def post_recorded(self, sport: str, current: dict, tips: dict) -> bool:
        """Post the first recorded, unsettled, not-yet-started blind play that goes out; earlier ones
        that fail their checks are skipped. True once one was posted (or dry-run posted)."""
        for play in _blind_to_post(self.conn, sport, self.now):
            key = f"blind:{sport}:{play['n']}"
            if self.settled("blind", key):
                continue
            status = await self.guarded(key, self.blind_one, sport, key, play, current, tips)
            if status in ("posted", "dry_run"):
                return True
            if self.stopped():
                return False
        return False

    async def blind_one(self, sport: str, key: str, play: dict, current: dict, tips: dict) -> str:
        async def build():
            now_row = current.get((*_bet_key(play), play["book"]))
            if now_row is None:
                return Skip("no longer +EV at its book")
            if not safety.pick_ok(now_row, self.now, now_row["ev"]):
                return Skip("under 60 minutes to tip or EV gone")
            day = play["slate_date"]
            if day not in tips:
                tips[day] = _slate_tips(self.conn, sport, day)
            png = cards.render_blind(blind.to_card(play, tips[day]))
            n = play["n"]
            return Draft([content.blind_text(sport, n), content.blind_reply(sport, n)], [[png], []],
                         {"n": n, "event_id": play["event_id"]},
                         recheck=lambda t: safety.pick_ok(now_row, t, now_row["ev"]))
        return await self.publish(sport, "blind", key, build)

    def posted_in(self, sport: str, event: dict) -> list[dict]:
        slate = content.slate_for(safety.tip_of(event))
        out = []
        for p in content.posted_plays(self.conn, event["event_id"]):
            if p.get("sport") != sport:
                continue
            key = (("blind", f"blind:{sport}:{p['n']}") if p["kind"] == "blind"
                   else ("value", f"value:{sport}:{slate.isoformat()}"))
            if self.was_posted(*key):
                out.append(p)
        return out

    async def finals(self, sport: str) -> None:
        if not ledger.enabled("final", sport):
            return
        records: list[dict] = []
        for event in _final_events(self.conn, sport, self.now):
            key = f"final:{event['event_id']}"
            if self.settled("final", key):
                continue
            plays = self.posted_in(sport, event)
            if not plays or any(p.get("result") is None for p in plays):
                continue  # nothing of ours in it, or not graded yet: wait
            if not safety.final_ok(event, _second_source(self.conn, sport, event)):
                continue  # not final or the scores disagree: wait, never post
            if not records:  # SOCIAL_SHOW_RECORDS off: the card draws no W-L records
                records.append(content.team_records(self.conn, sport, content.SEASON_START[sport])
                               if settings.social_show_records else {})
            await self.guarded(key, self.final_one, sport, key, event, plays, records[0])

    async def final_one(self, sport: str, key: str, event: dict, plays: list[dict], records: dict) -> None:
        async def build():
            w, l, p, units = content.record_line(plays)
            rows = [{**pl, "home_team": event["home_team"], "away_team": event["away_team"]} for pl in plays]
            _assign_teams(self.conn, sport, rows)
            abbr = _abbr_map(sport)
            logos = await self.logos(sport, (abbr.get(event["home_team"]), abbr.get(event["away_team"])))
            png = cards.render_final({**event, "sport": sport}, records, logos, rows, await self.faces(sport, rows))
            return Draft([content.final_text(sport, event, w, l, p, units)], [[png]],
                         {"event_id": event["event_id"], "plays": len(plays)})
        await self.publish(sport, "final", key, build)

    async def receipts(self, sport: str) -> None:
        if not ledger.enabled("receipt", sport):
            return
        blind_bets = set()
        cands = _receipt_candidates(self.conn, sport, self.now)
        for r in cands:  # only a blind play that actually went out covers its bet
            if r["kind"] == "blind" and self.was_posted("blind", f"blind:{sport}:{r['n']}"):
                blind_bets.add(_bet_key(r))
        checked: dict[str, bool] = {}
        for r in cands:
            if r["kind"] == "blind":
                orig = ("blind", f"blind:{sport}:{r['n']}")
                key = f"receipt:blind:{sport}:{r['n']}"
            else:
                if _bet_key(r) in blind_bets:
                    continue  # the blind receipt covers this bet
                slate = r["slate_date"]
                orig = ("value", f"value:{sport}:{slate.isoformat()}")
                key = f"receipt:value:{sport}:{slate.isoformat()}:{r['rank']}"
            if self.settled("receipt", key):
                continue
            original = self.was_posted(*orig)
            if original is None:
                continue  # never posted (or only in dry run while live): no receipt
            ev = r["event"]
            if ev["event_id"] not in checked:
                checked[ev["event_id"]] = safety.final_ok(ev, _second_source(self.conn, sport, ev))
            if not checked[ev["event_id"]]:
                continue  # wait for a verified final
            await self.guarded(key, self.receipt_one, sport, key, r, original)

    async def receipt_one(self, sport: str, key: str, r: dict, original: dict) -> None:
        async def build():
            play = {**r, "profit_units": round(profit_units("win", int(r["price"])), 4)}
            if r["kind"] == "value":
                play["n"] = int(r["rank"]) + 1
            actual, stat_line = _box_line(self.conn, sport, play)
            if actual is not None:
                hit = actual > play["line"] if play["side"] == "over" else actual < play["line"]
                if not hit:
                    return Skip(f"box score {actual:g} disagrees with the win grade")
                play["actual"], play["stat_line"] = actual, stat_line
            _assign_teams(self.conn, sport, [play])
            face = (await headshots.fetch_png(self.conn, sport, play["player"], play["team"], self.http)
                    if play.get("team") else None)
            logo = await headshots.fetch_logo(sport, play["team"], self.http) if play.get("team") else None
            orig_png = bytes(original["image"]) if original.get("image") else await self.rerender(sport, r)
            url = original.get("post_url")
            if r["kind"] == "blind":
                text = content.blind_receipt_text(r["n"], r["player"], int(r["price"]), sport, url)
            else:
                text = content.value_receipt_text(r["bet"], int(r["price"])) + (f"\n\n{url}" if url else "")
            if url:
                images = [cards.render_receipt(play, face, logo, orig_png)]
            else:
                images = [cards.render_receipt(play, face, logo)] + ([orig_png] if orig_png else [])
            return Draft([text], [images], {"original": original.get("key"), "quoted": bool(url)})
        await self.publish(sport, "receipt", key, build)

    async def rerender(self, sport: str, r: dict) -> bytes | None:
        """The original card when the ledger kept no image. None if it can't be drawn."""
        try:
            if r["kind"] == "blind":
                return cards.render_blind(blind.to_card(r, _slate_tips(self.conn, sport, r["slate_date"])))
            legs = _top_legs(self.conn, sport, r["slate_date"])
            return await self.render_value(sport, r["slate_date"], legs) if legs else None
        except Exception:  # noqa: BLE001
            log.exception("social: original card re-render failed")
            self._rollback()
            return None

    # ---- post URLs ----

    def ask(self, sid: str):
        """One Blotato status lookup, or None when the pass is stopped or Blotato didn't answer
        (which stops the pass)."""
        if self.stopped():
            return None
        try:
            found = self.poster.lookup(sid)
        except Exception as e:  # noqa: BLE001
            self.trip(redact(f"lookup {type(e).__name__}: {e}")[:300])
            return None
        if found.state == "unknown":
            self.trip("status lookup got no answer")
            return None
        return found

    def resolve_urls(self) -> None:
        """Ask Blotato about recent posts: confirm `unknown` rows it published (they become posted;
        anything else stays unknown and is never re-posted), then fill in missing public URLs.
        At most 5 + 5 lookups per tick (Blotato allows 60/min)."""
        if self.poster.dry_run:
            return
        for row_id, sid in ledger.unknown_submissions(self.conn, self.now):
            found = self.ask(sid)
            if found is None:
                return
            if found.state == "published":
                ledger.confirm_posted(self.conn, row_id, found.url)
        for row_id, sid in ledger.unresolved(self.conn, self.now):
            found = self.ask(sid)
            if found is None:
                return
            if found.state == "published":
                ledger.note_lookup(self.conn, row_id, "published", found.url)
            elif found.state == "failed":
                log.error("social: Blotato reports submission %s failed: %s", sid, found.error)
                ledger.note_lookup(self.conn, row_id, "failed", error=found.error)


def _sports() -> list[str]:
    seen = []
    for s in settings.social_sports.split(","):
        s = s.strip()
        if s in SPORTS and s not in seen:
            seen.append(s)
    return seen


async def social_step(conn, now: datetime, poster: Poster | None = None, http=None) -> dict:
    """One pass over every enabled sport. Never raises."""
    own_http = None
    p: _Pass | None = None
    try:
        poster = poster or default_poster()
        if http is None:
            own_http = http = _http()
        p = _Pass(conn, now, poster, http)
        try:
            _note_unknown(conn, now)
        except Exception:  # noqa: BLE001
            log.exception("social: could not count unknown posts")
            p._rollback()
        try:
            await asyncio.to_thread(p.resolve_urls)
        except Exception:  # noqa: BLE001
            log.exception("social post-URL lookup failed")
            p._rollback()
        for sport in _sports():
            try:
                await p.run_sport(sport)
            except Exception:  # noqa: BLE001
                log.exception("social %s pass failed", sport)
                p._rollback()
    except Exception:  # noqa: BLE001 - social never breaks the worker
        log.exception("social step failed")
        try:
            conn.rollback()
        except Exception:  # noqa: BLE001
            pass
    finally:
        if own_http is not None:
            try:
                await own_http.aclose()
            except Exception:  # noqa: BLE001
                pass
    return p.out if p else {}
