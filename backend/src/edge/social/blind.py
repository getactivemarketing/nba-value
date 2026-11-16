# backend/src/edge/social/blind.py
"""Blind plays: +EV props posted to X with the player and team hidden, revealed on the site.

select() is pure. build_panels() is the DB side (matchup context). record() numbers and stores
plays in edge_blind_plays; listing()/detail() serve the public reveal API. The BlindCard is what
the X image draws and has no identity fields; the API returns the name (the reveal is the point).
"""
from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

from psycopg2.extras import Json, RealDictCursor

from src.edge import matchups, matchups_store, nightly, players, views
from src.edge.engine import implied_prob
from src.edge.grading import profit_units
from src.edge.players import ET, et_date
from src.edge.social import headshots, nba_matchup, nfl_matchup
from src.edge.tagging import ALL_STORED_WEEKS, event_teams
from src.edge.teams import ABBR_BY_NAME, NFL_ABBR_BY_NAME
from src.utils.season import current_nba_season, current_nfl_season

log = logging.getLogger(__name__)

MIN_BEFORE_TIP = timedelta(minutes=60)
MAX_PLAYS = 3
MAX_PER_SLATE = 3  # plays per (sport, slate_date), across ticks
LIST_LIMIT = 60
BOOK_NAMES = {
    "draftkings": "DraftKings", "fanduel": "FanDuel", "betmgm": "BetMGM", "williamhill_us": "Caesars",
    "espnbet": "ESPN BET", "fanatics": "Fanatics", "betrivers": "BetRivers", "hardrockbet": "Hard Rock Bet",
    "betonlineag": "BetOnline", "bovada": "Bovada", "lowvig": "LowVig", "pinnacle": "Pinnacle",
    "prizepicks": "PrizePicks", "underdog": "Underdog",
}


def panel_key(r: dict) -> tuple:
    """(event_id, market, player, side, line): the key of the `panels` dict."""
    return (r["event_id"], r["market"], r["player"], r["side"], float(r["line"]))


def _tip(r: dict) -> datetime:
    t = r["commence_time"]
    return datetime.fromisoformat(t) if isinstance(t, str) else t


# ---- pure selection -------------------------------------------------------------------------

def select(rows: list[dict], top5_players: set[str], panels: dict[tuple, object], now: datetime,
           n: int = MAX_PLAYS) -> list[dict]:
    """Best-edge props with a favorable matchup, not already in tonight's top 5, at least an hour
    before the tip; one per game and per player. A row with no panel is never favorable."""
    out = []
    for r in nightly.eligible(rows):
        if not r["market"].startswith("player_") or r["player"] in top5_players:
            continue
        p = panels.get(panel_key(r))
        if p is None or not p.favorable:
            continue
        if _tip(r) - now < MIN_BEFORE_TIP:
            continue
        out.append(r)
    return nightly.one_per_game(out, n)


# ---- matchup context (DB) -------------------------------------------------------------------

def _teams_of(r: dict) -> tuple[str, str]:
    away, _, home = r["matchup"].partition(" @ ")
    return home, away


def _l10_avg(games: list[dict], market: str) -> float | None:
    played = sorted((g for g in games if (g["minutes"] or 0) > 0), key=lambda g: g["game_date"], reverse=True)[:10]
    return round(sum(players.stat_for(market, g) for g in played) / len(played), 1) if len(played) >= 5 else None


def build_panels(conn, sport: str, rows: list[dict], now: datetime) -> dict[tuple, object]:
    """Matchup panel per prop row, keyed by panel_key. Fails closed: a row whose context can't be
    built gets no panel. Also annotates each row in place with home_team, away_team, team and
    position (and form["avg"] for NBA), which record() and the card need."""
    props = [r for r in rows if r["player"] and r["market"].startswith("player_")]
    for r in props:
        r["home_team"], r["away_team"] = _teams_of(r)
    try:
        if sport == "nba":
            return _nba_panels(conn, props, now)
        if sport == "nfl":
            return _nfl_panels(conn, props, now)
    except Exception:  # noqa: BLE001 - no context, no blind plays
        log.exception("blind panels unavailable for %s", sport)
        try:
            conn.rollback()
        except Exception:  # noqa: BLE001
            pass
    return {}


def _nba_panels(conn, props: list[dict], now: datetime) -> dict[tuple, object]:
    ranks, usage_for = nba_matchup.load_context(conn, current_nba_season(now.date()))
    with conn.cursor() as cur:
        cur.execute("SELECT odds_name, team, position FROM edge_players WHERE odds_name = ANY(%s)",
                    (sorted({r["player"] for r in props}),))
        known = {n: (t, pos) for n, t, pos in cur.fetchall()}
    conn.commit()
    recent = players.load_recent_games(conn, sorted({r["bdl_player_id"] for r in props if r.get("bdl_player_id")}))
    out = {}
    for r in props:
        team, pos = known.get(r["player"], (None, None))
        home, away = ABBR_BY_NAME.get(r["home_team"]), ABBR_BY_NAME.get(r["away_team"])
        if team not in (home, away) or not pos or not r.get("bdl_player_id"):
            continue
        r["team"], r["position"] = team, pos
        if r.get("form"):
            r["form"] = {**r["form"], "avg": _l10_avg(recent.get(r["bdl_player_id"], []), r["market"])}
        opp, usage = ranks.get(away if team == home else home), usage_for(r["bdl_player_id"])
        if opp is None or usage is None:
            continue
        out[panel_key(r)] = nba_matchup.panel(r, pos, opp, usage)
    return out


def _nfl_panels(conn, props: list[dict], now: datetime) -> dict[tuple, object]:
    season = current_nfl_season(now.date())
    team_games, player_games = matchups_store.load(conn, [season - 1, season])
    ctx = matchups.build_context(team_games, player_games, {}, season, ALL_STORED_WEEKS)
    out = {}
    for r in props:
        teams = event_teams(r["home_team"], r["away_team"])
        who = matchups.resolve_player(ctx, r["player"], teams) if teams else None
        if who is None:
            continue
        r["team"], r["position"] = who[1], who[2]
        out[panel_key(r)] = nfl_matchup.panel(r, ctx)
    return out


# ---- storage --------------------------------------------------------------------------------

def record(conn, sport: str, picks: list[dict], panels: dict[tuple, object], now: datetime) -> list[int]:
    """Store picks as the next play numbers of `sport` and return the new numbers. Skipped: a bet
    already recorded (any book), a game or a player that already has a play that slate, and anything
    past MAX_PER_SLATE plays for the slate, so repeated ticks never exceed the daily limits."""
    new: list[int] = []
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (f"edge_blind_plays:{sport}",))
            for r in picks:
                tip = _tip(r)
                day = et_date(tip)
                cur.execute("""SELECT
                    count(*) FILTER (WHERE slate_date = %s),
                    count(*) FILTER (WHERE event_id = %s),
                    count(*) FILTER (WHERE slate_date = %s AND player = %s),
                    count(*) FILTER (WHERE event_id = %s AND market = %s AND player = %s AND side = %s
                                     AND line = %s)
                    FROM edge_blind_plays WHERE sport = %s""",
                            (day, r["event_id"], day, r["player"], r["event_id"], r["market"], r["player"],
                             r["side"], r["line"], sport))
                slate, game, player, same = cur.fetchone()
                if slate >= MAX_PER_SLATE or game or player or same:
                    continue
                cur.execute("SELECT COALESCE(MAX(n), 0) + 1 FROM edge_blind_plays WHERE sport = %s", (sport,))
                n = cur.fetchone()[0]
                panel = panels.get(panel_key(r))
                cur.execute(
                    """INSERT INTO edge_blind_plays (sport, n, slate_date, event_id, market, market_label,
                           player, side, line, book, price, fair_prob, ev, n_books, form, position, panel,
                           commence_time, created_at)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                    (sport, n, day, r["event_id"], r["market"], r["market_label"], r["player"], r["side"],
                     r["line"], r["book"], r["price"], r["fair_prob"], r["ev"], r["n_books"],
                     Json(r["form"]) if r.get("form") else None, r.get("position"),
                     Json(dataclasses.asdict(panel)) if panel is not None else None, tip, now))
                new.append(n)
    except Exception:
        conn.rollback()
        raise
    conn.commit()
    return new


# ---- the card (what X sees) -----------------------------------------------------------------

@dataclass(frozen=True)
class BlindCard:
    sport: str
    n: int
    position: str | None
    market_label: str
    side: str
    line: float
    price: int
    ev: float
    n_books: int
    l10_hits: int | None
    l10_games: int | None
    l10_avg: float | None
    tip_label: str
    panel: dict | None


def tip_label(tip: datetime, slate_tips: dict[str, datetime]) -> str:
    """ET tip time, unless this is the only game at that time (a time that narrows it to one game).
    `slate_tips` maps event_id -> tip, so props from one game count once."""
    if sum(1 for t in slate_tips.values() if t == tip) <= 1:
        return "Tonight"
    local = tip.astimezone(ET)
    return f"{local.strftime('%I:%M %p').lstrip('0')} ET"


def to_card(play: dict, slate_tips: dict[str, datetime]) -> BlindCard:
    form = play.get("form") or {}
    return BlindCard(
        sport=play["sport"], n=play["n"], position=play.get("position"), market_label=play["market_label"],
        side=play["side"], line=float(play["line"]), price=int(play["price"]), ev=float(play["ev"]),
        n_books=play["n_books"], l10_hits=form.get("hits"), l10_games=form.get("games"),
        l10_avg=form.get("avg"), tip_label=tip_label(_tip(play), slate_tips), panel=play.get("panel"))


def _price(p: int) -> str:
    return f"+{p}" if p > 0 else str(p)


def explain(play: dict) -> str:
    book = BOOK_NAMES.get(play["book"], play["book"])
    return (f"{play['n_books']} books price this at {round(float(play['fair_prob']) * 100)}%; "
            f"{book} pays {_price(int(play['price']))}, which needs only "
            f"{round(implied_prob(int(play['price'])) * 100)}%.")


# ---- reading (reveal API) -------------------------------------------------------------------

_RESULT_JOIN = """LEFT JOIN LATERAL (
    SELECT o.result FROM edge_opportunities o
    WHERE o.event_id = b.event_id AND o.market = b.market AND o.player = b.player
      AND o.side = b.side AND o.line = b.line AND o.result IS NOT NULL LIMIT 1) res ON true"""


def result(conn, sport: str, n: int) -> dict:
    """Grading of a play: the same bet at any book has the same result."""
    with conn.cursor() as cur:
        cur.execute(f"""SELECT res.result, b.price FROM edge_blind_plays b {_RESULT_JOIN}
                        WHERE b.sport = %s AND b.n = %s""", (sport, n))
        row = cur.fetchone()
    res, price = (row if row else (None, None))
    return {"result": res, "profit_units": round(profit_units(res, int(price)), 4) if res else None}


def _plays(conn, sport: str, n: int | None, limit: int) -> list[dict]:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(f"""SELECT b.*, e.home_team, e.away_team, res.result
                        FROM edge_blind_plays b LEFT JOIN edge_events e ON e.event_id = b.event_id
                        {_RESULT_JOIN}
                        WHERE b.sport = %s AND (%s::int IS NULL OR b.n = %s) ORDER BY b.n DESC LIMIT %s""",
                    (sport, n, n, limit))
        return [dict(r) for r in cur.fetchall()]


def _identity(conn, sport: str, plays: list[dict]) -> dict[int, tuple[str | None, int | None]]:
    """{play n: (team, espn_id)} for the plays' players. Both are None unless the roster team and
    the headshot row's team agree and are one of the game's teams: never a wrong face."""
    names = sorted({p["player"] for p in plays})
    if not names:
        return {}
    table = "edge_players" if sport == "nba" else "edge_nfl_players"
    with conn.cursor() as cur:
        cur.execute(f"SELECT odds_name, team FROM {table} WHERE odds_name = ANY(%s)", (names,))
        rosters: dict[str, set[str]] = {}
        for name, team in cur.fetchall():
            rosters.setdefault(name, set()).add(team)
        cur.execute("SELECT odds_name, team, espn_id FROM edge_headshots WHERE sport = %s AND odds_name = ANY(%s)",
                    (sport, names))
        shots = {name: (team, eid) for name, team, eid in cur.fetchall()}
    conn.rollback()
    abbr = ABBR_BY_NAME if sport == "nba" else NFL_ABBR_BY_NAME
    out: dict[int, tuple[str | None, int | None]] = {}
    for p in plays:
        in_game = {abbr.get(p["home_team"]), abbr.get(p["away_team"])}
        cands = rosters.get(p["player"], set()) & in_game
        team = next(iter(cands)) if len(cands) == 1 else None
        shot = shots.get(p["player"])
        out[p["n"]] = (team, shot[1] if shot and team and shot[0] == team else None)
    return out


def _logo(sport: str, name: str | None) -> str | None:
    abbr = (ABBR_BY_NAME if sport == "nba" else NFL_ABBR_BY_NAME).get(name or "")
    espn = headshots.espn_abbr(sport, abbr) if abbr else None
    return headshots.LOGO_URL.format(league=sport, abbr=espn) if espn else None


def _out(p: dict, who: tuple[str | None, int | None]) -> dict:
    team, espn_id = who
    sport, res = p["sport"], p["result"]
    return {
        "sport": sport, "n": p["n"], "slate_date": p["slate_date"].isoformat(), "event_id": p["event_id"],
        "player": p["player"], "team": team, "home_team": p["home_team"], "away_team": p["away_team"],
        "bet": views.describe(p["market"], p["player"], p["side"], float(p["line"]),
                              p["home_team"] or "", p["away_team"] or ""),
        "market": p["market"], "market_label": p["market_label"], "side": p["side"], "line": float(p["line"]),
        "book": p["book"], "price": int(p["price"]), "fair_prob": float(p["fair_prob"]), "ev": float(p["ev"]),
        "n_books": p["n_books"], "form": p["form"], "position": p["position"], "panel": p["panel"],
        "commence_time": p["commence_time"].astimezone(timezone.utc).isoformat(),
        "explanation": explain(p), "result": res,
        "profit_units": round(profit_units(res, int(p["price"])), 4) if res else None,
        "headshot_url": headshots.HEADSHOT_URL.format(league=sport, espn_id=espn_id) if espn_id else None,
        "home_logo_url": _logo(sport, p["home_team"]), "away_logo_url": _logo(sport, p["away_team"]),
    }


def _record(conn, sport: str) -> dict:
    with conn.cursor() as cur:
        cur.execute(f"""SELECT res.result, b.price FROM edge_blind_plays b {_RESULT_JOIN}
                        WHERE b.sport = %s""", (sport,))
        rows = [(r, p) for r, p in cur.fetchall() if r in ("win", "loss", "push", "void")]
    conn.rollback()
    return {"n": len(rows), "wins": sum(r == "win" for r, _ in rows), "losses": sum(r == "loss" for r, _ in rows),
            "pushes": sum(r == "push" for r, _ in rows), "voids": sum(r == "void" for r, _ in rows),
            "units": round(sum(profit_units(r, int(p)) for r, p in rows), 2)}


def listing(conn, sport: str) -> dict:
    plays = _plays(conn, sport, None, LIST_LIMIT)
    who = _identity(conn, sport, plays)
    return {"plays": [_out(p, who[p["n"]]) for p in plays], "record": _record(conn, sport)}


def detail(conn, sport: str, n: int) -> dict | None:
    plays = _plays(conn, sport, n, 1)
    if not plays:
        return None
    return _out(plays[0], _identity(conn, sport, plays)[plays[0]["n"]])
