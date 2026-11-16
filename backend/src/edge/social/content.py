# backend/src/edge/social/content.py
"""Post text and the data behind it: templates, slate date, second-source scores, posted plays."""
from __future__ import annotations

from datetime import date, datetime, timedelta

from psycopg2.extras import RealDictCursor

from src.config import settings
from src.edge import nfl_players, nightly, players, views
from src.edge.grading import profit_units
from src.edge.players import ET, et_date
from src.edge.social.safety import tip_of
from src.edge.teams import ABBR_BY_NAME, NFL_ABBR_BY_NAME

CHEATSHEET_MIN, CHEATSHEET_MAX = 6, 10
SEASON_START = {"nba": date(2026, 10, 20), "nfl": date(2026, 9, 1)}
_NICK_OVERRIDES = {"Portland Trail Blazers": "TrailBlazers"}


def slate_for(commence: datetime) -> date:
    """The slate a game belongs to: the ET date of its tip (a game that crosses midnight stays put)."""
    return nightly.slate_date(commence)


# ---- links and text -------------------------------------------------------------------------

def link(path: str, campaign: str) -> str:
    return f"{settings.site_url.rstrip('/')}{path}?utm_source=x&utm_medium=social&utm_campaign={campaign}"


def _price(p: int) -> str:
    return f"+{p}" if p > 0 else str(p)


def blind_text(sport: str, n: int) -> str:
    return f"🙈 Blind Play #{n} · {sport.upper()}\n\nNo names. No teams. No bias. Who is it? 👇"


def blind_reply(sport: str, n: int) -> str:
    return f"Reveal + full breakdown → {link(f'/blind/{sport.lower()}/{n}', 'blind')}"


def cheatsheet_text(sport: str, day: date) -> str:
    return (f"📋 {sport.upper()} Cheat Sheet · {day.strftime('%a')} {day.month}/{day.day}\n\n"
            "Tonight's hottest props by last-10 hit rate, all priced as value on the board right now.")


def value_text(sport: str, locked_at: datetime) -> str:
    t = locked_at.astimezone(ET)
    return (f"💰 Tonight's Top 5 value plays ({sport.upper()})\n\n"
            "Books paying more than the market says these are worth. "
            f"Locked {t.strftime('%I:%M %p').lstrip('0')} ET.\n\n"
            f"All plays + odds → {link('/', 'value')}")


def nickname(full_name: str) -> str:
    """Hashtag nickname from a full team name: the name minus its city ('Boston Celtics' -> 'Celtics')."""
    return _NICK_OVERRIDES.get(full_name) or full_name.split()[-1]


def final_text(sport: str, event: dict, w: int, l: int, pushes: int, units: float) -> str:
    rec = f"{w}-{l}" + (f"-{pushes}" if pushes > 0 else "")
    tail = f"{rec} {'✅ ' if round(units, 2) > 0 else ''}{units:+.2f}u"
    away, home = event["away_team"], event["home_team"]
    return (f"FINAL\n\n{away} {event['away_score']}, {home} {event['home_score']}\n\n"
            f"TruLine plays in this game: {tail}\n\n"
            f"#{sport.upper()} #{nickname(away)} #{nickname(home)}")


def blind_receipt_text(n: int, name: str, price: int, sport: str, post_url: str | None = None) -> str:
    url = post_url or link(f"/blind/{sport.lower()}/{n}", "receipt")
    return (f"✅ CASHED. Blind Play #{n} was {name}.\n\n"
            f"Called it before tip at {_price(price)}. Did you guess right?\n\n{url}")


def value_receipt_text(bet: str, price: int) -> str:
    return f"✅ CASHED. {bet} at {_price(price)} from tonight's Top 5."


# ---- cheat sheet ----------------------------------------------------------------------------

FORM_GAMES, MIN_FORM_GAMES = 10, 5


def nfl_recent_form(games: list[dict], market: str, line: float, side: str, n: int = FORM_GAMES) -> dict | None:
    """NFL twin of players.recent_form: how often the bet side won in the player's last n games
    (nfl_players.load_games rows: box scores plus zero rows for snaps without a stat). Pushes are
    left out of `games`. None below 5 games or for a market box scores can't grade."""
    if market not in nfl_players.NFL_STAT_PARTS:
        return None
    played = sorted(games, key=lambda g: g["game_date"], reverse=True)[:n]
    if len(played) < MIN_FORM_GAMES:
        return None
    stats = [nfl_players.stat_for(market, g) for g in played]
    hits = sum(1 for v in stats if (v > line if side == "over" else v < line))
    decided = sum(1 for v in stats if v != line)
    return {"hits": hits, "games": decided, "rate": round(hits / decided, 3)} if decided else None


def cheatsheet_rows(rows: list[dict], games_by_player: dict[int, list[dict]], sport: str = "nba") -> list[dict]:
    """+EV player props ranked by last-10 hit rate then EV. Each row's `l10` is recomputed from the
    player's box scores (NBA edge_player_games, NFL nfl_players.load_games rows). Fewer than 6
    usable rows means no post: [] (a thin sheet looks bad)."""
    form_of = nfl_recent_form if sport == "nfl" else players.recent_form
    out = []
    for r in rows:
        if r["ev"] <= 0 or not r["market"].startswith("player_") or r.get("verify") or not r.get("bdl_player_id"):
            continue
        form = form_of(games_by_player.get(r["bdl_player_id"], []), r["market"], float(r["line"]), r["side"])
        if form is None or form["rate"] is None:
            continue
        out.append({**r, "l10": form})
    out.sort(key=lambda r: (r["l10"]["rate"], r["ev"]), reverse=True)
    out = out[:CHEATSHEET_MAX]
    return out if len(out) >= CHEATSHEET_MIN else []


# ---- second sources -------------------------------------------------------------------------

def second_source_nba(conn, event: dict) -> tuple[int, int] | None:
    """(home, away) points summed from stored box scores (Final games only) for the tip's ET date."""
    home, away = ABBR_BY_NAME.get(event["home_team"]), ABBR_BY_NAME.get(event["away_team"])
    if not home or not away:
        return None
    with conn.cursor() as cur:
        cur.execute("""SELECT team, SUM(pts) FROM edge_player_games
                       WHERE game_date = %s AND team = ANY(%s) AND opponent = ANY(%s) GROUP BY team""",
                    (et_date(tip_of(event)), [home, away], [home, away]))
        pts = {t: int(p) for t, p in cur.fetchall()}
    conn.rollback()
    return (pts[home], pts[away]) if home in pts and away in pts else None


def second_source_nfl(conn, event: dict) -> tuple[int, int] | None:
    """(home, away) from BDL's final score on edge_nfl_games. BDL dates are UTC, so a night game's
    row can be a day after the ET date: look at both."""
    home, away = NFL_ABBR_BY_NAME.get(event["home_team"]), NFL_ABBR_BY_NAME.get(event["away_team"])
    if not home or not away:
        return None
    day = et_date(tip_of(event))
    with conn.cursor() as cur:
        cur.execute("""SELECT home_score, away_score FROM edge_nfl_games
                       WHERE home = %s AND away = %s AND status_state = 'final'
                         AND game_date BETWEEN %s AND %s AND home_score IS NOT NULL AND away_score IS NOT NULL""",
                    (home, away, day, day + timedelta(days=1)))
        rows = cur.fetchall()
    conn.rollback()
    return (int(rows[0][0]), int(rows[0][1])) if len(rows) == 1 else None


# ---- posted plays ---------------------------------------------------------------------------

_RESULT = """LEFT JOIN LATERAL (
    SELECT o.result FROM edge_opportunities o
    WHERE o.event_id = x.event_id AND o.market = x.market AND o.player = x.player
      AND o.side = x.side AND o.line = x.line AND o.result IS NOT NULL LIMIT 1) res ON true"""


def posted_plays(conn, event_id: str) -> list[dict]:
    """Plays TruLine posted for a game: blind plays and the Top-5 legs (parlay legs are not
    single plays). {kind: blind|value, sport, n, bet, market, player, side, line, price, result,
    profit_units}; result is None until graded."""
    out: list[dict] = []
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(f"""SELECT x.sport, x.n, x.market, x.player, x.side, x.line, x.price, res.result
                        FROM edge_blind_plays x {_RESULT} WHERE x.event_id = %s ORDER BY x.sport, x.n""",
                    (event_id,))
        out += [{**dict(r), "kind": "blind", "bet": views.describe(r["market"], r["player"], r["side"], float(r["line"]), "", "")}
                for r in cur.fetchall()]
        cur.execute(f"""SELECT x.sport, x.rank AS n, x.market, x.player, x.side, x.line, x.price, x.bet,
                               res.result FROM edge_nightly_legs x {_RESULT}
                        WHERE x.event_id = %s AND x.kind = 'top' ORDER BY x.slate_date, x.rank""", (event_id,))
        out += [{**dict(r), "kind": "value"} for r in cur.fetchall()]
    conn.rollback()
    seen: set[tuple] = set()
    deduped = []
    for p in out:  # blind rows come first, so a bet that is also a Top-5 leg keeps the blind row
        key = (p["market"], p["player"], p["side"], float(p["line"]))
        if key not in seen:
            seen.add(key)
            deduped.append(p)
    out = deduped
    for p in out:
        p["line"], p["price"] = float(p["line"]), int(p["price"])
        p["profit_units"] = round(profit_units(p["result"], p["price"]), 4) if p["result"] else None
    return out


def record_line(plays: list[dict]) -> tuple[int, int, int, float]:
    """(wins, losses, pushes, units) over graded plays; void and ungraded plays are left out."""
    graded = [p for p in plays if p.get("result") in ("win", "loss", "push")]
    return (sum(p["result"] == "win" for p in graded), sum(p["result"] == "loss" for p in graded),
            sum(p["result"] == "push" for p in graded), round(sum(p["profit_units"] or 0.0 for p in graded), 4))


# ---- team records ---------------------------------------------------------------------------

def team_records(conn, sport: str, season_start: date) -> dict[str, dict]:
    """W-L(-T) per team abbreviation from completed edge_events (both scores present) since
    `season_start`. Postponed/cancelled/suspended rows never count."""
    abbr = ABBR_BY_NAME if sport == "nba" else NFL_ABBR_BY_NAME
    key = "basketball_nba" if sport == "nba" else "americanfootball_nfl"
    start = datetime(season_start.year, season_start.month, season_start.day, tzinfo=ET)
    with conn.cursor() as cur:
        cur.execute("""SELECT home_team, away_team, home_score, away_score FROM edge_events
                       WHERE sport = %s AND completed AND home_score IS NOT NULL AND away_score IS NOT NULL
                         AND commence_time >= %s AND status !~* '(postponed|cancel|suspended)'""", (key, start))
        games = cur.fetchall()
    conn.rollback()
    out: dict[str, dict] = {}
    for home, away, hs, a_s in games:
        for name, mine, theirs in ((home, hs, a_s), (away, a_s, hs)):
            t = abbr.get(name)
            if not t:
                continue
            r = out.setdefault(t, {"w": 0, "l": 0, "t": 0, "games": 0})
            r["games"] += 1
            r["w" if mine > theirs else "l" if mine < theirs else "t"] += 1
    for r in out.values():
        r["label"] = f"{r['w']}-{r['l']}" + (f"-{r['t']}" if r["t"] else "")
    return out
