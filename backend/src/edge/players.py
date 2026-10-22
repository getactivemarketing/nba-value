# backend/src/edge/players.py
"""Player identity, BallDontLie box scores and hit rates (spec §7).

A wrong match is worse than no match — it shows another player's history next
to a price — so fuzzy matching needs >= 0.92 similarity AND a team that is
playing in this event; anything else stays `unmatched` (retried each props poll) until it
matches or gets a hand-checked entry in NAME_ALIASES.
"""
from __future__ import annotations

import argparse
import asyncio
import difflib
import logging
import time
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import httpx
from psycopg2.extras import RealDictCursor, execute_values

from src.config import settings
from src.edge.quotes import Quote
from src.edge.teams import ABBR_BY_NAME
from src.utils.redact import redact

ET = ZoneInfo("America/New_York")
ROSTER_TTL = 6 * 3600
log = logging.getLogger("edge.players")
SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}
FUZZY_MIN = 0.92
# Sportsbook nickname -> BallDontLie legal name (both normalized). Too far apart for the fuzzy
# matcher; each was checked by hand against /players/active (2026-10-21). Still needs the team.
NAME_ALIASES = {
    "alex sarr": "alexandre sarr",
    "herb jones": "herbert jones",
    "ron holland": "ronald holland",
    "ace bailey": "airious bailey",
}
STAT_PARTS = {
    "player_points": ("pts",), "player_rebounds": ("reb",), "player_assists": ("ast",),
    "player_threes": ("fg3m",), "player_points_rebounds_assists": ("pts", "reb", "ast"),
    "player_points_rebounds": ("pts", "reb"), "player_points_assists": ("pts", "ast"),
    "player_rebounds_assists": ("reb", "ast"),
}


def et_date(dt: datetime) -> date:
    return dt.astimezone(ET).date()


def normalize_name(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    s = s.lower().replace(".", "").replace("'", "").replace("-", " ")
    return " ".join(t for t in s.split() if t not in SUFFIXES)


@dataclass(frozen=True)
class Match:
    bdl_player_id: int
    team: str
    method: str
    confidence: float


def match_player(odds_name: str, candidates: list[dict], event_teams: set[str]) -> Match | None:
    playing = [c for c in candidates if c.get("team_abbr") in event_teams]
    for c in playing:
        if f"{c['first_name']} {c['last_name']}" == odds_name:
            return Match(c["id"], c["team_abbr"], "exact", 1.0)
    target = normalize_name(odds_name)
    for c in playing:
        if normalize_name(f"{c['first_name']} {c['last_name']}") == target:
            return Match(c["id"], c["team_abbr"], "normalized", 1.0)
    alias = NAME_ALIASES.get(target)
    for c in playing:
        if alias and normalize_name(f"{c['first_name']} {c['last_name']}") == alias:
            return Match(c["id"], c["team_abbr"], "manual", 1.0)
    best, score = None, 0.0
    for c in playing:
        r = difflib.SequenceMatcher(None, target, normalize_name(f"{c['first_name']} {c['last_name']}")).ratio()
        if r > score:
            best, score = c, r
    if best is not None and score >= FUZZY_MIN:
        return Match(best["id"], best["team_abbr"], "fuzzy", round(score, 3))
    return None


def parse_minutes(v) -> float:
    if not v:
        return 0.0
    s = str(v)
    if ":" in s:
        m, sec = s.split(":", 1)
        return int(m or 0) + int(sec or 0) / 60
    try:
        return float(s)
    except ValueError:
        return 0.0


def box_row(stat: dict, abbr_by_team_id: dict[int, str]) -> dict | None:
    try:
        game, team_id = stat["game"], stat["team"]["id"]
        is_home = game["home_team_id"] == team_id
        opp_id = game["visitor_team_id"] if is_home else game["home_team_id"]
        return {
            "bdl_player_id": stat["player"]["id"], "bdl_game_id": game["id"], "season": game["season"],
            "game_date": date.fromisoformat(game["date"][:10]),
            "team": stat["team"].get("abbreviation") or abbr_by_team_id[team_id],
            "opponent": abbr_by_team_id[opp_id], "is_home": is_home,
            "minutes": parse_minutes(stat.get("min")),
            "pts": stat.get("pts") or 0, "reb": stat.get("reb") or 0, "ast": stat.get("ast") or 0,
            "fg3m": stat.get("fg3m") or 0, "stl": stat.get("stl") or 0, "blk": stat.get("blk") or 0,
            "tov": stat.get("turnover") or 0,
        }
    except (KeyError, TypeError, ValueError):
        return None


def stat_for(market: str, g: dict) -> float:
    return float(sum(g[k] or 0 for k in STAT_PARTS[market]))


def _rate(rows: list[dict], market: str, line: float) -> dict:
    hits = sum(1 for r in rows if stat_for(market, r) > line)
    misses = sum(1 for r in rows if stat_for(market, r) < line)
    decided = hits + misses
    return {"hits": hits, "misses": misses, "pushes": len(rows) - decided, "games": len(rows),
            "rate": round(hits / decided, 3) if decided else None}


def hit_rates(games: list[dict], market: str, line: float, opponent: str | None, is_home: bool | None,
              season: int, min_minutes: float = 0.0) -> dict:
    played = sorted((x for x in games if (x["minutes"] or 0) > 0 and (x["minutes"] or 0) >= min_minutes),
                    key=lambda x: x["game_date"], reverse=True)
    return {
        "l5": _rate(played[:5], market, line),
        "l10": _rate(played[:10], market, line),
        "l20": _rate(played[:20], market, line),
        "season": _rate([x for x in played if x["season"] == season], market, line),
        "vs_opponent": _rate([x for x in played if opponent and x["opponent"] == opponent], market, line),
        "home_away": _rate([x for x in played if is_home is not None and x["is_home"] == is_home], market, line),
    }


class BdlError(Exception):
    def __init__(self, status: int, message: str):
        self.status = status
        super().__init__(redact(message))


class BdlClient:
    BASE = "https://api.balldontlie.io/v1"

    def __init__(self, api_key: str | None = None, transport: httpx.AsyncBaseTransport | None = None,
                 min_interval: float = 1.05):
        self.api_key = api_key if api_key is not None else settings.balldontlie_api_key
        self.transport = transport
        self.min_interval = min_interval  # ALL-STAR allows 60 req/min
        self._last_call = 0.0
        self._teams: dict[int, str] | None = None
        self._rosters: dict[str, tuple[float, list[dict]]] = {}

    async def _get(self, path: str, params: dict) -> dict:
        wait = self._last_call + self.min_interval - time.monotonic()
        if wait > 0:
            await asyncio.sleep(wait)
        self._last_call = time.monotonic()
        try:
            async with httpx.AsyncClient(transport=self.transport, timeout=30.0) as c:
                r = await c.get(f"{self.BASE}/{path}", params=params, headers={"Authorization": self.api_key})
        except httpx.HTTPError as e:
            raise BdlError(0, f"BallDontLie network error on {path}: {type(e).__name__}") from None
        if r.status_code != 200:
            raise BdlError(r.status_code, f"BallDontLie {r.status_code} on {path}")
        return r.json()

    async def roster(self, team_abbr: str) -> list[dict]:
        """Current players on one team, cached for ROSTER_TTL seconds.

        One roster call per team replaces one search per player name, which
        matters: opening night brings ~200 new names, and BallDontLie allows
        5 requests/min (free) or 60/min (ALL-STAR).
        """
        hit = self._rosters.get(team_abbr)
        if hit and time.monotonic() - hit[0] < ROSTER_TTL:
            return hit[1]
        team_id = next((i for i, a in (await self.teams()).items() if a == team_abbr), None)
        if team_id is None:
            return []
        out, cursor = [], None
        while True:
            params = {"team_ids[]": team_id, "per_page": 100}
            if cursor:
                params["cursor"] = cursor
            # /players/active, not /players: the latter is every player whose last team was this franchise.
            data = await self._get("players/active", params)
            out += [{"id": x["id"], "first_name": x["first_name"], "last_name": x["last_name"],
                     "team_abbr": (x.get("team") or {}).get("abbreviation")} for x in data["data"]]
            cursor = (data.get("meta") or {}).get("next_cursor")
            if not cursor:
                break
        self._rosters[team_abbr] = (time.monotonic(), out)
        return out

    async def teams(self) -> dict[int, str]:
        if self._teams is None:
            data = await self._get("teams", {"per_page": 100})
            self._teams = {t["id"]: t["abbreviation"] for t in data["data"]}
        return self._teams

    async def stats_for_date(self, d: date) -> list[dict]:
        out, cursor = [], None
        while True:
            params = {"dates[]": d.isoformat(), "per_page": 100}
            if cursor:
                params["cursor"] = cursor
            data = await self._get("stats", params)
            out += data["data"]
            cursor = (data.get("meta") or {}).get("next_cursor")
            if not cursor:
                return out


async def resolve_new_players(conn, bdl: BdlClient, quotes: list[Quote], event: dict, now: datetime) -> int:
    names = sorted({q.player for q in quotes if q.player})
    if not names:
        return 0
    with conn.cursor() as cur:
        # 'unmatched' names are retried on every poll (rosters are cached), so a later alias or a
        # late roster addition fixes them; matched names are never looked up again.
        cur.execute("SELECT odds_name FROM edge_players WHERE odds_name = ANY(%s) AND match_method <> 'unmatched'",
                    (names,))
        known = {r[0] for r in cur.fetchall()}
    new = [n for n in names if n not in known]
    if not new:
        return 0
    teams = {ABBR_BY_NAME.get(event["home_team"]), ABBR_BY_NAME.get(event["away_team"])} - {None}
    try:
        candidates = [c for t in sorted(teams) for c in await bdl.roster(t)]
    except BdlError as e:
        # Rate-limited or down: leave these names unresolved; the next props
        # poll retries them. Never let player lookup break odds ingestion.
        log.warning("edge player resolve skipped: %s", e)
        return 0
    rows = []
    for name in new:
        m = match_player(name, candidates, teams)
        rows.append((name, m.bdl_player_id if m else None, m.team if m else None,
                     m.method if m else "unmatched", m.confidence if m else 0.0, now))
    if rows:
        with conn.cursor() as cur:
            execute_values(cur, """INSERT INTO edge_players (odds_name, bdl_player_id, team, match_method,
                                   confidence, updated_at) VALUES %s
                                   ON CONFLICT (odds_name) DO UPDATE SET bdl_player_id = EXCLUDED.bdl_player_id,
                                   team = EXCLUDED.team, match_method = EXCLUDED.match_method,
                                   confidence = EXCLUDED.confidence, updated_at = EXCLUDED.updated_at
                                   WHERE edge_players.match_method = 'unmatched'""", rows)
        conn.commit()
    return len(rows)


async def ingest_box_scores(conn, bdl: BdlClient, d: date) -> int:
    abbr = await bdl.teams()
    rows = [r for r in (box_row(s, abbr) for s in await bdl.stats_for_date(d)) if r]
    if not rows:
        return 0
    cols = list(rows[0].keys())
    with conn.cursor() as cur:
        execute_values(cur, f"""INSERT INTO edge_player_games ({", ".join(cols)}) VALUES %s
            ON CONFLICT (bdl_player_id, game_date) DO UPDATE SET
            {", ".join(f"{c} = EXCLUDED.{c}" for c in cols if c not in ("bdl_player_id", "game_date"))}""",
                       [tuple(r[c] for c in cols) for r in rows])
    conn.commit()
    return len(rows)


def load_games(conn, bdl_player_id: int, limit: int = 82) -> list[dict]:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""SELECT * FROM edge_player_games WHERE bdl_player_id = %s
                       ORDER BY game_date DESC LIMIT %s""", (bdl_player_id, limit))
        return [dict(r) for r in cur.fetchall()]


async def backfill(conn, bdl: BdlClient, start: date, end: date) -> int:
    d, total = start, 0
    while d <= end:
        n = await ingest_box_scores(conn, bdl, d)
        total += n
        print(f"{d} {n} rows (total {total})", flush=True)
        d += timedelta(days=1)  # BdlClient spaces requests itself
    return total


if __name__ == "__main__":
    from src.edge import store
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["backfill"])
    ap.add_argument("start", type=date.fromisoformat)
    ap.add_argument("end", type=date.fromisoformat)
    a = ap.parse_args()
    asyncio.run(backfill(store.connect(), BdlClient(), a.start, a.end))
