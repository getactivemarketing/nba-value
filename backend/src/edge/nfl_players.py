# backend/src/edge/nfl_players.py
"""NFL player identity, BallDontLie NFL box scores, nflverse snap counts,
hit rates and prop grading (Phase B spec).

Stats come only from games listed by BallDontLie /games (regular season and
playoffs): its /stats?seasons[] feed also carries preseason games.
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import date, datetime, timedelta

import httpx
from psycopg2.extras import RealDictCursor, execute_values

from src.config import settings
from src.edge.players import BdlError, et_date, match_player, normalize_name
from src.edge.quotes import Quote
from src.edge.teams import NFL_ABBR_BY_NAME, NFLVERSE_TO_BDL

log = logging.getLogger("edge.nfl_players")
ROSTER_TTL = 6 * 3600

STAT_COLS = ("pass_att", "pass_cmp", "pass_yds", "pass_tds", "pass_int",
             "rush_att", "rush_yds", "receptions", "rec_yds")
NFL_STAT_PARTS = {
    "player_pass_yds": ("pass_yds",), "player_pass_tds": ("pass_tds",),
    "player_pass_completions": ("pass_cmp",), "player_pass_attempts": ("pass_att",),
    "player_pass_interceptions": ("pass_int",), "player_rush_yds": ("rush_yds",),
    "player_rush_attempts": ("rush_att",), "player_receptions": ("receptions",),
    "player_reception_yds": ("rec_yds",), "player_rush_reception_yds": ("rush_yds", "rec_yds"),
}
_BDL_FIELDS = {"pass_att": "passing_attempts", "pass_cmp": "passing_completions", "pass_yds": "passing_yards",
               "pass_tds": "passing_touchdowns", "pass_int": "passing_interceptions",
               "rush_att": "rushing_attempts", "rush_yds": "rushing_yards",
               "receptions": "receptions", "rec_yds": "receiving_yards"}


def stat_for(market: str, g: dict) -> float:
    return float(sum(g[k] or 0 for k in NFL_STAT_PARTS[market]))


def game_row(g: dict) -> dict:
    return {"bdl_game_id": g["id"], "season": g["season"], "week": g["week"],
            "game_date": date.fromisoformat(g["date"][:10]), "home": g["home_team"]["abbreviation"],
            "away": g["visitor_team"]["abbreviation"], "postseason": bool(g.get("postseason")),
            "status_state": g.get("status_state")}


def stat_row(s: dict) -> dict | None:
    try:
        game, team = s["game"], s["team"]["abbreviation"]
        home, away = game["home_team"]["abbreviation"], game["visitor_team"]["abbreviation"]
        row = {"bdl_player_id": s["player"]["id"], "bdl_game_id": game["id"], "season": game["season"],
               "week": game["week"], "game_date": date.fromisoformat(game["date"][:10]), "team": team,
               "opponent": away if team == home else home, "is_home": team == home}
        row.update({c: int(s.get(f) or 0) for c, f in _BDL_FIELDS.items()})
        return row
    except (KeyError, TypeError, ValueError):
        return None


class BdlNflClient:
    BASE = "https://api.balldontlie.io/nfl/v1"

    def __init__(self, api_key: str | None = None, transport: httpx.AsyncBaseTransport | None = None,
                 min_interval: float = 1.05):
        self.api_key = api_key if api_key is not None else settings.balldontlie_api_key
        self.transport = transport
        self.min_interval = min_interval
        self._last_call = 0.0
        self._teams: dict[int, str] | None = None
        self._rosters: dict[str, tuple[float, list[dict]]] = {}

    async def _get(self, path: str, params) -> dict:
        wait = self._last_call + self.min_interval - time.monotonic()
        if wait > 0:
            await asyncio.sleep(wait)
        self._last_call = time.monotonic()
        try:
            async with httpx.AsyncClient(transport=self.transport, timeout=30.0) as c:
                r = await c.get(f"{self.BASE}/{path}", params=params, headers={"Authorization": self.api_key})
        except httpx.HTTPError as e:
            raise BdlError(0, f"BallDontLie NFL network error on {path}: {type(e).__name__}") from None
        if r.status_code != 200:
            raise BdlError(r.status_code, f"BallDontLie NFL {r.status_code} on {path}")
        return r.json()

    async def _paged(self, path: str, params: list[tuple[str, object]]) -> list[dict]:
        out, cursor = [], None
        while True:
            p = params + [("per_page", 100)] + ([("cursor", cursor)] if cursor else [])
            data = await self._get(path, p)
            out += data["data"]
            cursor = (data.get("meta") or {}).get("next_cursor")
            if not cursor:
                return out

    async def teams(self) -> dict[int, str]:
        if self._teams is None:
            data = await self._get("teams", [])
            self._teams = {t["id"]: t["abbreviation"] for t in data["data"]}
        return self._teams

    async def roster(self, team_abbr: str) -> list[dict]:
        hit = self._rosters.get(team_abbr)
        if hit and time.monotonic() - hit[0] < ROSTER_TTL:
            return hit[1]
        team_id = next((i for i, a in (await self.teams()).items() if a == team_abbr), None)
        if team_id is None:
            return []
        out = [{"id": x["id"], "first_name": x["first_name"], "last_name": x["last_name"],
                "team_abbr": (x.get("team") or {}).get("abbreviation"), "position": x.get("position_abbreviation")}
               for x in await self._paged("players", [("team_ids[]", team_id)])]
        self._rosters[team_abbr] = (time.monotonic(), out)
        return out

    async def games(self, season: int, week: int | None = None) -> list[dict]:
        params: list[tuple[str, object]] = [("seasons[]", season)]
        if week is not None:
            params.append(("weeks[]", week))
        return await self._paged("games", params)

    async def stats(self, game_ids: list[int]) -> list[dict]:
        return await self._paged("stats", [("game_ids[]", g) for g in game_ids])


async def resolve_new_players(conn, bdl: BdlNflClient, quotes: list[Quote], event: dict, now: datetime) -> int:
    """Match unseen (name, team) pairs from one props poll against both teams' rosters.

    A name is 'seen' per team: a traded player gets a fresh match on his new team.
    Roster errors leave names unseen (retried next poll), never 'unmatched'.
    """
    names = sorted({q.player for q in quotes if q.player})
    teams = {NFL_ABBR_BY_NAME.get(event["home_team"]), NFL_ABBR_BY_NAME.get(event["away_team"])} - {None}
    if not names or not teams:
        return 0
    with conn.cursor() as cur:
        cur.execute("SELECT odds_name FROM edge_nfl_players WHERE odds_name = ANY(%s) AND team = ANY(%s)",
                    (names, sorted(teams)))
        known = {r[0] for r in cur.fetchall()}
    new = [n for n in names if n not in known]
    if not new:
        return 0
    try:
        candidates = [c for t in sorted(teams) for c in await bdl.roster(t)]
    except BdlError as e:
        log.warning("edge NFL player resolve skipped: %s", e)
        return 0
    by_id = {c["id"]: c for c in candidates}
    rows = []
    for name in new:
        m = match_player(name, candidates, teams)
        if m:
            rows.append((name, m.team, m.bdl_player_id, by_id[m.bdl_player_id].get("position"),
                         m.method, m.confidence, now))
        else:
            # No roster match: stored once under the first of the two teams (alphabetical), so the
            # `known` check above stops it being retried every poll; listed on /edge/status.
            rows.append((name, sorted(teams)[0], None, None, "unmatched", 0.0, now))
    with conn.cursor() as cur:
        execute_values(cur, """INSERT INTO edge_nfl_players (odds_name, team, bdl_player_id, position,
                               match_method, confidence, updated_at) VALUES %s
                               ON CONFLICT (odds_name, team) DO NOTHING""", rows)
    conn.commit()
    return len(rows)


def player_ids(conn, teams: set[str]) -> dict[str, int]:
    with conn.cursor() as cur:
        cur.execute("""SELECT odds_name, bdl_player_id FROM edge_nfl_players
                       WHERE bdl_player_id IS NOT NULL AND team = ANY(%s)""", (sorted(teams),))
        return {n: pid for n, pid in cur.fetchall()}


STATS_SETTLE = timedelta(hours=2)  # stat corrections land after the whistle


async def refresh_schedule(conn, bdl: BdlNflClient, season: int, now: datetime) -> int:
    rows = [game_row(g) for g in await bdl.games(season)]
    if not rows:
        return 0
    with conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO edge_nfl_games (bdl_game_id, season, week, game_date, home, away, postseason,
                                        status_state, final_seen_at)
            VALUES %s
            ON CONFLICT (bdl_game_id) DO UPDATE SET
                week = EXCLUDED.week, game_date = EXCLUDED.game_date, status_state = EXCLUDED.status_state,
                final_seen_at = COALESCE(edge_nfl_games.final_seen_at, EXCLUDED.final_seen_at)
        """, [(r["bdl_game_id"], r["season"], r["week"], r["game_date"], r["home"], r["away"], r["postseason"],
               r["status_state"], now if r["status_state"] == "final" else None) for r in rows])
    conn.commit()
    return len(rows)


async def load_due_stats(conn, bdl: BdlNflClient, now: datetime, limit: int = 6) -> int:
    """Box scores for final games settled >= 2 h, oldest first, at most `limit` games per call.

    Serves both the hourly step and the backfill: anything final without stats is due.
    """
    with conn.cursor() as cur:
        cur.execute("""SELECT bdl_game_id FROM edge_nfl_games
                       WHERE status_state = 'final' AND stats_loaded_at IS NULL AND final_seen_at <= %s
                       ORDER BY game_date LIMIT %s""", (now - STATS_SETTLE, limit))
        ids = [r[0] for r in cur.fetchall()]
    if not ids:
        return 0
    rows = [r for r in (stat_row(s) for s in await bdl.stats(ids)) if r and r["bdl_game_id"] in ids]
    cols = ["bdl_player_id", "bdl_game_id", "season", "week", "game_date", "team", "opponent", "is_home", *STAT_COLS]
    with conn.cursor() as cur:
        if rows:
            execute_values(cur, f"""INSERT INTO edge_nfl_player_games ({", ".join(cols)}) VALUES %s
                ON CONFLICT (bdl_player_id, bdl_game_id) DO UPDATE SET
                {", ".join(f"{c} = EXCLUDED.{c}" for c in cols[2:])}""",
                           [tuple(r[c] for c in cols) for r in rows])
        cur.execute("UPDATE edge_nfl_games SET stats_loaded_at = %s WHERE bdl_game_id = ANY(%s)", (now, ids))
    conn.commit()
    return len(rows)


def load_snaps(conn, season: int, now: datetime, frame=None) -> int:
    if frame is None:
        import nfl_data_py as nfl_data  # heavy import, only in the worker
        frame = nfl_data.import_snap_counts([season])
    rows: dict[tuple, int] = {}
    for r in frame.itertuples(index=False):
        team = NFLVERSE_TO_BDL.get(r.team, r.team)
        snaps = int((r.offense_snaps or 0) + (r.defense_snaps or 0) + (r.st_snaps or 0))
        if snaps > 0:
            key = (int(r.season), int(r.week), team, normalize_name(r.player))
            rows[key] = rows.get(key, 0) + snaps
    if not rows:
        return 0
    with conn.cursor() as cur:
        execute_values(cur, """INSERT INTO edge_nfl_snaps (season, week, team, player_norm, snaps, loaded_at)
                               VALUES %s ON CONFLICT (season, week, team, player_norm)
                               DO UPDATE SET snaps = EXCLUDED.snaps, loaded_at = EXCLUDED.loaded_at""",
                       [(*k, v, now) for k, v in rows.items()])
        cur.execute("""UPDATE edge_nfl_games g SET snaps_loaded_at = %s
                       WHERE g.snaps_loaded_at IS NULL AND g.season = %s
                         AND EXISTS (SELECT 1 FROM edge_nfl_snaps s
                                     WHERE s.season = g.season AND s.week = g.week AND s.team = g.home)
                         AND EXISTS (SELECT 1 FROM edge_nfl_snaps s
                                     WHERE s.season = g.season AND s.week = g.week AND s.team = g.away)""",
                    (now, season))
    conn.commit()
    return len(rows)
