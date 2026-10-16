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
from datetime import date

import httpx

from src.config import settings
from src.edge.players import BdlError

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
