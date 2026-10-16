# backend/tests/unit/test_edge_nfl_players.py
import json
from datetime import date

import httpx
import pytest

from src.edge import nfl_players as nfl
from src.edge.config import NFL_PROP_MARKETS

FIX = "tests/fixtures/edge"


def test_every_nfl_market_has_stat_parts():
    assert set(nfl.NFL_STAT_PARTS) == set(NFL_PROP_MARKETS)
    g = {"pass_att": 30, "pass_cmp": 20, "pass_yds": 250, "pass_tds": 2, "pass_int": 1,
         "rush_att": 5, "rush_yds": 22, "receptions": 0, "rec_yds": 0}
    assert nfl.stat_for("player_rush_reception_yds", {**g, "rec_yds": 8}) == 30
    assert nfl.stat_for("player_pass_completions", g) == 20


def test_game_and_stat_rows_from_the_real_fixtures():
    games = json.load(open(f"{FIX}/bdl_nfl_games_w5.json"))["data"]
    gr = nfl.game_row(games[0])
    assert gr["bdl_game_id"] == games[0]["id"] and gr["week"] == 5 and gr["season"] == 2026
    assert gr["home"] == games[0]["home_team"]["abbreviation"] and isinstance(gr["game_date"], date)
    stats = json.load(open(f"{FIX}/bdl_nfl_stats_game.json"))["data"]
    rows = [r for r in (nfl.stat_row(s) for s in stats) if r]
    assert len(rows) == len(stats)
    qb = max(rows, key=lambda r: r["pass_att"])
    assert qb["pass_yds"] > 0 and qb["opponent"] in {"TB", "DAL"} and qb["opponent"] != qb["team"]
    assert all(isinstance(r[c], int) for r in rows for c in nfl.STAT_COLS)  # NULL -> 0


@pytest.mark.asyncio
async def test_client_follows_cursors_and_uses_the_nfl_base():
    seen = []

    def handler(request):
        seen.append(str(request.url))
        if "players" in request.url.path and "cursor" not in request.url.params:
            return httpx.Response(200, json={"data": [{"id": 1, "first_name": "A", "last_name": "B",
                                                       "position_abbreviation": "WR", "team": {"abbreviation": "ATL"}}],
                                             "meta": {"next_cursor": 9}})
        if "players" in request.url.path:
            return httpx.Response(200, json={"data": [{"id": 2, "first_name": "C", "last_name": "D",
                                                       "position_abbreviation": "RB", "team": {"abbreviation": "ATL"}}],
                                             "meta": {}})
        if "teams" in request.url.path:
            return httpx.Response(200, json={"data": [{"id": 7, "abbreviation": "ATL"}]})
        return httpx.Response(200, json={"data": [], "meta": {}})

    c = nfl.BdlNflClient(api_key="k", transport=httpx.MockTransport(handler), min_interval=0)
    roster = await c.roster("ATL")
    assert [p["id"] for p in roster] == [1, 2] and roster[0]["position"] == "WR"
    await c.stats([11, 12])
    assert all("/nfl/v1/" in u for u in seen)
    assert "game_ids%5B%5D=11" in seen[-1] and "game_ids%5B%5D=12" in seen[-1]
