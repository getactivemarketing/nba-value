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


def test_nfl_prop_result_rules():
    opp = {"market": "player_reception_yds", "line": 40.5, "side": "over"}
    row = {c: 0 for c in nfl.STAT_COLS} | {"rec_yds": 55}
    assert nfl.nfl_prop_result(opp, row, snaps_loaded=False, played=False) == "win"     # rule 3
    assert nfl.nfl_prop_result(opp, None, snaps_loaded=True, played=True) == "loss"     # rule 4: played, 0 yards
    assert nfl.nfl_prop_result({**opp, "side": "under"}, None, True, True) == "win"
    assert nfl.nfl_prop_result(opp, None, snaps_loaded=True, played=False) == "void"    # rule 5
    assert nfl.nfl_prop_result(opp, None, snaps_loaded=False, played=False) is None     # rule 6: wait


def test_nfl_hit_rates_count_zero_games():
    rows = [{"season": 2026, "week": w, "game_date": date(2026, 9, 7 * w), "opponent": "TB" if w == 3 else "NO",
             "is_home": w % 2 == 0, **{c: 0 for c in nfl.STAT_COLS}, "rec_yds": y}
            for w, y in ((1, 80), (2, 0), (3, 45), (4, 60))]
    hr = nfl.hit_rates(rows, "player_reception_yds", 50.5, "TB", True, 2026)
    assert hr["l5"] == {"hits": 2, "misses": 2, "pushes": 0, "games": 4, "rate": 0.5}  # the 0 counts
    assert hr["vs_opponent"]["games"] == 1 and hr["home_away"]["games"] == 2


def _bdl_stat(**extra):
    return {"player": {"id": 7}, "team": {"abbreviation": "KC"},
            "game": {"id": 99, "season": 2026, "week": 6, "date": "2026-10-18T17:00:00.000Z",
                     "home_team": {"abbreviation": "KC"}, "visitor_team": {"abbreviation": "LV"}}, **extra}


def test_stat_row_reads_td_fields_and_sums_return_tds():
    r = nfl.stat_row(_bdl_stat(rushing_touchdowns=2, receiving_touchdowns=1,
                               kick_return_touchdowns=1, punt_return_touchdowns=1))
    assert (r["rush_tds"], r["rec_tds"], r["ret_tds"]) == (2, 1, 2)
    r = nfl.stat_row(_bdl_stat(kick_return_touchdowns=None))
    assert (r["rush_tds"], r["rec_tds"], r["ret_tds"]) == (0, 0, 0)


def test_anytime_td_grades_rush_rec_return_and_ignores_passing():
    base = {c: 0 for c in nfl.STAT_COLS} | {"pass_tds": 3}
    opp = {"market": "player_anytime_td", "line": 0.5, "side": "over"}
    assert nfl.nfl_prop_result(opp, base, True, True) == "loss"
    assert nfl.nfl_prop_result(opp, base | {"ret_tds": 1}, True, True) == "win"
    assert nfl.nfl_prop_result(opp, base | {"rush_tds": 1}, True, True) == "win"
    assert nfl.nfl_prop_result({**opp, "side": "under"}, base | {"rec_tds": 1}, True, True) == "loss"
    assert nfl.nfl_prop_result({**opp, "side": "under"}, base, True, True) == "win"


def test_anytime_td_inactive_is_void():
    opp = {"market": "player_anytime_td", "line": 0.5, "side": "over"}
    assert nfl.nfl_prop_result(opp, None, snaps_loaded=True, played=False) == "void"


def test_anytime_td_hit_rate_and_zero_rows_use_td_columns():
    games = [{"game_date": date(2026, 10, d), "season": 2026, "opponent": "LV", "is_home": True,
              **{c: 0 for c in nfl.STAT_COLS}, "rec_tds": t} for d, t in ((1, 1), (8, 0), (15, 2))]
    r = nfl.hit_rates(games, "player_anytime_td", 0.5, "LV", True, 2026)["l5"]
    assert (r["hits"], r["misses"], r["games"]) == (2, 1, 3)
