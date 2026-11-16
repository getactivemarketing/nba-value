from datetime import date, timedelta

from src.edge.social import nba_matchup as m

PRA = "player_points_rebounds_assists"
TEAMS = [f"T{i:02d}" for i in range(30)]


def _row(gid, team, opp, pos, pts, minutes=30, d=date(2026, 11, 1), pid=1, **kw):
    base = dict(bdl_game_id=gid, team=team, opponent=opp, position=pos, pts=pts, reb=0, ast=0, fg3m=0,
                tov=0, fga=10, fta=2, oreb=1, minutes=minutes, game_date=d, bdl_player_id=pid)
    return {**base, **kw}


def _league(games_each=5, n=30):
    """Teams T00..T(n-1) each play `games_each` games vs a fixed neighbour; each side has one F and one G row."""
    rows, gid = [], 0
    for i, t in enumerate(TEAMS[:n]):
        opp = TEAMS[(i + 1) % n]
        for _ in range(games_each):
            gid += 1
            rows.append(_row(gid, t, opp, "F", 20, pid=100 + i))
            rows.append(_row(gid, t, opp, "G", 10, pid=200 + i))
            rows.append(_row(gid, opp, t, "F", 20, pid=100 + (i + 1) % n))
            rows.append(_row(gid, opp, t, "G", 10, pid=200 + (i + 1) % n))
    return rows


def test_team_allowing_most_forward_pra_ranks_30():
    rows = _league(n=29)  # 29 teams + WAS = a 30-team league
    # WAS allows the most: every opposing forward scores 50 against it
    rows += [_row(900 + n, f"T{n % 28 + 1:02d}", "WAS", "F", 50, pid=300 + n) for n in range(10)]
    rows += [_row(900 + n, "WAS", f"T{n % 28 + 1:02d}", "G", 5, pid=400) for n in range(10)]
    ranks = m.build_ranks(rows)
    assert ranks["WAS"].vs_pos[("F", PRA)] == 30
    assert all(r.vs_pos[("F", PRA)] < 30 for t, r in ranks.items() if t != "WAS" and r.vs_pos)


def test_drtg_rank_1_is_fewest_points_per_possession():
    rows = []
    gid = 0
    for i, t in enumerate(TEAMS[:29]):  # 29 + ZZ = 30 teams
        for n in range(5):
            gid += 1
            # team i scores i+60 pts; opponent "OPP" rows are the other side. t's defense allows OPP's score.
            rows.append(_row(gid, t, "ZZ", "F", 100, pid=1000 + i, fga=80, fta=0, tov=0, oreb=0))
            rows.append(_row(gid, "ZZ", t, "F", 60 + i, pid=2000, fga=80, fta=0, tov=0, oreb=0))
    ranks = m.build_ranks(rows)
    assert ranks["T00"].drtg_rank == 1      # allowed the fewest points on identical possessions
    assert ranks["T28"].drtg_rank == 29 and ranks["ZZ"].drtg_rank == 30  # ZZ allows 100 per game, the most


def test_pace_rank_1_is_most_possessions():
    rows, gid = [], 0
    for i, t in enumerate(TEAMS[:29]):  # 29 + ZZ = 30 teams
        for n in range(5):
            gid += 1
            rows.append(_row(gid, t, "ZZ", "F", 20, pid=1000 + i, fga=70 + i, fta=0, tov=0, oreb=0))
            rows.append(_row(gid, "ZZ", t, "F", 20, pid=2000, fga=70, fta=0, tov=0, oreb=0))
    ranks = m.build_ranks(rows)
    assert ranks["T28"].pace_rank == 1 and ranks["T00"].pace_rank == 29


def test_null_position_is_excluded_from_vs_position_but_counts_in_team_totals():
    rows = _league()
    rows += [_row(990, "T05", "WAS", None, 99, pid=777, fga=50)]
    totals = m.team_totals(rows)
    assert totals[(990, "T05")]["pts"] == 99
    ranks = m.build_ranks(rows)  # does not crash and gives no None-position keys
    assert all(pos in ("F", "G") for r in ranks.values() for pos, _ in r.vs_pos)


def test_team_with_fewer_than_five_games_is_unranked_and_not_favorable():
    rows = _league() + [_row(950, "NEW", "T01", "F", 30, pid=5), _row(950, "T01", "NEW", "F", 20, pid=6)]
    ranks = m.build_ranks(rows)
    assert ranks["NEW"] == m.TeamRanks("NEW", {}, 0, 0)
    assert not m.panel(OVER, "F", ranks["NEW"], USAGE_OK).favorable


# usage: team 88 FGA, 22 FTA, 13 TOV, 240 min; player 16 FGA, 6 FTA, 3 TOV, 35 min
USAGE_GAME = dict(bdl_game_id=1, team="AAA", fga=16, fta=6, tov=3, minutes=35, game_date=date(2026, 11, 1))
USAGE_TOTALS = {(1, "AAA"): dict(pts=110, fga=88, fta=22, oreb=9, tov=13, minutes=240)}


def test_usage_formula_matches_hand_calc():
    # 100 * (16 + 2.64 + 3) * 48 / (35 * (88 + 9.68 + 13)) = 26.8
    u = m.player_usage([USAGE_GAME], USAGE_TOTALS)
    assert round(u.usage_l10, 1) == 26.8 and round(u.usage_season, 1) == 26.8
    assert u.games == 1 and u.minutes_l5 == 35


def test_usage_windows_and_dnp_games_are_ignored():
    old = [{**USAGE_GAME, "bdl_game_id": 2, "game_date": date(2026, 11, 1) - timedelta(days=k), "fga": 4}
           for k in range(1, 12)]
    totals = {**USAGE_TOTALS, **{(2, "AAA"): USAGE_TOTALS[(1, "AAA")]}}
    dnp = {**USAGE_GAME, "game_date": date(2026, 11, 20), "minutes": 0}
    u = m.player_usage([USAGE_GAME] + old + [dnp], totals)
    assert u.games == 12
    assert u.usage_l10 > u.usage_season          # recent big game outweighs old small ones in the l10 mean


OVER = {"side": "over", "market": PRA, "line": 30.5}
UNDER = {**OVER, "side": "under"}
USAGE_OK = m.PlayerUsage(usage_l10=25, usage_season=25, minutes_l5=34, minutes_season=34, games=40)



def ranks(vs, drtg, market=PRA, pos="F"):
    return m.TeamRanks("X", {(pos, market): vs}, drtg, 15)


def usage(min_l5=34, min_season=34, u10=25, useason=25):
    return m.PlayerUsage(u10, useason, min_l5, min_season, 40)


def test_over_favorable_needs_bottom_third_and_not_top10_defense():
    assert m.panel(OVER, "F", ranks(27, 22), USAGE_OK).favorable
    assert not m.panel(OVER, "F", ranks(27, 5), USAGE_OK).favorable
    assert not m.panel(OVER, "F", ranks(15, 22), USAGE_OK).favorable
    assert m.panel(OVER, "F", ranks(21, 11), USAGE_OK).favorable
    assert not m.panel(OVER, "F", ranks(20, 22), USAGE_OK).favorable
    assert not m.panel(OVER, "F", ranks(27, 10), USAGE_OK).favorable


def test_minutes_drop_kills_favorable():
    assert not m.panel(OVER, "F", ranks(27, 22), usage(min_l5=28, min_season=35)).favorable
    assert m.panel(OVER, "F", ranks(27, 22), usage(min_l5=29.75, min_season=35)).favorable  # exactly 85%


def test_usage_drop_kills_favorable():
    assert not m.panel(OVER, "F", ranks(27, 22), usage(u10=21.9, useason=25)).favorable
    assert m.panel(OVER, "F", ranks(27, 22), usage(u10=22, useason=25)).favorable


def test_under_is_mirrored():
    assert m.panel(UNDER, "F", ranks(4, 12), USAGE_OK).favorable
    assert m.panel(UNDER, "F", ranks(10, 20), USAGE_OK).favorable
    assert not m.panel(UNDER, "F", ranks(11, 12), USAGE_OK).favorable
    assert not m.panel(UNDER, "F", ranks(4, 21), USAGE_OK).favorable
    assert not m.panel(UNDER, "F", ranks(27, 22), USAGE_OK).favorable


def test_panel_fields_and_missing_position_market_is_not_favorable():
    p = m.panel(OVER, "F", ranks(27, 22), usage(u10=27, useason=25))
    assert (p.opp_vs_pos_rank, p.drtg_rank, p.pace_rank, p.usage_change, p.minutes_l5) == (27, 22, 15, 2, 34)
    assert not m.panel(OVER, "C", ranks(27, 22), USAGE_OK).favorable


def test_fewer_than_30_teams_never_favorable():
    ranks25 = m.build_ranks(_league(n=25))
    assert len(ranks25) == 25
    for t, r in ranks25.items():
        assert r.drtg_rank == 0 and r.pace_rank == 0 and r.vs_pos == {}
        assert not m.panel(OVER, "F", r, USAGE_OK).favorable
        assert not m.panel(UNDER, "F", r, USAGE_OK).favorable
    # even at extreme hand-fed ranks, an unknown drtg (0) is never favorable
    assert not m.panel(UNDER, "F", m.TeamRanks("X", {("F", PRA): 1}, 0, 0), USAGE_OK).favorable
    assert not m.panel(OVER, "F", m.TeamRanks("X", {("F", PRA): 30}, 0, 0), USAGE_OK).favorable


def test_key_missing_for_a_defender_drops_key():
    rows = _league()
    # nobody in the league ever scored against T07 from the C spot except on 29 other teams: C key covers 29 teams
    rows += [_row(800 + i, "T01", TEAMS[i], "C", 10, pid=900 + i) for i in range(29) if TEAMS[i] != "T01"]
    ranks = m.build_ranks(rows)
    assert all(("C", PRA) not in r.vs_pos for r in ranks.values())
    assert all(("F", PRA) in r.vs_pos for r in ranks.values())


def test_player_with_few_games_not_favorable():
    assert not m.panel(OVER, "F", ranks(27, 22), m.PlayerUsage(25, 25, 34, 34, 4)).favorable
    assert m.panel(OVER, "F", ranks(27, 22), m.PlayerUsage(25, 25, 34, 34, 5)).favorable
    assert not m.panel(OVER, "F", ranks(27, 22), m.PlayerUsage(0, 0, 0, 0, 0)).favorable
