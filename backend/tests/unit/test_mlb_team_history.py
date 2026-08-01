"""Reconstructing point-in-time TEAM rate stats from game logs.

Same problem the pitcher backfill solved, one level up: mlb_team_stats carried
OPS/AVG/OBP/SLG/ERA/WHIP only from the day they started being populated
(2026-07-31), so all 111 days of history sat at 0% coverage and the 2026
training set could not use them.

Rates must be recomputed from cumulative COMPONENTS, never averaged from the
per-game rates — a 1-for-4 game and a 3-for-3 game do not average to .500.
"""
import pytest

from src.services.mlb.team_history import (
    TeamGameLog,
    build_team_stat_history,
    cumulative_team_stats,
)


def hit(date, ab=36, h=10, tb=16, bb=3, hbp=1, sf=0):
    return TeamGameLog(
        date=date, at_bats=ab, hits=h, total_bases=tb,
        walks=bb, hit_by_pitch=hbp, sac_flies=sf,
    )


def pit(date, ip="9.0", er=3, h=8, bb=2):
    return TeamGameLog(
        date=date, innings_pitched=ip, earned_runs=er,
        hits_allowed=h, walks_allowed=bb,
    )


class TestCumulativeTeamStats:
    def test_batting_average_from_components(self):
        got = cumulative_team_stats([hit("2026-04-01", ab=40, h=10)], [], "2026-05-01")
        assert got["batting_avg"] == pytest.approx(0.250)

    def test_rates_are_recomputed_not_averaged(self):
        """1-for-4 then 3-for-3 is 4-for-7 (.571), not (.250+1.000)/2 = .625."""
        logs = [hit("2026-04-01", ab=4, h=1), hit("2026-04-02", ab=3, h=3)]
        got = cumulative_team_stats(logs, [], "2026-05-01")
        assert got["batting_avg"] == pytest.approx(4 / 7, abs=1e-4)

    def test_obp_includes_walks_hbp_and_sac_flies(self):
        got = cumulative_team_stats(
            [hit("2026-04-01", ab=30, h=9, bb=5, hbp=1, sf=2)], [], "2026-05-01"
        )
        # (9+5+1) / (30+5+1+2)
        assert got["obp"] == pytest.approx(15 / 38, abs=1e-4)

    def test_slg_uses_total_bases(self):
        got = cumulative_team_stats([hit("2026-04-01", ab=40, tb=18)], [], "2026-05-01")
        assert got["slg"] == pytest.approx(0.450)

    def test_ops_is_obp_plus_slg(self):
        got = cumulative_team_stats([hit("2026-04-01")], [], "2026-05-01")
        assert got["ops"] == pytest.approx(got["obp"] + got["slg"], abs=1e-6)

    def test_era_and_whip_from_pitching_logs(self):
        got = cumulative_team_stats([], [pit("2026-04-01", ip="9.0", er=3, h=8, bb=2)], "2026-05-01")
        assert got["era"] == pytest.approx(3.0)
        # Stored as Numeric(5,3), so compare at that precision.
        assert got["team_whip"] == pytest.approx(10 / 9, abs=1e-3)

    def test_innings_use_baseball_notation(self):
        """'8.2' is eight and two THIRDS innings."""
        got = cumulative_team_stats([], [pit("2026-04-01", ip="8.2", er=3)], "2026-05-01")
        assert got["era"] == pytest.approx(9 * 3 / (8 + 2 / 3), abs=1e-3)

    def test_only_games_strictly_before_the_date_count(self):
        logs = [hit("2026-04-01", ab=4, h=4), hit("2026-04-10", ab=4, h=0)]
        got = cumulative_team_stats(logs, [], "2026-04-10")
        assert got["batting_avg"] == pytest.approx(1.0)

    def test_no_prior_games_returns_none(self):
        assert cumulative_team_stats([hit("2026-04-10")], [], "2026-04-01") is None

    def test_zero_at_bats_does_not_divide_by_zero(self):
        got = cumulative_team_stats([hit("2026-04-01", ab=0, h=0, tb=0)], [], "2026-05-01")
        assert got is None or got["batting_avg"] is None

    def test_hitting_only_still_yields_hitting_rates(self):
        got = cumulative_team_stats([hit("2026-04-01")], [], "2026-05-01")
        assert got["ops"] is not None
        assert got["era"] is None


class TestBuildTeamStatHistory:
    def test_one_row_per_game_day_dated_the_day_after(self):
        hitting = [hit("2026-04-01"), hit("2026-04-02")]
        pitching = [pit("2026-04-01"), pit("2026-04-02")]
        hist = build_team_stat_history(hitting, pitching)
        assert [h["stat_date"] for h in hist] == ["2026-04-02", "2026-04-03"]

    def test_rows_accumulate(self):
        hitting = [hit("2026-04-01", ab=4, h=1), hit("2026-04-02", ab=4, h=3)]
        hist = build_team_stat_history(hitting, [])
        assert hist[0]["batting_avg"] == pytest.approx(0.250)
        assert hist[1]["batting_avg"] == pytest.approx(0.500)

    def test_merges_hitting_and_pitching_dates(self):
        hist = build_team_stat_history([hit("2026-04-01")], [pit("2026-04-01")])
        assert len(hist) == 1
        assert hist[0]["ops"] is not None and hist[0]["era"] is not None

    def test_empty_input_yields_nothing(self):
        assert build_team_stat_history([], []) == []
