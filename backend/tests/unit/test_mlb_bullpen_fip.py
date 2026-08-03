"""Bullpen workload and FIP — the two feature families the model cannot see.

Relievers throw roughly 40% of modern innings and the model has ZERO
visibility into them. Meanwhile it judges pitchers by ERA, an outcome stat
carrying large defensive and sequencing luck, when FIP uses only the three
outcomes a pitcher controls almost entirely.

Neither needs a new data source. A team's pitching game log covers all its
pitchers; subtracting that game's starter leaves the bullpen. FIP components
(HR, BB, HBP, K, IP) are already in the same per-game logs.

These are ACQUISITION, not promotion: the 2026 retrain showed 1,405 games
cannot validate more features. History accumulates now so a future retrain
can use it — the alternative is discovering in April that we have none.
"""
import pytest

from src.services.mlb.bullpen import (
    FIP_CONSTANT_DEFAULT,
    BullpenLine,
    bullpen_from_team_and_starter,
    cumulative_bullpen,
    fip,
    recent_workload,
)


class TestFip:
    def test_uses_only_the_three_true_outcomes(self):
        """13*HR + 3*(BB+HBP) - 2*K, over IP, plus the league constant."""
        got = fip(home_runs=2, walks=5, hit_by_pitch=1, strikeouts=10, innings=20.0)
        expected = (13 * 2 + 3 * 6 - 2 * 10) / 20.0 + FIP_CONSTANT_DEFAULT
        assert got == pytest.approx(expected, abs=1e-6)

    def test_strikeouts_lower_it_and_walks_raise_it(self):
        base = dict(home_runs=1, walks=3, hit_by_pitch=0, innings=20.0)
        assert fip(strikeouts=30, **base) < fip(strikeouts=10, **base)
        assert fip(home_runs=1, walks=10, hit_by_pitch=0, strikeouts=20, innings=20.0) > \
               fip(home_runs=1, walks=2, hit_by_pitch=0, strikeouts=20, innings=20.0)

    def test_zero_innings_is_none_not_infinity(self):
        assert fip(home_runs=0, walks=0, hit_by_pitch=0, strikeouts=0, innings=0.0) is None

    def test_accepts_a_league_specific_constant(self):
        a = fip(home_runs=1, walks=2, hit_by_pitch=0, strikeouts=8, innings=10.0, constant=3.0)
        b = fip(home_runs=1, walks=2, hit_by_pitch=0, strikeouts=8, innings=10.0, constant=3.2)
        assert b - a == pytest.approx(0.2, abs=1e-9)


class TestBullpenFromTeamAndStarter:
    def test_subtracts_the_starter_from_the_team_line(self):
        got = bullpen_from_team_and_starter(
            team_ip=9.0, team_er=4, team_h=8, team_bb=3,
            starter_ip=6.0, starter_er=3, starter_h=5, starter_bb=2,
        )
        assert got.innings == pytest.approx(3.0)
        assert got.earned_runs == 1 and got.hits == 3 and got.walks == 1

    def test_handles_thirds_correctly(self):
        """Team 8.2 minus starter 5.1 is 3.1 innings, i.e. 3 1/3."""
        got = bullpen_from_team_and_starter(
            team_ip=8 + 2 / 3, team_er=3, team_h=7, team_bb=2,
            starter_ip=5 + 1 / 3, starter_er=2, starter_h=5, starter_bb=1,
        )
        assert got.innings == pytest.approx(3 + 1 / 3, abs=1e-6)

    def test_complete_game_yields_an_idle_bullpen(self):
        got = bullpen_from_team_and_starter(
            team_ip=9.0, team_er=2, team_h=5, team_bb=1,
            starter_ip=9.0, starter_er=2, starter_h=5, starter_bb=1,
        )
        assert got.innings == pytest.approx(0.0)
        assert got.earned_runs == 0

    def test_missing_starter_line_returns_none(self):
        """Without the starter's line the split is unknowable — do not guess."""
        assert bullpen_from_team_and_starter(
            team_ip=9.0, team_er=4, team_h=8, team_bb=3,
            starter_ip=None, starter_er=None, starter_h=None, starter_bb=None,
        ) is None

    def test_negative_residual_is_clamped_to_zero_not_negative(self):
        """Data noise must not produce a bullpen that un-allowed runs."""
        got = bullpen_from_team_and_starter(
            team_ip=6.0, team_er=2, team_h=4, team_bb=1,
            starter_ip=6.0, starter_er=3, starter_h=5, starter_bb=2,
        )
        assert got.earned_runs == 0 and got.hits == 0 and got.walks == 0


class TestCumulativeBullpen:
    def _line(self, date, ip=3.0, er=1, h=3, bb=1):
        return (date, BullpenLine(innings=ip, earned_runs=er, hits=h, walks=bb))

    def test_rates_from_cumulative_components(self):
        lines = [self._line("2026-04-01", ip=3.0, er=1), self._line("2026-04-02", ip=3.0, er=3)]
        got = cumulative_bullpen(lines, as_of="2026-05-01")
        assert got["bullpen_era"] == pytest.approx(9 * 4 / 6.0)

    def test_only_games_strictly_before_the_date(self):
        lines = [self._line("2026-04-01"), self._line("2026-04-10")]
        assert cumulative_bullpen(lines, as_of="2026-04-10")["bullpen_ip"] == pytest.approx(3.0)

    def test_no_prior_work_is_none(self):
        assert cumulative_bullpen([self._line("2026-04-10")], as_of="2026-04-01") is None


class TestRecentWorkload:
    def _line(self, date, ip):
        return (date, BullpenLine(innings=ip, earned_runs=0, hits=0, walks=0))

    def test_sums_innings_in_the_trailing_window(self):
        lines = [self._line("2026-04-08", 4.0), self._line("2026-04-09", 3.0),
                 self._line("2026-04-10", 2.0)]
        assert recent_workload(lines, as_of="2026-04-11", days=3) == pytest.approx(9.0)

    def test_excludes_games_outside_the_window(self):
        """A heavy outing last week says nothing about tonight's freshness."""
        lines = [self._line("2026-04-01", 9.0), self._line("2026-04-10", 2.0)]
        assert recent_workload(lines, as_of="2026-04-11", days=3) == pytest.approx(2.0)

    def test_excludes_the_target_day_itself(self):
        lines = [self._line("2026-04-11", 5.0)]
        assert recent_workload(lines, as_of="2026-04-11", days=3) == pytest.approx(0.0)

    def test_idle_bullpen_is_zero_not_none(self):
        """Zero is a real, meaningful workload — a rested pen."""
        assert recent_workload([], as_of="2026-04-11", days=3) == 0.0
