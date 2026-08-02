"""The runline cover curve is fitted, not derived by shifting the win curve.

Deriving P(cover -1.5) as "the win curve shifted by the spread" assumes margins
are logistic with a fixed scale. They are not: 27.7% of 2026 games were decided
by exactly one run, and the +/-1.5 line sits directly on that spike, so the
probability is most wrong exactly where it matters.

Fitted on 1,405 games (pooled to 2,810 observations by symmetry, since
P(margin>=2 | rd) mirrors P(margin<=-2 | -rd)):

    P(cover -1.5 | rd) = sigmoid(-0.577 + 0.245*rd)

versus the shifted win curve's implied sigmoid(-0.587 + 0.391*rd). Nearly the
same intercept, but a slope 1.6x flatter — cover probability responds far less
to predicted run differential than winning does, because the one-run spike
absorbs the movement.

HONEST CAVEAT, pinned here so it is not forgotten: a 281-game holdout could
NOT separate the two curves (log-loss 0.6364 fitted vs 0.6348 shifted — the
shifted curve was marginally better, well inside noise). The slope difference
IS significant in-sample (bootstrap 95% CI [0.168, 0.317], excluding 0.391),
and the curves diverge only where |rd| > ~1.5, which is ~17% of games. This is
a principled correction, not a demonstrated improvement. Runline stays PAUSED.
"""
import math

import pytest

from src.services.mlb.scorer import (
    MLBScorer,
    RUNLINE_COVER_INTERCEPT,
    RUNLINE_COVER_SLOPE,
    RUN_DIFF_LOGISTIC_K,
)

cover = MLBScorer._run_diff_to_cover_prob


class TestFittedConstants:
    def test_slope_is_flatter_than_the_win_curve(self):
        """The whole point: cover responds less to rd than winning does."""
        assert RUNLINE_COVER_SLOPE < RUN_DIFF_LOGISTIC_K
        assert RUN_DIFF_LOGISTIC_K / RUNLINE_COVER_SLOPE == pytest.approx(1.6, abs=0.15)

    def test_slope_sits_inside_the_bootstrap_interval(self):
        assert 0.168 <= RUNLINE_COVER_SLOPE <= 0.317

    def test_win_curve_slope_is_outside_that_interval(self):
        """0.391 was measurably the wrong slope for this question."""
        assert not (0.168 <= RUN_DIFF_LOGISTIC_K <= 0.317)

    def test_intercept_matches_the_base_rate_at_zero(self):
        """~35% of games are won by 2+ runs; at rd=0 the curve must say so."""
        assert cover(0.0, 1.5) == pytest.approx(0.35, abs=0.02)


class TestCoverProbability:
    def test_rises_with_predicted_run_differential(self):
        assert cover(-2.0, 1.5) < cover(0.0, 1.5) < cover(2.0, 1.5)

    def test_is_flatter_than_the_old_shifted_curve(self):
        """Diverges where it matters: at large |rd|, where runline bets live."""
        old = lambda rd: 1 / (1 + math.exp(-RUN_DIFF_LOGISTIC_K * (rd - 1.5)))
        assert cover(2.0, 1.5) < old(2.0)
        assert cover(-2.0, 1.5) > old(-2.0)

    def test_agrees_with_the_old_curve_near_zero(self):
        """Both are anchored to the same base rate; only the slope differs."""
        old = 1 / (1 + math.exp(-RUN_DIFF_LOGISTIC_K * (0.0 - 1.5)))
        assert cover(0.0, 1.5) == pytest.approx(old, abs=0.01)

    def test_away_side_uses_the_mirrored_run_diff(self):
        """_runline_side_values flips rd for the away team; symmetry must hold."""
        assert cover(1.3, 1.5) + cover(-1.3, 1.5) == pytest.approx(
            cover(0.0, 1.5) * 2, abs=0.02
        )

    def test_stays_within_clamped_bounds(self):
        assert 0.05 <= cover(-25.0, 1.5) <= 0.95
        assert 0.05 <= cover(25.0, 1.5) <= 0.95

    def test_non_standard_spread_falls_back_rather_than_lying(self):
        """The fit is specific to +/-1.5. Anything else must not silently reuse it."""
        got = cover(1.0, 2.5)
        shifted = 1 / (1 + math.exp(-RUN_DIFF_LOGISTIC_K * (1.0 - 2.5)))
        assert got == pytest.approx(shifted, abs=1e-9)
