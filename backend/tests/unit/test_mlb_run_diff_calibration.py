"""The run-diff -> probability logistic slope is fitted, not hand-picked.

The old k=0.5 was a guess ("k = 0.5 gives reasonable spread"). Measured over
1,463 completed 2026 games the actual margin SD is 4.64 runs, but k=0.5 implies
3.63 — i.e. the curve was ~28% too steep, making every probability too far from
0.5. That overconfidence showed up directly in the picks: the model overstated
its edge by +9pts on underdogs and +23pts on favorites.

Backtest through the real gate on 351 graded ML picks: +38.90u at k=0.391 vs
+32.87u at k=0.50 (per-pick +0.149 vs +0.130).
"""
import math

import pytest

from src.services.mlb.scorer import MLBScorer, RUN_DIFF_LOGISTIC_K

OBSERVED_MARGIN_SD = 4.64  # runs, 1463 completed games


def test_k_is_derived_from_the_observed_margin_sd():
    """A logistic with scale s=1/k has SD = s*pi/sqrt(3)."""
    expected = (math.pi / math.sqrt(3)) / OBSERVED_MARGIN_SD
    assert abs(RUN_DIFF_LOGISTIC_K - expected) < 0.005


def test_implied_spread_matches_reality_not_the_old_guess():
    implied_sd = (1 / RUN_DIFF_LOGISTIC_K) * math.pi / math.sqrt(3)
    assert abs(implied_sd - OBSERVED_MARGIN_SD) < 0.10
    # the old k=0.5 implied a far too narrow distribution
    old_sd = (1 / 0.5) * math.pi / math.sqrt(3)
    assert old_sd < OBSERVED_MARGIN_SD - 0.75


def test_win_prob_is_less_overconfident_than_the_old_curve():
    scorer = object.__new__(MLBScorer)
    for rd in (1.0, 2.0, 3.0):
        new = scorer._run_diff_to_win_prob(rd)
        old = 1 / (1 + math.exp(-0.5 * rd))
        assert new < old, f"rd={rd}: {new} should be less extreme than {old}"
        assert new > 0.5  # still favours the better team, just less strongly
    # symmetric
    assert abs(scorer._run_diff_to_win_prob(0.0) - 0.5) < 1e-9


def test_cover_prob_no_longer_shifts_the_win_curve():
    """SUPERSEDED 2026-08-02 — this once asserted the opposite.

    The cover curve was originally the win curve shifted by the spread, and
    this test pinned that. Fitting cover outcomes directly showed the
    assumption was wrong: margins are not logistic with a fixed scale, because
    27.7% of games end on a one-run margin and the +/-1.5 line sits on that
    spike. The fitted slope is 1.6x flatter (0.245 vs 0.391), with the win
    curve's slope falling outside the fitted bootstrap CI [0.168, 0.317].

    Kept as a regression guard so the shifted derivation cannot creep back.
    See test_mlb_runline_cover_curve.py for the fitted curve's own tests.
    """
    scorer = object.__new__(MLBScorer)
    shifted = scorer._run_diff_to_win_prob(2.0 - 1.5)
    assert MLBScorer._run_diff_to_cover_prob(2.0, 1.5) != pytest.approx(shifted, abs=1e-6)

    # They still agree near rd=0, where both are anchored to the same base rate.
    assert MLBScorer._run_diff_to_cover_prob(0.0, 1.5) == pytest.approx(
        scorer._run_diff_to_win_prob(-1.5), abs=0.01
    )
