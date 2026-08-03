"""Rolling-origin validation and the metrics that decide promotion.

The existing trainer used `train_test_split(test_size=0.2)` — a RANDOM split on
time-series data. Validation games came from the same seasons and weeks as
training, so `resid_std` (which IS the probability scale) was measured on
partly-interleaved data and came out ~2-4% too small. Every probability was
correspondingly overconfident.

Rolling-origin fixes it: each season is predicted by a model that saw only
EARLIER seasons, so residuals and calibration are estimated the way the model
will actually be used. 2025 is never touched by any fold.
"""
import numpy as np
import pytest

from src.services.nfl.rolling_eval import (
    calibration_error,
    cover_probability,
    expanding_folds,
    brier_score,
    log_loss,
    side_accuracy,
    summarize_predictions,
)


class TestExpandingFolds:
    def test_each_fold_trains_only_on_earlier_seasons(self):
        folds = expanding_folds(list(range(2010, 2025)), first_test=2020)
        for train, test in folds:
            assert max(train) < test

    def test_training_window_grows(self):
        folds = expanding_folds(list(range(2010, 2025)), first_test=2020)
        assert [len(t) for t, _ in folds] == sorted(len(t) for t, _ in folds)
        assert len(folds[0][0]) < len(folds[-1][0])

    def test_covers_every_season_from_first_test_onward(self):
        folds = expanding_folds(list(range(2010, 2025)), first_test=2020)
        assert [test for _, test in folds] == [2020, 2021, 2022, 2023, 2024]

    def test_holdout_season_is_never_a_fold(self):
        """2025 must not appear in any fold, as train or as test."""
        folds = expanding_folds(list(range(2010, 2025)), first_test=2020)
        seasons = {s for tr, te in folds for s in list(tr) + [te]}
        assert 2025 not in seasons

    def test_requires_a_minimum_training_history(self):
        with pytest.raises(ValueError):
            expanding_folds([2010, 2011], first_test=2010)


class TestCoverProbability:
    def test_even_line_is_a_coin_flip(self):
        assert cover_probability(pred=3.0, line=3.0, resid_std=13.0) == pytest.approx(0.5)

    def test_model_above_the_line_favours_the_over_side(self):
        assert cover_probability(pred=7.0, line=3.0, resid_std=13.0) > 0.5

    def test_wider_residuals_pull_toward_a_coin_flip(self):
        tight = cover_probability(pred=10.0, line=3.0, resid_std=8.0)
        loose = cover_probability(pred=10.0, line=3.0, resid_std=18.0)
        assert 0.5 < loose < tight

    def test_understated_resid_std_produces_overconfidence(self):
        """The exact defect the random split introduced."""
        honest = cover_probability(pred=10.0, line=3.0, resid_std=13.36)
        optimistic = cover_probability(pred=10.0, line=3.0, resid_std=12.98)
        assert optimistic > honest


class TestMetrics:
    def test_brier_rewards_confident_correctness(self):
        y = np.array([1, 1, 0, 0])
        assert brier_score(np.array([0.9, 0.9, 0.1, 0.1]), y) < \
               brier_score(np.array([0.6, 0.6, 0.4, 0.4]), y)

    def test_log_loss_punishes_confident_errors_hardest(self):
        y = np.array([1])
        assert log_loss(np.array([0.01]), y) > log_loss(np.array([0.4]), y)

    def test_log_loss_is_finite_at_the_extremes(self):
        assert np.isfinite(log_loss(np.array([0.0, 1.0]), np.array([1, 0])))

    def test_side_accuracy_ignores_pushes(self):
        pred = np.array([5.0, 5.0, 5.0])
        actual = np.array([7.0, 1.0, 3.0])
        line = np.array([3.0, 3.0, 3.0])
        # game 1: model over, actual over   -> correct
        # game 2: model over, actual under  -> wrong
        # game 3: actual lands ON the line  -> push, excluded entirely
        assert side_accuracy(pred, actual, line) == pytest.approx(0.5)

    def test_calibration_error_is_zero_for_a_perfect_forecaster(self):
        p = np.repeat([0.2, 0.5, 0.8], 200)
        rng = np.random.default_rng(0)
        y = (rng.random(len(p)) < p).astype(int)
        assert calibration_error(p, y, bins=3) < 0.05

    def test_calibration_error_catches_systematic_overconfidence(self):
        p = np.repeat(0.9, 400)
        y = (np.arange(400) % 2).astype(int)   # actually 50/50
        assert calibration_error(p, y, bins=3) > 0.3


class TestSummarizePredictions:
    def _frame(self, n=200):
        rng = np.random.default_rng(1)
        line = rng.normal(0, 6, n)
        actual = line + rng.normal(0, 13, n)
        pred = line + rng.normal(0, 3, n)
        return pred, actual, line

    def test_reports_every_required_metric(self):
        pred, actual, line = self._frame()
        got = summarize_predictions(pred, actual, line, resid_std=13.0)
        for key in ("n", "mae", "rmse", "brier", "log_loss", "calibration_error", "side_accuracy"):
            assert key in got, f"missing {key}"

    def test_n_excludes_nothing_but_side_accuracy_excludes_pushes(self):
        pred = np.array([5.0, 5.0])
        actual = np.array([7.0, 3.0])
        line = np.array([3.0, 3.0])
        got = summarize_predictions(pred, actual, line, resid_std=13.0)
        assert got["n"] == 2
        assert got["side_accuracy"] == pytest.approx(1.0)
