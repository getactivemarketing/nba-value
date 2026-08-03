"""Rolling-origin validation and the metrics that gate promotion.

The previous trainer split randomly (`train_test_split(test_size=0.2)`) on
time-series data. Validation games were drawn from the same seasons — often the
same weeks — as training, so `resid_std` was measured on partly-interleaved
data and came out 1.6% (MOV) to 4.3% (totals) too small. `resid_std` IS the
probability scale here, the NFL analogue of MLB's RUN_DIFF_LOGISTIC_K, so every
probability inherited that overconfidence.

Rolling-origin removes it: each season is predicted by a model that saw only
EARLIER seasons. Residuals and calibration are then estimated the way the model
will actually be used — forward, on unseen time.

Pure functions; no I/O, no model dependencies.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

# Clip probabilities before taking logs. Without it a single confident miss
# returns inf and destroys the whole log-loss comparison.
_EPS = 1e-6


def expanding_folds(
    seasons: Sequence[int],
    first_test: int,
    min_train_seasons: int = 5,
) -> list[tuple[list[int], int]]:
    """(train_seasons, test_season) pairs with a growing training window.

    Every fold trains strictly on earlier seasons. Any season not passed in —
    notably the final holdout — cannot appear as train or test, which is what
    keeps it genuinely untouched rather than merely unreported.
    """
    ordered = sorted(seasons)
    folds: list[tuple[list[int], int]] = []
    for season in ordered:
        if season < first_test:
            continue
        train = [s for s in ordered if s < season]
        if len(train) < min_train_seasons:
            raise ValueError(
                f"season {season} has only {len(train)} prior seasons, "
                f"need {min_train_seasons}"
            )
        folds.append((train, season))
    if not folds:
        raise ValueError("no folds produced; check first_test against seasons")
    return folds


def cover_probability(pred: float | np.ndarray, line: float | np.ndarray,
                      resid_std: float) -> float | np.ndarray:
    """P(actual > line) for a Normal(pred, resid_std) forecast.

    An understated resid_std pushes this away from 0.5 — the precise mechanism
    by which the random-split residual made every probability overconfident.
    """
    if resid_std <= 0:
        raise ValueError("resid_std must be positive")
    z = (np.asarray(pred, dtype=float) - np.asarray(line, dtype=float)) / resid_std
    # Normal CDF via erf, avoiding a scipy dependency in the hot path.
    prob = 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))
    return float(prob) if np.isscalar(pred) and np.isscalar(line) else prob


def brier_score(prob: np.ndarray, outcome: np.ndarray) -> float:
    return float(np.mean((np.asarray(prob, float) - np.asarray(outcome, float)) ** 2))


def log_loss(prob: np.ndarray, outcome: np.ndarray) -> float:
    p = np.clip(np.asarray(prob, float), _EPS, 1 - _EPS)
    y = np.asarray(outcome, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def side_accuracy(pred: np.ndarray, actual: np.ndarray, line: np.ndarray) -> float:
    """Fraction of non-push games where the model picked the correct side.

    Pushes are dropped rather than scored as half — a push is not a forecast
    that was half-right, it is a bet that never resolved.
    """
    pred, actual, line = (np.asarray(a, float) for a in (pred, actual, line))
    live = actual != line
    if not live.any():
        return float("nan")
    return float(np.mean((pred[live] > line[live]) == (actual[live] > line[live])))


def calibration_error(prob: np.ndarray, outcome: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error: mean |predicted - realised| over probability bins.

    Weighted by bin population, so a well-populated miscalibrated bin counts
    for more than a nearly-empty one.
    """
    p = np.asarray(prob, float)
    y = np.asarray(outcome, float)
    if len(p) == 0:
        return float("nan")

    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
        if not mask.any():
            continue
        total += mask.sum() * abs(p[mask].mean() - y[mask].mean())
    return float(total / len(p))


def summarize_predictions(
    pred: np.ndarray,
    actual: np.ndarray,
    line: np.ndarray,
    resid_std: float,
    bins: int = 10,
) -> dict:
    """Every promotion metric for one set of predictions.

    Point accuracy (MAE/RMSE) and probabilistic quality (Brier, log loss,
    calibration) are reported together on purpose: a model can improve MAE
    while getting worse at the thing bets are actually priced on.
    """
    pred, actual, line = (np.asarray(a, float) for a in (pred, actual, line))
    prob = cover_probability(pred, line, resid_std)
    outcome = (actual > line).astype(float)

    return {
        "n": int(len(pred)),
        "mae": float(np.mean(np.abs(pred - actual))),
        "rmse": float(np.sqrt(np.mean((pred - actual) ** 2))),
        "brier": brier_score(prob, outcome),
        "log_loss": log_loss(prob, outcome),
        "calibration_error": calibration_error(prob, outcome, bins=bins),
        "side_accuracy": side_accuracy(pred, actual, line),
    }
