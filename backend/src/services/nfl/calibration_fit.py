"""Isotonic probability calibration for the NFL models (reuses sklearn)."""
import numpy as np
from sklearn.isotonic import IsotonicRegression


def fit_isotonic(raw_probs, outcomes) -> IsotonicRegression:
    cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    cal.fit(np.asarray(raw_probs, dtype=float), np.asarray(outcomes, dtype=float))
    return cal


def apply_calibration(cal: IsotonicRegression, probs) -> np.ndarray:
    return np.clip(cal.predict(np.asarray(probs, dtype=float)), 0.0, 1.0)
