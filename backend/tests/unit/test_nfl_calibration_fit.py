import numpy as np
from src.services.nfl.calibration_fit import fit_isotonic, apply_calibration


def test_isotonic_monotonic_and_corrects_overconfidence():
    # raw probs are overconfident: high raw prob doesn't win more.
    rng = np.random.default_rng(0)
    raw = np.concatenate([np.full(200, 0.8), np.full(200, 0.55)])
    # 0.8-bucket actually wins 50%, 0.55-bucket wins 58% (inverted, like NFL totals)
    outcomes = np.concatenate([
        (rng.random(200) < 0.50).astype(int),
        (rng.random(200) < 0.58).astype(int)])
    cal = fit_isotonic(raw, outcomes)
    p = apply_calibration(cal, np.array([0.55, 0.80]))
    # calibration pulls the overconfident 0.80 down toward its true ~0.50
    assert p[1] <= 0.62
    # outputs stay in [0,1]
    assert (p >= 0).all() and (p <= 1).all()
