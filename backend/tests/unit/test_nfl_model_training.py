import numpy as np
import pandas as pd
from src.services.nfl.model_training import train_regressor, save_bundle, load_bundle, predict_mov
from src.services.ml.probability import mov_to_spread_prob, mov_to_moneyline_prob


def _synth(n=400, seed=0):
    rng = np.random.default_rng(seed)
    power_diff = rng.normal(0, 0.15, n)
    off = rng.normal(0, 0.1, n)
    noise = rng.normal(0, 10, n)
    # margin driven by power_diff (~50 pts per unit) + noise
    margin = 50 * power_diff + noise
    return pd.DataFrame({
        "off_epa_diff": off, "def_epa_diff": rng.normal(0, 0.1, n),
        "pass_epa_diff": rng.normal(0, 0.1, n), "rush_epa_diff": rng.normal(0, 0.1, n),
        "success_rate_diff": rng.normal(0, 0.03, n), "pace_diff": rng.normal(0, 3, n),
        "power_diff": power_diff, "rest_diff": rng.integers(-3, 4, n),
        "is_divisional": rng.integers(0, 2, n), "is_primetime": rng.integers(0, 2, n),
        "margin": margin,
    })


def test_train_returns_positive_resid_std_and_signal(tmp_path):
    from src.services.nfl.training_data import MOV_FEATURES
    frame = _synth()
    model, resid_std = train_regressor(frame, MOV_FEATURES, "margin")
    assert resid_std > 0
    # NFL MOV residual std is realistically ~9-15; synthetic noise=10 -> in range
    assert 5 < resid_std < 20
    # Higher power_diff -> higher predicted margin (learned signal)
    hi = frame.assign(power_diff=0.4).iloc[[0]]
    lo = frame.assign(power_diff=-0.4).iloc[[0]]
    bundle = {"model": model, "feature_cols": MOV_FEATURES, "resid_std": resid_std}
    assert predict_mov(bundle, hi)[0] > predict_mov(bundle, lo)[0]


def test_bundle_roundtrip_and_prob_monotonic(tmp_path):
    from src.services.nfl.training_data import MOV_FEATURES
    frame = _synth()
    model, resid_std = train_regressor(frame, MOV_FEATURES, "margin")
    p = tmp_path / "nfl_mov_test.joblib"
    save_bundle(str(p), model, MOV_FEATURES, resid_std, [2020, 2021])
    b = load_bundle(str(p))
    assert b["feature_cols"] == MOV_FEATURES
    assert b["trained_seasons"] == [2020, 2021]
    # A team predicted to win by 7, favored by 3 -> covers with prob > 0.5
    assert mov_to_spread_prob(7.0, -3.0, b["resid_std"]) > 0.5
    # Predicted to win by 7 -> ML win prob > 0.5
    assert mov_to_moneyline_prob(7.0, b["resid_std"]) > 0.5
