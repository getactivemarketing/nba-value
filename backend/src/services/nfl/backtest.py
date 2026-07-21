"""Walk-forward NFL backtest: train on prior seasons, grade the test season's
spread + total picks against nflverse closing lines at flat -110 pricing."""
import pandas as pd
import structlog

from src.services.nfl.model_training import train_regressor, predict_mov
from src.services.nfl.training_data import MOV_FEATURES, TOTALS_FEATURES
from src.services.ml.probability import mov_to_spread_prob, mov_to_total_prob

logger = structlog.get_logger()

_WIN_110 = 100 * (100 / 110)  # +90.909... profit on a winning -110 bet


def grade_spread_pick(pred_mov, resid_std, spread_line, actual_margin, threshold):
    """Grade one spread pick vs the closing line at flat -110 pricing.

    nflverse spread_line = points HOME is favored by (positive => home favored).
    Home covers iff actual_margin > spread_line. Model P(home covers) is derived
    from mov_to_spread_prob's own sign convention (spread_line negative = home
    favored), so we negate spread_line on the way in -- deliberate, pinned by tests.
    """
    if pd.isna(spread_line) or pd.isna(actual_margin):
        return None
    p_home = mov_to_spread_prob(pred_mov, -spread_line, resid_std)
    edge = p_home - 0.5
    if abs(edge) < threshold:
        return None
    side = "home" if edge > 0 else "away"
    if actual_margin == spread_line:
        return None  # push
    home_covers = actual_margin > spread_line
    won = home_covers if side == "home" else not home_covers
    return {"side": side, "edge": float(edge), "won": bool(won),
            "profit": _WIN_110 if won else -100.0}


def grade_total_pick(pred_total, total_std, total_line, actual_total, threshold):
    """Grade one total (over/under) pick vs the closing total at flat -110 pricing.

    mov_to_total_prob(home_total_estimate, away_total_estimate, total_line, total_std)
    sums the two estimates internally to get the predicted total, so our single
    combined-total regressor prediction is passed as one component and 0 as the
    other -- the sum (and therefore the probability) is unchanged. This reuses
    probability.py verbatim; no new norm.cdf math.
    """
    if pd.isna(total_line) or pd.isna(actual_total):
        return None
    p_over = mov_to_total_prob(pred_total, 0.0, total_line, total_std)
    edge = p_over - 0.5
    if abs(edge) < threshold:
        return None
    side = "over" if edge > 0 else "under"
    if actual_total == total_line:
        return None  # push
    over_hit = actual_total > total_line
    won = over_hit if side == "over" else not over_hit
    return {"side": side, "edge": float(edge), "won": bool(won),
            "profit": _WIN_110 if won else -100.0}


def _aggregate(picks):
    graded = [p for p in picks if p is not None]
    wins = sum(1 for p in graded if p["won"])
    units = sum(p["profit"] for p in graded) / 100.0
    n = len(graded)
    return {"n": n, "wins": wins, "ats_pct": round(100 * wins / n, 1) if n else 0.0,
            "units": round(units, 2)}


def _reliability(picks):
    """Calibration check: bucket graded picks by |edge| and report realized win
    rate per band. A calibrated model wins MORE often in higher-edge bands."""
    graded = [p for p in picks if p is not None]
    bands = [(0.03, 0.06), (0.06, 0.10), (0.10, 0.15), (0.15, 1.01)]
    out = []
    for lo, hi in bands:
        b = [p for p in graded if lo <= abs(p["edge"]) < hi]
        n = len(b)
        wr = round(100 * sum(1 for p in b if p["won"]) / n, 1) if n else None
        out.append({"edge_band": f"{lo:.2f}-{hi:.2f}", "n": n, "win_pct": wr})
    return out


def walk_forward(frame: pd.DataFrame, test_seasons: list[int], threshold: float = 0.05) -> dict:
    """For each test season, train MOV + totals regressors on strictly earlier
    seasons (frame["season"] < s -- never the test season itself), predict and
    grade that season's picks, then aggregate across all test seasons."""
    spread_picks, total_picks, sat = [], [], 0.0
    for s in test_seasons:
        train = frame[frame["season"] < s]
        test = frame[frame["season"] == s]
        if train.empty or test.empty:
            continue
        mov_model, mov_std = train_regressor(train, MOV_FEATURES, "margin")
        tot_model, tot_std = train_regressor(train, TOTALS_FEATURES, "total")
        mov_pred = predict_mov({"model": mov_model, "feature_cols": MOV_FEATURES}, test)
        tot_pred = predict_mov({"model": tot_model, "feature_cols": TOTALS_FEATURES}, test)
        for i, (_, g) in enumerate(test.iterrows()):
            sp = grade_spread_pick(mov_pred[i], mov_std, g["spread_line"], g["margin"], threshold)
            tp = grade_total_pick(tot_pred[i], tot_std, g["total_line"], g["total"], threshold)
            spread_picks.append(sp)
            total_picks.append(tp)
            sat = max(sat, abs(mov_to_spread_prob(mov_pred[i], -g["spread_line"], mov_std) - 0.5) + 0.5)
    return {"spread": _aggregate(spread_picks), "totals": _aggregate(total_picks),
            "spread_reliability": _reliability(spread_picks),
            "totals_reliability": _reliability(total_picks),
            "saturation_max_prob": round(sat, 3)}
