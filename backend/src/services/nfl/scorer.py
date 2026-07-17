"""Score an NFL game: project margin + total, value each market, select bests."""
import pandas as pd
import structlog

from src.config import settings
from src.services.nfl.model_training import predict_mov
from src.services.nfl.value_calculator import NFLValueCalculator
from src.services.ml.probability import (
    mov_to_spread_prob, mov_to_moneyline_prob, mov_to_total_prob)

logger = structlog.get_logger()


def _enabled_markets() -> set[str]:
    e = set()
    if settings.nfl_totals_in_best_bet:
        e.add("total")
    if settings.nfl_spread_in_best_bet:
        e.add("spread")
    if settings.nfl_ml_in_best_bet:
        e.add("moneyline")
    return e


def score_game(feature_row, market_rows, mov_bundle, totals_bundle) -> dict:
    frame = pd.DataFrame([feature_row])
    pred_margin = float(predict_mov(mov_bundle, frame)[0])
    pred_total = float(predict_mov(totals_bundle, frame)[0])
    mstd, tstd = mov_bundle["resid_std"], totals_bundle["resid_std"]
    totals_cal = totals_bundle.get("calibrator")
    calc = NFLValueCalculator
    results = []

    for m in market_rows:
        mt = m["market_type"]
        if mt == "spread" and m.get("home_odds") and m.get("away_odds"):
            p_home = mov_to_spread_prob(pred_margin, -float(m["line"]), mstd)
            mh, ma = calc.devig_two_way(m["home_odds"], m["away_odds"])
            results.append(calc.calculate_value("spread", "home_spread", p_home, mh,
                           m["home_odds"], team="home", line=m["line"], model_confidence=0.6))
            results.append(calc.calculate_value("spread", "away_spread", 1 - p_home, ma,
                           m["away_odds"], team="away", line=-float(m["line"]), model_confidence=0.6))
        elif mt == "moneyline" and m.get("home_odds") and m.get("away_odds"):
            p_home = mov_to_moneyline_prob(pred_margin, mstd)
            mh, ma = calc.devig_two_way(m["home_odds"], m["away_odds"])
            results.append(calc.calculate_value("moneyline", "home_ml", p_home, mh,
                           m["home_odds"], team="home", model_confidence=0.6))
            results.append(calc.calculate_value("moneyline", "away_ml", 1 - p_home, ma,
                           m["away_odds"], team="away", model_confidence=0.6))
        elif mt == "total" and m.get("over_odds") and m.get("under_odds"):
            p_over = mov_to_total_prob(pred_total, 0.0, float(m["line"]), tstd)
            if totals_cal is not None:
                from src.services.nfl.calibration_fit import apply_calibration
                p_over = float(apply_calibration(totals_cal, [p_over])[0])
            mo, mu = calc.devig_two_way(m["over_odds"], m["under_odds"])
            results.append(calc.calculate_value("total", "over", p_over, mo,
                           m["over_odds"], line=m["line"], model_confidence=0.6))
            results.append(calc.calculate_value("total", "under", 1 - p_over, mu,
                           m["under_odds"], line=m["line"], model_confidence=0.6))

    def best_of(mtype):
        return calc.find_best_value([r for r in results if r.market_type == mtype])

    return {
        "predicted_margin": pred_margin,
        "predicted_total": pred_total,
        "best_spread": best_of("spread"),
        "best_ml": best_of("moneyline"),
        "best_total": best_of("total"),
        "best_bet": calc.find_best_bet(results, _enabled_markets()),
    }
