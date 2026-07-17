"""NFL Phase 3 gate-tuning backtest (GO/NO-GO for shipping totals live).

Walk-forward 2019-2024. Per test season: train the totals model on prior
seasons, compute raw P(over) for the test games, apply an isotonic calibration
fit on ACCUMULATED prior-season out-of-sample (raw_p_over, outcome) pairs, devig
the -110 total market (P=0.5), and grade the totals that qualify THROUGH the
real NFLValueCalculator gate. Sweep (min_edge, max_edge) bands so the profitable
band is chosen from data, not assumed. Spread/ML are reported as shadow only.

Run:  export DATABASE_URL=$(grep -oE "postgresql://[^\"']+" src/tasks/prediction_tracker.py | head -1)
      python3 -m src.tasks.nfl_score_backtest
"""
import asyncio

from src.config import settings
from src.database import async_session_maker
from src.services.nfl.training_data import (
    load_training_frames, build_feature_frame, TOTALS_FEATURES)
from src.services.nfl.model_training import train_regressor, predict_mov
from src.services.nfl.calibration_fit import fit_isotonic, apply_calibration
from src.services.nfl.value_calculator import NFLValueCalculator
from src.services.ml.probability import mov_to_total_prob

TEST_SEASONS = list(range(2019, 2025))
BANDS = [(0.02, 0.08), (0.03, 0.10), (0.03, 0.12), (0.05, 0.10), (0.05, 0.99), (0.02, 0.99)]
_WIN_110 = 100 * (100 / 110)


def _grade_total_through_gate(p_over, total_line, actual_total, min_edge, max_edge):
    """Run one game's total through the real gate at a given band; grade if it qualifies."""
    settings.nfl_min_edge = min_edge
    settings.nfl_max_edge = max_edge
    over = NFLValueCalculator.calculate_value("total", "over", p_over, 0.5, 1.909,
                                              line=total_line, model_confidence=0.6)
    under = NFLValueCalculator.calculate_value("total", "under", 1 - p_over, 0.5, 1.909,
                                               line=total_line, model_confidence=0.6)
    pick = NFLValueCalculator.find_best_value([over, under])
    if pick is None:
        return None
    if actual_total == total_line:
        return {"won": None, "profit": 0.0, "edge": pick.raw_edge, "side": pick.bet_type}
    over_hit = actual_total > total_line
    won = over_hit if pick.bet_type == "over" else not over_hit
    return {"won": won, "profit": _WIN_110 if won else -100.0,
            "edge": pick.raw_edge, "side": pick.bet_type}


def _summarize(picks):
    graded = [p for p in picks if p and p["won"] is not None]
    n = len(graded)
    wins = sum(1 for p in graded if p["won"])
    units = sum(p["profit"] for p in graded) / 100.0
    wr = round(100 * wins / n, 1) if n else 0.0
    return n, wins, wr, round(units, 2)


async def main():
    async with async_session_maker() as s:
        frames = await load_training_frames(s, list(range(2010, 2025)))
    frame = build_feature_frame(*frames)
    print(f"modelable games: {len(frame)}")

    # Precompute per-test-season calibrated P(over) using accumulated OOS calibration.
    cal_raw, cal_out = [], []          # accumulated prior-season OOS (raw_p_over, outcome)
    per_game = []                       # (calibrated_p_over, total_line, actual_total)
    calibrated_seasons = 0
    for season in TEST_SEASONS:
        train = frame[frame["season"] < season]
        test = frame[frame["season"] == season]
        tot, tstd = train_regressor(train, TOTALS_FEATURES, "total")
        tpred = predict_mov({"model": tot, "feature_cols": TOTALS_FEATURES}, test)
        raw = [mov_to_total_prob(float(tp), 0.0, float(tl), tstd)
               for tp, tl in zip(tpred, test["total_line"])]
        if len(cal_raw) >= 200:
            cal = fit_isotonic(cal_raw, cal_out)
            p_over = list(apply_calibration(cal, raw))
            calibrated_seasons += 1
        else:
            p_over = raw
        for i, (_, g) in enumerate(test.iterrows()):
            outcome = 1 if g["total"] > g["total_line"] else 0
            cal_raw.append(raw[i]); cal_out.append(outcome)
            per_game.append((p_over[i], float(g["total_line"]), int(g["total"])))

    print(f"calibrated seasons: {calibrated_seasons}/{len(TEST_SEASONS)} "
          f"(first seasons run raw until >=200 OOS pairs accumulate)\n")

    print("=== TOTALS through the real gate, by (min_edge, max_edge) band (flat -110) ===")
    print(f"{'band':>14}  {'n':>4}  {'W':>4}  {'win%':>6}  {'units':>8}")
    for lo, hi in BANDS:
        picks = [_grade_total_through_gate(p, tl, at, lo, hi) for p, tl, at in per_game]
        n, wins, wr, units = _summarize(picks)
        flag = "  <- profitable" if units > 0 and n >= 100 else ""
        print(f"  ({lo:.2f},{hi:.2f})  {n:>4}  {wins:>4}  {wr:>6}  {units:>+8.2f}{flag}")

    # restore config defaults
    settings.nfl_min_edge, settings.nfl_max_edge = 0.03, 0.12
    print("\nNOTE: pick the band with positive units AND a meaningful sample (n>=~150).")
    print("If no band clears break-even, do NOT ship totals live (best_bet stays empty).")


if __name__ == "__main__":
    asyncio.run(main())
