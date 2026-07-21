# src/tasks/nfl_qb_backtest.py
"""P2.5a GO/NO-GO: does qb_delta push NFL spread through the real gate?

Walk-forward 2019-24. Runs the model WITH qb_delta (v2) vs a v1-equivalent
(qb_delta zeroed so the feature is inert), and reports spread ATS% / units:
overall AND on the subset of games where qb_delta != 0 (the games the feature
is meant to help). Totals are untouched (sanity: should be identical run-to-run
since qb_delta is not in TOTALS_FEATURES).

GO criterion: v2 spread clears >= 52.4% ATS with positive units over 2019-24,
AND the lift is concentrated in the non-zero-delta subset (not spread evenly /
noise). On GO, retrains the full-data MOV model with qb_delta live and saves
it as models/nfl_mov_v2.joblib (v1 is left untouched; no gate flip; no live
wiring -- this script only produces a candidate bundle for review).

Run: export DATABASE_URL=$(grep -oE "postgresql://[^\"']+" src/tasks/prediction_tracker.py | head -1)
     python3 -m src.tasks.nfl_qb_backtest
"""
import asyncio

import structlog

from src.database import async_session_maker
from src.services.nfl.training_data import (
    load_training_frames, build_feature_frame, MOV_FEATURES)
from src.services.nfl.backtest import walk_forward, _aggregate
from src.services.nfl.model_training import train_regressor, save_bundle

logger = structlog.get_logger()
ALL_SEASONS = list(range(2010, 2025))
TEST_SEASONS = [2019, 2020, 2021, 2022, 2023, 2024]
GO_ATS_THRESHOLD = 52.4
V2_MODEL_PATH = "models/nfl_mov_v2.joblib"


def _fmt(label: str, agg: dict) -> str:
    return (f"  {label:28} n={agg['n']:4d}  ATS={agg['ats_pct']:5.1f}%  "
            f"units={agg['units']:+7.2f}")


def _subset_aggregate(spread_picks: list, game_ids: set) -> dict:
    """Filter already-graded picks (from a full-data walk-forward run) down to
    a subset of games and re-aggregate. Does NOT retrain -- retraining on the
    ~15% non-zero-delta subset alone would be methodologically wrong (too
    little data, and it would no longer be the same model being evaluated)."""
    subset = [p for p in spread_picks if p is not None and p["game_id"] in game_ids]
    return _aggregate(subset)


async def main() -> None:
    async with async_session_maker() as session:
        games, ts, ctx, lines, qb_deltas = await load_training_frames(session, ALL_SEASONS)

    frame_v2 = build_feature_frame(games, ts, ctx, lines, qb_deltas)      # qb_delta live
    frame_v1 = frame_v2.copy(); frame_v1["qb_delta"] = 0.0                # feature inert

    nonzero_mask = frame_v2["qb_delta"].abs() > 1e-9
    nonzero_game_ids = set(frame_v2.loc[nonzero_mask, "game_id"])

    print(f"modelable games={len(frame_v2)}  non-zero qb_delta games={int(nonzero_mask.sum())} "
          f"({100 * nonzero_mask.mean():.1f}%)")

    res_v1 = walk_forward(frame_v1, TEST_SEASONS)
    res_v2 = walk_forward(frame_v2, TEST_SEASONS)

    v1_subset = _subset_aggregate(res_v1["spread_picks"], nonzero_game_ids)
    v2_subset = _subset_aggregate(res_v2["spread_picks"], nonzero_game_ids)

    print("\n=== SPREAD: v1 (no qb_delta) vs v2 (qb_delta) -- 2019-24 walk-forward ===")
    print(_fmt("v1 overall", res_v1["spread"]))
    print(_fmt("v2 overall", res_v2["spread"]))
    print(_fmt("v1 non-zero-delta subset", v1_subset))
    print(_fmt("v2 non-zero-delta subset", v2_subset))

    print("\n=== TOTALS (unchanged sanity -- qb_delta is not in TOTALS_FEATURES) ===")
    print(_fmt("v1 totals", res_v1["totals"]))
    print(_fmt("v2 totals", res_v2["totals"]))
    totals_identical = res_v1["totals"] == res_v2["totals"]
    print(f"  totals identical v1 vs v2: {totals_identical}")

    overall_clears_gate = (res_v2["spread"]["ats_pct"] >= GO_ATS_THRESHOLD
                            and res_v2["spread"]["units"] > 0)
    lift_from_nonzero = v2_subset["ats_pct"] > v1_subset["ats_pct"]

    go = overall_clears_gate and lift_from_nonzero

    print("\n=== DECISION ===")
    print(f"  v2 overall clears gate (>= {GO_ATS_THRESHOLD}% ATS, positive units): {overall_clears_gate}")
    print(f"  lift concentrated in non-zero-delta subset (v2 subset ATS > v1 subset ATS): {lift_from_nonzero}")
    print(f"  GO" if go else "  NO-GO")

    if go:
        final_mov, final_std = train_regressor(frame_v2, MOV_FEATURES, "margin")
        save_bundle(V2_MODEL_PATH, final_mov, MOV_FEATURES, final_std, ALL_SEASONS)
        print(f"\nGO -> saved candidate bundle to {V2_MODEL_PATH} (mov_std={final_std:.2f}). "
              f"v1 left untouched; nfl_spread_in_best_bet NOT flipped; not wired live.")
    else:
        print("\nNO-GO -> v2 bundle NOT saved. v1 remains the live model; spread stays SHADOW.")


if __name__ == "__main__":
    asyncio.run(main())
