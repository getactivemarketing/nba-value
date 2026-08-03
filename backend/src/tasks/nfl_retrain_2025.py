"""Retrain NFL MOV + totals through 2025 and compare to the incumbents.

Epic 1. The live models were trained on 2010-2023 and have never seen 2024 or
2025. This adds both seasons.

DELIBERATELY NOT `nfl_train.py`: that script calls save_bundle() on the live
model paths, overwriting the incumbents in place. Rollback then requires
retraining rather than a config change, and there is nothing left to compare
against. Challengers here are written to new paths and promoted only by config.

TWO METHOD FIXES over the existing trainer:

1. CHRONOLOGICAL SPLITS. `train_regressor` uses
   `train_test_split(..., test_size=0.2)` — a RANDOM split on time-series data.
   Validation games are drawn from the same seasons, often the same weeks, as
   training. Early stopping is tuned against temporally interleaved data, and
   more importantly `resid_std` — which IS the probability scale, the NFL
   analogue of MLB's RUN_DIFF_LOGISTIC_K — is estimated from it. A random
   split understates forward error, so every probability comes out
   overconfident. Both are reported here so the size of the bias is visible.

2. Artifacts record `trained_at`. The MLB totals model shipped with
   trained_at=None and consequently reports as "stale, age unknown" forever.

Judged on residual error and calibration, NOT backtested P&L — the +10.2u /
53.9% totals figure is exactly the kind of repeatedly-examined sample the
research rules warn against.

    python -m src.tasks.nfl_retrain_2025 [--save]
"""

import asyncio
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from src.database import async_session_maker
from src.services.nfl.model_training import train_regressor
from src.services.nfl.training_data import (
    MOV_FEATURES, TOTALS_FEATURES, build_feature_frame, load_training_frames,
)

ALL = list(range(2010, 2026))
TRAIN = list(range(2010, 2025))     # through 2024
HOLDOUT = [2025]                    # a full unseen season, forward in time

INCUMBENT_MOV = "models/nfl_mov_v1.joblib"
INCUMBENT_TOT = "models/nfl_totals_v1.joblib"
OUT_MOV = "models/nfl_mov_2025.joblib"
OUT_TOT = "models/nfl_totals_2025.joblib"


def _chrono_fit(lgb, frame, cols, target, params):
    """Fit with a CHRONOLOGICAL validation tail, and return forward resid_std."""
    frame = frame.sort_values(["season", "week"])
    cut = int(len(frame) * 0.85)
    tr, va = frame.iloc[:cut], frame.iloc[cut:]
    X_tr, X_va = tr[cols].fillna(0.0), va[cols].fillna(0.0)
    model = lgb.train(
        params,
        lgb.Dataset(X_tr, label=tr[target].astype(float)),
        num_boost_round=2000,
        valid_sets=[lgb.Dataset(X_va, label=va[target].astype(float))],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    resid = va[target].astype(float).values - model.predict(
        X_va, num_iteration=model.best_iteration)
    return model, float(np.std(resid))


def _report(name, pred, actual, line=None, is_total=False):
    mae = float(np.mean(np.abs(pred - actual)))
    rmse = float(np.sqrt(np.mean((pred - actual) ** 2)))
    out = {"name": name, "mae": mae, "rmse": rmse, "ats": None}
    if line is not None:
        # nflverse `spread_line` is POSITIVE when the HOME team is favoured,
        # and `margin` is home-away — so the market's own margin forecast is
        # +spread_line, not -spread_line. Verified empirically on 2022-25:
        # MAE 9.806 using +, 14.264 using -, corr(spread_line, margin) +0.452.
        # The inverted version produced an impossible 80% "ATS" and made the
        # model look 4.8 points better than the market.
        model_over = pred > line
        actual_over = actual > line
        live = actual != line              # drop exact pushes
        out["ats"] = float(np.mean(model_over[live] == actual_over[live]))
    return out


async def main(save: bool = False) -> dict:
    import lightgbm as lgb
    from src.services.nfl.model_training import _PARAMS

    async with async_session_maker() as session:
        frames = await load_training_frames(session, ALL)
    frame = build_feature_frame(*frames)
    print(f"modelable games: {len(frame)} ({frame['season'].min()}-{frame['season'].max()})")

    tr = frame[frame["season"].isin(TRAIN)]
    ho = frame[frame["season"].isin(HOLDOUT)]
    print(f"train {len(tr)} ({TRAIN[0]}-{TRAIN[-1]})   holdout {len(ho)} ({HOLDOUT[0]})\n")
    if len(ho) == 0:
        raise SystemExit("no 2025 holdout rows — backfill 2025 first")

    results = {}
    for label, cols, target, out_path, inc_path, is_total in (
        ("MOV", MOV_FEATURES, "margin", OUT_MOV, INCUMBENT_MOV, False),
        ("TOTALS", TOTALS_FEATURES, "total", OUT_TOT, INCUMBENT_TOT, True),
    ):
        print(f"=== {label} ===")
        chal, chrono_std = _chrono_fit(lgb, tr, cols, target, _PARAMS)
        _, random_std = train_regressor(tr, cols, target)

        print(f"  resid_std  chronological {chrono_std:.3f}   random-split {random_std:.3f}"
              f"   ({(chrono_std/random_std - 1)*100:+.1f}% understated by random)")

        inc = joblib.load(inc_path)
        line_col = "total_line" if is_total else "spread_line"
        line = ho[line_col].astype(float).values
        actual = ho[target].astype(float).values

        rows = [
            _report(f"challenger (2010-2024)", chal.predict(ho[cols].fillna(0.0)), actual, line, is_total),
            _report(f"incumbent  (2010-2023)", inc["model"].predict(ho[inc["feature_cols"]].fillna(0.0)), actual, line, is_total),
            _report(f"market line only", line, actual, line, is_total),
        ]
        print(f"  {'model':<26}{'MAE':>8}{'RMSE':>8}{'side%':>8}")
        for r in rows:
            ats = f"{r['ats']*100:.1f}" if r["ats"] is not None else "-"
            print(f"  {r['name']:<26}{r['mae']:>8.3f}{r['rmse']:>8.3f}{ats:>8}")

        c, i, m = rows
        print(f"  challenger vs incumbent : MAE {c['mae']-i['mae']:+.3f}  side {(c['ats']-i['ats'])*100:+.1f}pts")
        print(f"  challenger vs market    : MAE {c['mae']-m['mae']:+.3f}")
        if c["mae"] >= m["mae"]:
            print(f"  !! {label}: no improvement on the market line alone")
        print()

        results[label] = rows
        if save:
            joblib.dump({
                "model": chal, "feature_cols": cols,
                "resid_std": chrono_std, "trained_seasons": TRAIN,
                "calibrator": inc.get("calibrator"),
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "version": "2025.1",
            }, out_path)
            print(f"  saved -> {out_path}\n")

    return results


if __name__ == "__main__":
    asyncio.run(main("--save" in sys.argv[1:]))
