"""Rolling-origin retrain: OOF residuals, OOF calibration, SHADOW artifact.

Replaces the random-split training path. Every estimate that feeds a
probability now comes from predictions made on unseen future time.

  folds        expanding window — season S predicted by a model trained only
               on seasons < S
  resid_std    estimated from the pooled OUT-OF-FOLD residuals, never from an
               interleaved random split
  calibration  measured on OOF probabilities only
  2025         NEVER passed to any fold. Reserved as the untouched final
               holdout and reported separately, once.
  artifact     2010-2024 challenger written to a VERSIONED SHADOW path.
               Production paths are not touched; promotion is a config change.

Why this matters concretely: the random split understated resid_std by 1.6%
(MOV) and 4.3% (totals). resid_std is the probability scale, so totals
probabilities were ~4% too confident before a single game was played.

    python -m src.tasks.nfl_rolling_retrain [--save]
"""

import asyncio
import sys
from datetime import datetime, timezone

import joblib
import numpy as np

from src.database import async_session_maker
from src.services.nfl.rolling_eval import expanding_folds, summarize_predictions
from src.services.nfl.training_data import (
    MOV_FEATURES, TOTALS_FEATURES, build_feature_frame, load_training_frames,
)

ALL = list(range(2010, 2026))
TRAINABLE = list(range(2010, 2025))   # 2025 excluded on purpose
FIRST_TEST = 2018                     # >= 8 seasons of history before scoring
FINAL_HOLDOUT = 2025

SHADOW_MOV = "models/nfl_mov_shadow_2025.1.joblib"
SHADOW_TOT = "models/nfl_totals_shadow_2025.1.joblib"
INCUMBENT = {"mov": "models/nfl_mov_v1.joblib", "totals": "models/nfl_totals_v1.joblib"}


def _fit(lgb, params, frame, cols, target):
    from src.services.nfl.model_training import _PARAMS
    return lgb.train(
        params or _PARAMS,
        lgb.Dataset(frame[cols].fillna(0.0), label=frame[target].astype(float)),
        num_boost_round=600,
    )


async def main(save: bool = False) -> dict:
    import lightgbm as lgb
    from src.services.nfl.model_training import _PARAMS

    async with async_session_maker() as session:
        frames = await load_training_frames(session, ALL)
    frame = build_feature_frame(*frames)

    trainable = frame[frame["season"].isin(TRAINABLE)]
    holdout = frame[frame["season"] == FINAL_HOLDOUT]
    print(f"trainable {len(trainable)} ({TRAINABLE[0]}-{TRAINABLE[-1]})   "
          f"final holdout {len(holdout)} ({FINAL_HOLDOUT}, untouched)\n")

    folds = expanding_folds(TRAINABLE, first_test=FIRST_TEST)
    print(f"rolling-origin folds: {len(folds)}  "
          f"({folds[0][1]}..{folds[-1][1]}, train window {len(folds[0][0])}->{len(folds[-1][0])} seasons)\n")

    out: dict = {}
    for label, cols, target, line_col, shadow_path in (
        ("MOV", MOV_FEATURES, "margin", "spread_line", SHADOW_MOV),
        ("TOTALS", TOTALS_FEATURES, "total", "total_line", SHADOW_TOT),
    ):
        print(f"=== {label} ===")

        oof_pred, oof_actual, oof_line, oof_season = [], [], [], []
        for train_seasons, test_season in folds:
            tr = trainable[trainable["season"].isin(train_seasons)]
            te = trainable[trainable["season"] == test_season]
            if len(te) == 0:
                continue
            model = _fit(lgb, _PARAMS, tr, cols, target)
            oof_pred.append(model.predict(te[cols].fillna(0.0)))
            oof_actual.append(te[target].astype(float).values)
            oof_line.append(te[line_col].astype(float).values)
            oof_season.append(np.full(len(te), test_season))

        pred = np.concatenate(oof_pred)
        actual = np.concatenate(oof_actual)
        line = np.concatenate(oof_line)
        season = np.concatenate(oof_season)

        # THE key number: residual scale from out-of-fold predictions only.
        oof_std = float(np.std(actual - pred))
        inc = joblib.load(INCUMBENT[label.lower()])
        print(f"  resid_std  OOF {oof_std:.3f}   incumbent(random-split) {inc['resid_std']:.3f}"
              f"   ({(oof_std/inc['resid_std'] - 1)*100:+.1f}%)")

        m = summarize_predictions(pred, actual, line, resid_std=oof_std)
        print(f"  OOF {m['n']} games  MAE {m['mae']:.3f}  RMSE {m['rmse']:.3f}  "
              f"Brier {m['brier']:.4f}  logloss {m['log_loss']:.4f}  "
              f"calib {m['calibration_error']:.4f}  side {m['side_accuracy']*100:.1f}%")

        mkt = summarize_predictions(line, actual, line, resid_std=oof_std)
        print(f"  market-only baseline    MAE {mkt['mae']:.3f}  RMSE {mkt['rmse']:.3f}")

        print("  by season:")
        for s in sorted(set(season.tolist())):
            k = season == s
            sm = summarize_predictions(pred[k], actual[k], line[k], resid_std=oof_std)
            print(f"    {int(s)}  n={sm['n']:>3}  MAE {sm['mae']:.2f}  "
                  f"side {sm['side_accuracy']*100:.1f}%  calib {sm['calibration_error']:.3f}")

        if save:
            final = _fit(lgb, _PARAMS, trainable, cols, target)
            joblib.dump({
                "model": final,
                "feature_cols": cols,
                # OOF residual scale, not an in-sample or random-split one.
                "resid_std": oof_std,
                "trained_seasons": TRAINABLE,
                "calibrator": inc.get("calibrator"),
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "version": "shadow-2025.1",
                "validation": "rolling-origin expanding window",
                "oof_metrics": m,
                "final_holdout_season": FINAL_HOLDOUT,
            }, shadow_path)
            print(f"  SHADOW saved -> {shadow_path}")
        print()
        out[label] = {"oof": m, "market": mkt, "resid_std": oof_std}

    print("2025 remains untouched — no fold trained on it, no metric fitted to it.")
    return out


if __name__ == "__main__":
    asyncio.run(main("--save" in sys.argv[1:]))
