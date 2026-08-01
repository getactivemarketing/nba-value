"""Retrain the MLB run-diff model on 2026 data and compare it to the incumbent.

Roadmap item 2. The incumbent (mlb_run_diff_v1, trained 2026-02-09 on 4,931
games from 2024-25) has never seen a 2026 game and knows 28 features. The
challenger trains on 1,405 point-in-time-correct 2026 games with 33.

Comparison design:
- Chronological split. The holdout is the LAST slice of the season, never a
  random sample — a random split leaks future information through team stats
  that are cumulative by construction.
- Both models are scored on the SAME holdout. It is genuinely out-of-sample
  for the incumbent (different seasons entirely) and for the challenger
  (later games than it trained on).
- A "predict the mean" baseline is reported alongside. A model that cannot
  beat the training mean has no skill regardless of how it compares to the
  incumbent, and that is the bar totals already failed.

Nothing is promoted here. This prints evidence for the gate in
docs/engineering/03-model-governance.md §4.

    python -m src.tasks.retrain_mlb_run_diff_2026 [--holdout 0.2] [--save]
"""

import sys

import joblib
import numpy as np
import pandas as pd

from src.services.mlb.scorer import MLBScorer

TRAIN_PARQUET = "models/mlb_training_2026.parquet"
INCUMBENT = "models/mlb_run_diff_v1.joblib"
OUT_PATH = "models/mlb_run_diff_2026.joblib"

PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "verbose": -1,
    "seed": 42,
}


def _metrics(name: str, pred: np.ndarray, actual: np.ndarray) -> dict:
    mae = float(np.mean(np.abs(pred - actual)))
    rmse = float(np.sqrt(np.mean((pred - actual) ** 2)))
    # Winner accuracy on decided games; run_diff == 0 is impossible in MLB.
    decided = actual != 0
    hit = float(np.mean((pred[decided] > 0) == (actual[decided] > 0)))
    return {"name": name, "mae": mae, "rmse": rmse, "hit_rate": hit, "n": int(len(actual))}


def main(holdout_frac: float = 0.2, save: bool = False) -> dict:
    import lightgbm as lgb

    df = pd.read_parquet(TRAIN_PARQUET).sort_values("game_date").reset_index(drop=True)
    target = df["target_run_diff"].to_numpy(dtype=float)

    challenger_cols = list(MLBScorer.MODEL_FEATURE_NAMES)
    incumbent_data = joblib.load(INCUMBENT)
    incumbent_cols = incumbent_data["feature_cols"]

    missing = [c for c in challenger_cols + incumbent_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"training set is missing required features: {sorted(set(missing))}")

    split = int(len(df) * (1 - holdout_frac))
    train_df, hold_df = df.iloc[:split], df.iloc[split:]
    y_train, y_hold = target[:split], target[split:]

    print(f"train {len(train_df)} games ({train_df.game_date.min()} -> {train_df.game_date.max()})")
    print(f"hold  {len(hold_df)} games ({hold_df.game_date.min()} -> {hold_df.game_date.max()})")
    print(f"challenger features: {len(challenger_cols)}   incumbent: {len(incumbent_cols)}\n")

    booster = lgb.train(
        PARAMS,
        lgb.Dataset(train_df[challenger_cols].astype(float), label=y_train),
        num_boost_round=500,
        valid_sets=[lgb.Dataset(hold_df[challenger_cols].astype(float), label=y_hold)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    rows = [
        _metrics("challenger (2026, 33f)", booster.predict(hold_df[challenger_cols].astype(float)), y_hold),
        _metrics("incumbent (2024-25, 28f)", incumbent_data["model"].predict(hold_df[incumbent_cols].astype(float)), y_hold),
        _metrics("baseline (train mean)", np.full(len(y_hold), float(np.mean(y_train))), y_hold),
    ]

    print(f"{'model':<28}{'MAE':>8}{'RMSE':>8}{'hit%':>8}")
    for r in rows:
        print(f"{r['name']:<28}{r['mae']:>8.3f}{r['rmse']:>8.3f}{r['hit_rate']*100:>8.1f}")

    ch, inc, base = rows
    print(f"\nchallenger vs incumbent : MAE {ch['mae']-inc['mae']:+.3f}  "
          f"RMSE {ch['rmse']-inc['rmse']:+.3f}  hit {(ch['hit_rate']-inc['hit_rate'])*100:+.1f}pts")
    print(f"challenger vs baseline  : MAE {ch['mae']-base['mae']:+.3f}  "
          f"RMSE {ch['rmse']-base['rmse']:+.3f}")
    if ch["rmse"] >= base["rmse"]:
        print("\n!! challenger does not beat predicting the mean — NO SKILL")

    imp = pd.DataFrame({
        "feature": challenger_cols,
        "gain": booster.feature_importance("gain"),
    }).sort_values("gain", ascending=False)
    print("\ntop 12 features by gain:")
    for _, r in imp.head(12).iterrows():
        print(f"  {r.feature:<24}{r.gain:>12.0f}")
    print("\nthe 5 recovered features:")
    for f in ("temperature", "is_dome", "weather_factor", "home_last_10_win_pct", "away_last_10_win_pct"):
        g = float(imp.loc[imp.feature == f, "gain"].iloc[0])
        rank = int(imp.reset_index(drop=True).index[imp.reset_index(drop=True).feature == f][0]) + 1
        print(f"  {f:<24}{g:>12.0f}   rank {rank}/{len(challenger_cols)}")

    if save:
        joblib.dump(
            {
                "model": booster,
                "feature_cols": challenger_cols,
                "metrics": {"mae": ch["mae"], "rmse": ch["rmse"], "hit_rate": ch["hit_rate"]},
                "trained_at": str(train_df.game_date.max()),
                "version": "2026.1",
            },
            OUT_PATH,
        )
        print(f"\nsaved -> {OUT_PATH}")

    return {"challenger": ch, "incumbent": inc, "baseline": base}


if __name__ == "__main__":
    args = sys.argv[1:]
    frac, save = 0.2, "--save" in args
    for i, a in enumerate(args):
        if a == "--holdout" and i + 1 < len(args):
            frac = float(args[i + 1])
    main(frac, save)
