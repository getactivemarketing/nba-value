"""Retrain on 2024-2026 with bullpen + FIP, and compare honestly.

The 2026-only attempt failed on 1,124 training games with num_leaves=31 — the
hyperparameters of a 4,931-game model applied to a quarter of the data. I
concluded "not enough data for 33 features" when the defensible conclusion was
"not at these hyperparameters". This run fixes both halves: ~6,000 games, and
a small hyperparameter sweep instead of one inherited configuration.

Comparisons, all on the SAME chronological holdout:
  challenger  2024-26, 41 features incl. bullpen + FIP
  no-new      identical data and tuning, WITHOUT bullpen/FIP — isolates
              whether the new features earn their place, rather than the
              extra seasons doing all the work
  incumbent   mlb_run_diff_v1, 2024-25, 28 features
  baseline    predict the training mean

Nothing is promoted here. This prints evidence for the gate in
docs/engineering/03-model-governance.md §4.

    python -m src.tasks.retrain_mlb_history [--save]
"""

import sys

import joblib
import numpy as np
import pandas as pd

from src.tasks.build_mlb_training_history import FEATURES

DATA = "models/mlb_training_history.parquet"
INCUMBENT = "models/mlb_run_diff_v1.joblib"
OUT = "models/mlb_run_diff_history.joblib"

NEW_FEATURES = [f for f in FEATURES if "bullpen" in f or "fip" in f]

# Small sweep rather than one inherited configuration. Leaf counts are the
# main overfitting lever on a few thousand rows.
GRID = [
    {"num_leaves": 7, "learning_rate": 0.03, "min_child_samples": 40},
    {"num_leaves": 15, "learning_rate": 0.03, "min_child_samples": 30},
    {"num_leaves": 31, "learning_rate": 0.05, "min_child_samples": 20},
]
BASE = {
    "objective": "regression", "metric": "rmse", "boosting_type": "gbdt",
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
    "verbose": -1, "seed": 42,
}


def _metrics(name, pred, actual):
    decided = actual != 0
    return {
        "name": name,
        "mae": float(np.mean(np.abs(pred - actual))),
        "rmse": float(np.sqrt(np.mean((pred - actual) ** 2))),
        "hit": float(np.mean((pred[decided] > 0) == (actual[decided] > 0))),
    }


def _fit(lgb, cols, tr, va, ytr, yva, params):
    return lgb.train(
        {**BASE, **params},
        lgb.Dataset(tr[cols].astype(float), label=ytr),
        num_boost_round=2000,
        valid_sets=[lgb.Dataset(va[cols].astype(float), label=yva)],
        callbacks=[lgb.early_stopping(80, verbose=False)],
    )


def main(save: bool = False) -> dict:
    import lightgbm as lgb

    df = pd.read_parquet(DATA).sort_values("game_date").reset_index(drop=True)
    y = df["target_run_diff"].to_numpy(float)

    # Chronological: train -> validate (for tuning + early stopping) -> holdout.
    # The holdout is touched exactly once, by the final comparison.
    n = len(df)
    i_tr, i_va = int(n * 0.70), int(n * 0.85)
    tr, va, ho = df.iloc[:i_tr], df.iloc[i_tr:i_va], df.iloc[i_va:]
    ytr, yva, yho = y[:i_tr], y[i_tr:i_va], y[i_va:]

    print(f"train {len(tr)} ({tr.game_date.min()} -> {tr.game_date.max()})")
    print(f"valid {len(va)} ({va.game_date.min()} -> {va.game_date.max()})")
    print(f"hold  {len(ho)} ({ho.game_date.min()} -> {ho.game_date.max()})")
    print(f"features {len(FEATURES)}  (new: {len(NEW_FEATURES)})\n")

    print(f"{'tuning (validation RMSE)':<34}{'leaves':>8}{'rmse':>9}")
    best, best_rmse = None, 1e9
    for params in GRID:
        m = _fit(lgb, FEATURES, tr, va, ytr, yva, params)
        r = float(np.sqrt(np.mean((m.predict(va[FEATURES].astype(float)) - yva) ** 2)))
        print(f"{'':<34}{params['num_leaves']:>8}{r:>9.4f}")
        if r < best_rmse:
            best, best_rmse = params, r
    print(f"\nchosen: {best}\n")

    reduced = [f for f in FEATURES if f not in NEW_FEATURES]
    chal = _fit(lgb, FEATURES, tr, va, ytr, yva, best)
    nonew = _fit(lgb, reduced, tr, va, ytr, yva, best)

    inc = joblib.load(INCUMBENT)
    rows = [
        _metrics("challenger (41f, bullpen+FIP)", chal.predict(ho[FEATURES].astype(float)), yho),
        _metrics("same minus bullpen/FIP (33f)", nonew.predict(ho[reduced].astype(float)), yho),
        _metrics("incumbent (2024-25, 28f)", inc["model"].predict(ho[inc["feature_cols"]].astype(float)), yho),
        _metrics("baseline (train mean)", np.full(len(yho), float(np.mean(ytr))), yho),
    ]

    print(f"{'model':<34}{'MAE':>8}{'RMSE':>8}{'hit%':>8}")
    for r in rows:
        print(f"{r['name']:<34}{r['mae']:>8.3f}{r['rmse']:>8.3f}{r['hit']*100:>8.1f}")

    c, nn, i, b = rows
    print(f"\nnew features earn?  RMSE {c['rmse']-nn['rmse']:+.4f}  hit {(c['hit']-nn['hit'])*100:+.1f}pts")
    print(f"vs incumbent     :  RMSE {c['rmse']-i['rmse']:+.4f}  hit {(c['hit']-i['hit'])*100:+.1f}pts")
    print(f"vs baseline      :  RMSE {c['rmse']-b['rmse']:+.4f}")
    if c["rmse"] >= b["rmse"]:
        print("\n!! challenger does not beat the mean — NO SKILL")

    imp = pd.DataFrame({"feature": FEATURES, "gain": chal.feature_importance("gain")}) \
        .sort_values("gain", ascending=False).reset_index(drop=True)
    print("\ntop 12 by gain:")
    for _, r in imp.head(12).iterrows():
        mark = "  <- NEW" if r.feature in NEW_FEATURES else ""
        print(f"  {r.feature:<24}{r.gain:>12.0f}{mark}")
    print("\nwhere the new features ranked:")
    for f in NEW_FEATURES:
        rank = int(imp.index[imp.feature == f][0]) + 1
        print(f"  {f:<24}rank {rank}/{len(FEATURES)}")

    if save:
        joblib.dump({
            "model": chal, "feature_cols": FEATURES,
            "metrics": {k: c[k] for k in ("mae", "rmse", "hit")},
            "trained_at": str(tr.game_date.max()), "version": "history.1",
            "params": best,
        }, OUT)
        print(f"\nsaved -> {OUT}")

    return {"rows": rows, "params": best}


if __name__ == "__main__":
    main("--save" in sys.argv[1:])
