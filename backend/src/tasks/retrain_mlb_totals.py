"""Retrain the MLB totals model on 2024-2026 data with a time-based holdout.

The current mlb_totals_v1.joblib was trained 2026-02-09 and has never seen
the 2026 season. Gate: v2 must beat v1 on holdout MAE (Jun 1 - Jul 5 2026)
or it is not saved. Over/under hit rate vs recorded snapshot lines is
reported as informational (only ~177 holdout games have a recorded line).

Usage:
    python3 -m src.tasks.retrain_mlb_totals              # eval, then save v2 if it wins
    python3 -m src.tasks.retrain_mlb_totals --eval-only  # eval only, never save
"""

import asyncio
import sys
from pathlib import Path

import joblib
import psycopg2
from sklearn.metrics import mean_absolute_error

from src.services.mlb.model_training import MLBModelTrainer
from src.tasks.prediction_tracker import DB_URL

SEASONS = [2024, 2025, 2026]
HOLDOUT_START = "2026-06-01"
V1_PATH = Path("models/mlb_totals_v1.joblib")
V2_PATH = Path("models/mlb_totals_v2.joblib")


def hit_rate(pred_by_game_id: dict) -> str:
    """Over/under accuracy vs snapshot best_total_line where recorded."""
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    cur.execute(
        """SELECT game_id, best_total_line, home_score + away_score
           FROM mlb_prediction_snapshots
           WHERE game_date >= %s AND home_score IS NOT NULL
             AND best_total_line IS NOT NULL""",
        (HOLDOUT_START,),
    )
    hits = misses = 0
    for game_id, line, actual_total in cur.fetchall():
        pred = pred_by_game_id.get(str(game_id))
        if pred is None or float(actual_total) == float(line):  # unmatched or push
            continue
        predicted_over = pred > float(line)
        actual_over = float(actual_total) > float(line)
        hits += predicted_over == actual_over
        misses += predicted_over != actual_over
    conn.close()
    n = hits + misses
    return f"{hits}/{n} = {100 * hits / n:.1f}%" if n else "no matchable games"


async def main() -> int:
    eval_only = "--eval-only" in sys.argv
    trainer = MLBModelTrainer()

    df = await trainer.collect_training_data(SEASONS)
    df = df[df["game_date"] != ""].sort_values("game_date").reset_index(drop=True)
    train_df = df[df["game_date"] < HOLDOUT_START]
    holdout_df = df[df["game_date"] >= HOLDOUT_START]
    print(f"train={len(train_df)} holdout={len(holdout_df)} (holdout from {HOLDOUT_START})")

    # v2 candidate: train pre-holdout, evaluate on holdout
    model, feature_cols, metrics = trainer.train_totals_model(train_df, test_df=holdout_df)
    v2_mae = metrics["mae"]

    # v1 baseline on the same holdout
    v1 = joblib.load(V1_PATH)
    X_hold = holdout_df.reindex(columns=v1["feature_cols"]).fillna(0)
    v1_pred = v1["model"].predict(X_hold)
    v1_mae = mean_absolute_error(holdout_df["total_runs"], v1_pred)

    v2_pred = model.predict(holdout_df.reindex(columns=feature_cols).fillna(0))
    game_ids = holdout_df["game_id"].astype(str).tolist()
    print(f"v1 holdout MAE={v1_mae:.3f}  hit-rate {hit_rate(dict(zip(game_ids, v1_pred)))}")
    print(f"v2 holdout MAE={v2_mae:.3f}  hit-rate {hit_rate(dict(zip(game_ids, v2_pred)))}")

    if v2_mae >= v1_mae:
        print("GATE FAIL: v2 does not beat v1 on holdout MAE — not saving.")
        return 1
    if eval_only:
        print("GATE PASS (eval-only, not saving).")
        return 0

    # Final model: retrain on ALL data (legacy internal split for early stopping)
    final_model, final_cols, final_metrics = trainer.train_totals_model(df)
    joblib.dump(
        {"model": final_model, "feature_cols": final_cols, "metrics": final_metrics},
        V2_PATH,
    )
    print(f"GATE PASS: saved {V2_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
