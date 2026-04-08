"""Retrain the MLB run differential model with V2 features.

V2 features add first inning scoring data to the existing 28 features:
- home/away_first_inning_score_pct
- home/away_first_inning_runs_avg

Usage:
    python -m src.tasks.retrain_mlb_v2 [--min-games 500] [--output models/mlb_run_diff_v2.joblib]

Requires at least --min-games completed games with predictions saved.
"""

import asyncio
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import structlog

import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.database import async_session
from src.services.mlb.features import build_training_data
from src.services.mlb.scorer import MLBScorer

logger = structlog.get_logger()

try:
    import lightgbm as lgb
    import pandas as pd
    import numpy as np
    import joblib
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


async def retrain(min_games: int = 500, output_path: str = "models/mlb_run_diff_v2.joblib") -> dict:
    """Retrain the MLB run differential model with V2 features."""
    if not HAS_DEPS:
        return {"error": "Missing dependencies: lightgbm, pandas, joblib, sklearn"}

    results = {}

    # Collect training data from all completed games with predictions
    async with async_session() as session:
        print(f"[RETRAIN] Building training data...", flush=True)
        end_date = date.today()
        # Use all available data — go back as far as we have
        start_date = date(2024, 3, 1)

        training_data = await build_training_data(session, start_date, end_date)
        results["total_rows"] = len(training_data)
        print(f"[RETRAIN] Collected {len(training_data)} training rows", flush=True)

        if len(training_data) < min_games:
            results["error"] = f"Not enough data: {len(training_data)} < {min_games} minimum"
            return results

    # Convert to DataFrame
    df = pd.DataFrame(training_data)

    # Use V2 feature names
    feature_cols = MLBScorer.V2_FEATURE_NAMES
    target_col = "target_run_diff"

    # Filter to only rows with complete features
    available_cols = [c for c in feature_cols if c in df.columns]
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        print(f"[RETRAIN] WARNING: Missing feature columns: {missing_cols}", flush=True)
        results["missing_features"] = missing_cols

    X = df[available_cols].fillna(0)
    y = df[target_col]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train LightGBM
    print(f"[RETRAIN] Training on {len(X_train)} rows, testing on {len(X_test)}", flush=True)
    model = lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    results["mae"] = float(mae)
    results["rmse"] = float(rmse)
    results["train_size"] = len(X_train)
    results["test_size"] = len(X_test)
    results["features_used"] = available_cols

    print(f"[RETRAIN] MAE: {mae:.3f}, RMSE: {rmse:.3f}", flush=True)

    # Save model
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model,
        "feature_names": available_cols,
        "mae": float(mae),
        "rmse": float(rmse),
        "trained_at": datetime.now().isoformat(),
    }, output)

    print(f"[RETRAIN] Saved model to {output}", flush=True)
    results["model_path"] = str(output)

    return results


if __name__ == '__main__':
    min_games = 500
    output = "models/mlb_run_diff_v2.joblib"

    # Parse args
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == '--min-games':
            min_games = int(args[i+1])
            i += 2
        elif args[i] == '--output':
            output = args[i+1]
            i += 2
        else:
            i += 1

    results = asyncio.run(retrain(min_games, output))
    print("\n[RETRAIN] Results:", flush=True)
    for k, v in results.items():
        if k != "features_used":
            print(f"  {k}: {v}", flush=True)
