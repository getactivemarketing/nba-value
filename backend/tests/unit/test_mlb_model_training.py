"""Trainer accepts an explicit holdout and excludes game_date from features."""

import numpy as np
import pandas as pd
import pytest

from src.services.mlb.model_training import HAS_LIGHTGBM, MLBModelTrainer

pytestmark = pytest.mark.skipif(not HAS_LIGHTGBM, reason="lightgbm not installed")


def synthetic_df(n, seed):
    rng = np.random.default_rng(seed)
    f1 = rng.normal(4.5, 1.0, n)
    f2 = rng.normal(4.5, 1.0, n)
    return pd.DataFrame({
        "home_runs_per_game": f1,
        "away_runs_per_game": f2,
        "total_runs": (f1 + f2 + rng.normal(0, 1.5, n)).round(),
        "run_diff": rng.integers(-5, 6, n),
        "home_win": rng.integers(0, 2, n),
        "season": 2026,
        "game_id": np.arange(n),
        "game_date": pd.date_range("2026-04-01", periods=n).strftime("%Y-%m-%d"),
    })


def test_explicit_holdout_used_for_eval():
    trainer = MLBModelTrainer(model_dir="/tmp")
    train_df = synthetic_df(400, seed=1)
    test_df = synthetic_df(100, seed=2)
    model, feature_cols, metrics = trainer.train_totals_model(train_df, test_df=test_df)
    assert "game_date" not in feature_cols
    assert metrics["mae"] > 0


def test_internal_split_still_works():
    trainer = MLBModelTrainer(model_dir="/tmp")
    model, feature_cols, metrics = trainer.train_totals_model(synthetic_df(500, seed=3))
    assert "game_date" not in feature_cols
