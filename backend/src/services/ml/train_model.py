"""Train MOV model and calibration layer on historical data."""

import numpy as np
import pandas as pd
from pathlib import Path
import structlog
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats

from src.services.ml.training_data import (
    build_training_dataset,
    games_to_dataframe,
    prepare_training_arrays,
    save_training_data,
)
from src.services.ml.mov_model import MOVModel, FEATURE_NAMES, DEFAULT_MOV_STD
from src.services.ml.calibration import CalibrationLayer
from src.services.ml.probability import mov_to_spread_prob, mov_to_moneyline_prob

logger = structlog.get_logger()

# Model save paths
MODEL_DIR = Path(__file__).parent.parent.parent.parent / "models"
MOV_MODEL_PATH = MODEL_DIR / "mov_model.pkl"
CALIBRATION_PATH = MODEL_DIR / "calibration.pkl"


def train_mov_model(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
) -> tuple[MOVModel, dict]:
    """
    Train MOV model on prepared data.

    Args:
        X: Feature matrix
        y: Target vector (game margins)
        test_size: Fraction for test set

    Returns:
        Trained model and metrics dict
    """
    logger.info(f"Training MOV model on {len(X)} samples")

    # Split data chronologically (last N games for test)
    n_test = int(len(X) * test_size)
    X_train, X_test = X[:-n_test], X[-n_test:]
    y_train, y_test = y[:-n_test], y[-n_test:]

    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Create and train model
    model = MOVModel()

    # Train with custom feature names matching our data
    # Our training data has: ortg, drtg, net_rtg, pace, win_pct for home and away
    training_features = [
        'home_ortg_10', 'home_drtg_10', 'home_net_rtg_10', 'home_pace_10', 'home_win_pct_10',
        'away_ortg_10', 'away_drtg_10', 'away_net_rtg_10', 'away_pace_10', 'away_win_pct_10',
    ]

    # Train the model
    try:
        import lightgbm as lgb

        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=training_features)
        val_data = lgb.Dataset(X_test, label=y_test, feature_name=training_features, reference=train_data)

        params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "seed": 42,
        }

        callbacks = [
            lgb.early_stopping(stopping_rounds=50),
        ]

        model.model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[train_data, val_data],
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )
        model.is_trained = True

        # Calculate MOV std from residuals
        test_preds = model.model.predict(X_test)
        residuals = y_test - test_preds
        model.mov_std = float(np.std(residuals))

        # Calculate metrics
        train_preds = model.model.predict(X_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
        test_mae = mean_absolute_error(y_test, test_preds)

        metrics = {
            "train_rmse": float(train_rmse),
            "test_rmse": float(test_rmse),
            "test_mae": float(test_mae),
            "mov_std": float(model.mov_std),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "best_iteration": model.model.best_iteration,
        }

        logger.info("MOV model trained", **metrics)

    except (ImportError, OSError):
        logger.warning("LightGBM not available, using simple linear regression")
        from sklearn.linear_model import Ridge

        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, y_train)

        test_preds = ridge.predict(X_test)
        train_preds = ridge.predict(X_train)

        # Store Ridge model in a wrapper that matches our interface
        model._ridge_model = ridge
        model._training_features = training_features
        model.is_trained = True
        model.mov_std = float(np.std(y_test - test_preds))

        metrics = {
            "train_rmse": float(np.sqrt(mean_squared_error(y_train, train_preds))),
            "test_rmse": float(np.sqrt(mean_squared_error(y_test, test_preds))),
            "test_mae": float(mean_absolute_error(y_test, test_preds)),
            "mov_std": model.mov_std,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "model_type": "ridge",
        }

    return model, metrics


class RidgeMOVModel:
    """Simple Ridge-based MOV model for when LightGBM is unavailable."""

    def __init__(self, ridge_model, training_features, mov_std):
        self.model = ridge_model
        self.training_features = training_features
        self.mov_std = mov_std
        self.is_trained = True

    def predict(self, features: dict) -> 'MOVPrediction':
        from src.services.ml.mov_model import MOVPrediction

        # Extract features in training order
        feature_vector = [features.get(f, 0.0) or 0.0 for f in self.training_features]
        predicted_mov = float(self.model.predict([feature_vector])[0])

        return MOVPrediction(
            predicted_mov=predicted_mov,
            mov_std=self.mov_std,
            confidence=0.7,
            features_used=self.training_features,
        )


def train_calibration_layer(
    df: pd.DataFrame,
    model: MOVModel,
) -> tuple[CalibrationLayer, dict]:
    """
    Train calibration layer based on model predictions vs actual outcomes.

    Uses the trained MOV model to predict probabilities, then calibrates
    against actual win/cover outcomes.
    """
    logger.info("Training calibration layer")

    calibration = CalibrationLayer(method="isotonic")

    # Prepare data for spread calibration
    spread_probs = []
    spread_outcomes = []

    # Prepare data for moneyline calibration
    ml_probs = []
    ml_outcomes = []

    for _, row in df.iterrows():
        # Build feature dict matching model expectations
        features = {
            'home_ortg_10': row['home_ortg_10'],
            'home_drtg_10': row['home_drtg_10'],
            'home_net_rtg_10': row['home_net_rtg_10'],
            'home_pace_10': row['home_pace_10'],
            'home_win_pct_10': row['home_win_pct_10'],
            'away_ortg_10': row['away_ortg_10'],
            'away_drtg_10': row['away_drtg_10'],
            'away_net_rtg_10': row['away_net_rtg_10'],
            'away_pace_10': row['away_pace_10'],
            'away_win_pct_10': row['away_win_pct_10'],
        }

        # Skip if missing features
        if any(pd.isna(v) for v in features.values()):
            continue

        # Get MOV prediction
        mov_pred = model.predict(features)
        margin = row['margin']

        # Moneyline: did home team win?
        ml_prob = mov_to_moneyline_prob(mov_pred.predicted_mov, mov_pred.mov_std)
        ml_outcome = 1 if margin > 0 else 0
        ml_probs.append(ml_prob)
        ml_outcomes.append(ml_outcome)

        # Spread: use market spread if available, else estimate
        spread_line = row.get('closing_spread')
        if spread_line is None:
            # Estimate spread from home court advantage (roughly -3 for home)
            spread_line = -3.0

        spread_prob = mov_to_spread_prob(mov_pred.predicted_mov, spread_line, mov_pred.mov_std)
        spread_outcome = 1 if (margin + spread_line) > 0 else 0  # Home covers
        spread_probs.append(spread_prob)
        spread_outcomes.append(spread_outcome)

    # Fit calibrators
    ml_probs = np.array(ml_probs)
    ml_outcomes = np.array(ml_outcomes)
    spread_probs = np.array(spread_probs)
    spread_outcomes = np.array(spread_outcomes)

    ml_metrics = calibration.fit(ml_probs, ml_outcomes, "moneyline")
    spread_metrics = calibration.fit(spread_probs, spread_outcomes, "spread")

    # Use moneyline calibrator as default for totals
    calibration.calibrators["total"] = calibration.calibrators["moneyline"]
    calibration.is_fitted["total"] = True

    metrics = {
        "moneyline": {
            "brier_before": ml_metrics.brier_score_before,
            "brier_after": ml_metrics.brier_score_after,
            "n_samples": ml_metrics.n_samples,
        },
        "spread": {
            "brier_before": spread_metrics.brier_score_before,
            "brier_after": spread_metrics.brier_score_after,
            "n_samples": spread_metrics.n_samples,
        },
    }

    logger.info("Calibration layer trained", **metrics)

    return calibration, metrics


def analyze_predictions(df: pd.DataFrame, model: MOVModel) -> dict:
    """Analyze model predictions vs actual outcomes."""
    predictions = []
    actuals = []
    errors = []

    for _, row in df.iterrows():
        features = {
            'home_ortg_10': row['home_ortg_10'],
            'home_drtg_10': row['home_drtg_10'],
            'home_net_rtg_10': row['home_net_rtg_10'],
            'home_pace_10': row['home_pace_10'],
            'home_win_pct_10': row['home_win_pct_10'],
            'away_ortg_10': row['away_ortg_10'],
            'away_drtg_10': row['away_drtg_10'],
            'away_net_rtg_10': row['away_net_rtg_10'],
            'away_pace_10': row['away_pace_10'],
            'away_win_pct_10': row['away_win_pct_10'],
        }

        if any(pd.isna(v) for v in features.values()):
            continue

        mov_pred = model.predict(features)
        predictions.append(mov_pred.predicted_mov)
        actuals.append(row['margin'])
        errors.append(mov_pred.predicted_mov - row['margin'])

    predictions = np.array(predictions)
    actuals = np.array(actuals)
    errors = np.array(errors)

    # Calculate correlation
    correlation = np.corrcoef(predictions, actuals)[0, 1]

    # Home win accuracy
    pred_home_wins = predictions > 0
    actual_home_wins = actuals > 0
    home_win_acc = np.mean(pred_home_wins == actual_home_wins)

    return {
        "correlation": float(correlation),
        "home_win_accuracy": float(home_win_acc),
        "mean_error": float(np.mean(errors)),
        "std_error": float(np.std(errors)),
        "mean_abs_error": float(np.mean(np.abs(errors))),
    }


def run_full_training_pipeline():
    """Run the full training pipeline."""
    print("=" * 60)
    print("MOV MODEL TRAINING PIPELINE")
    print("=" * 60)

    # Step 1: Build training data
    print("\n[1/4] Building training dataset...")
    games = build_training_dataset(seasons=["2023-24", "2024-25"])
    df = games_to_dataframe(games)
    print(f"  Total games: {len(df)}")
    print(f"  Date range: {df['game_date'].min()} to {df['game_date'].max()}")

    # Step 2: Prepare training arrays
    print("\n[2/4] Preparing training arrays...")
    X, y, df_clean = prepare_training_arrays(df)
    print(f"  Training samples: {len(X)}")

    # Step 3: Train MOV model
    print("\n[3/4] Training MOV model...")
    model, model_metrics = train_mov_model(X, y)
    print(f"  Train RMSE: {model_metrics['train_rmse']:.2f}")
    print(f"  Test RMSE: {model_metrics['test_rmse']:.2f}")
    print(f"  MOV Std: {model_metrics['mov_std']:.2f}")

    # If Ridge was used, wrap in RidgeMOVModel for analysis
    if hasattr(model, '_ridge_model'):
        analysis_model = RidgeMOVModel(
            model._ridge_model,
            model._training_features,
            model.mov_std
        )
    else:
        analysis_model = model

    # Analyze predictions
    analysis = analyze_predictions(df_clean, analysis_model)
    print(f"  Correlation: {analysis['correlation']:.3f}")
    print(f"  Home Win Accuracy: {analysis['home_win_accuracy']:.1%}")

    # Step 4: Train calibration
    print("\n[4/4] Training calibration layer...")
    calibration, cal_metrics = train_calibration_layer(df_clean, analysis_model)
    print(f"  Moneyline Brier: {cal_metrics['moneyline']['brier_before']:.4f} -> {cal_metrics['moneyline']['brier_after']:.4f}")
    print(f"  Spread Brier: {cal_metrics['spread']['brier_before']:.4f} -> {cal_metrics['spread']['brier_after']:.4f}")

    # Save models
    print("\n[5/5] Saving models...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Save the model (handle Ridge case)
    if hasattr(model, '_ridge_model'):
        import pickle
        with open(MOV_MODEL_PATH, 'wb') as f:
            pickle.dump({
                'model': model._ridge_model,
                'training_features': model._training_features,
                'mov_std': model.mov_std,
                'is_trained': True,
                'model_type': 'ridge',
            }, f)
    else:
        model.save(MOV_MODEL_PATH)

    calibration.save(CALIBRATION_PATH)
    print(f"  MOV model saved to: {MOV_MODEL_PATH}")
    print(f"  Calibration saved to: {CALIBRATION_PATH}")

    # Save training data for reference
    save_training_data(df, MODEL_DIR / "training_data.csv")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    return model, calibration, {
        "model_metrics": model_metrics,
        "calibration_metrics": cal_metrics,
        "analysis": analysis,
    }


if __name__ == "__main__":
    model, calibration, metrics = run_full_training_pipeline()
