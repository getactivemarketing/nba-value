"""Margin of Victory (MOV) prediction model using LightGBM."""

import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger()

# Try to import LightGBM, but allow running without it
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    lgb = None


# Feature names used in the model
FEATURE_NAMES = [
    # Home team offensive ratings (rolling windows)
    "home_ortg_5", "home_ortg_10", "home_ortg_20",
    # Home team defensive ratings
    "home_drtg_5", "home_drtg_10", "home_drtg_20",
    # Away team offensive ratings
    "away_ortg_5", "away_ortg_10", "away_ortg_20",
    # Away team defensive ratings
    "away_drtg_5", "away_drtg_10", "away_drtg_20",
    # Pace factors
    "home_pace_10", "away_pace_10",
    # Rest advantage
    "home_rest_days", "away_rest_days",
    # Back-to-back flags
    "home_b2b", "away_b2b",
    # Win percentage
    "home_win_pct_10", "away_win_pct_10",
]

# Default MOV standard deviation (for probability conversion)
DEFAULT_MOV_STD = 12.0


@dataclass
class MOVPrediction:
    """Result of MOV prediction."""

    predicted_mov: float  # Predicted home margin (positive = home favored)
    mov_std: float  # Standard deviation for probability conversion
    confidence: float  # Model confidence (0-1)
    features_used: list[str]


class MOVModel:
    """
    Margin of Victory prediction model.

    Uses LightGBM to predict the expected margin of victory for the home team.
    All spread and moneyline probabilities are derived from this prediction.
    """

    def __init__(self, model_path: str | Path | None = None):
        """
        Initialize MOV model.

        Args:
            model_path: Path to saved model file. If None, uses default location.
        """
        self.model = None
        self.model_path = model_path
        self.is_trained = False
        self.mov_std = DEFAULT_MOV_STD

        # Try to load existing model
        if model_path:
            self.load(model_path)

    def predict(self, features: dict[str, float]) -> MOVPrediction:
        """
        Predict margin of victory for a game.

        Args:
            features: Dictionary of feature name -> value

        Returns:
            MOVPrediction with predicted margin and confidence
        """
        if not self.is_trained or self.model is None:
            # Return baseline prediction if model not trained
            return self._baseline_prediction(features)

        # Prepare feature vector
        feature_vector = self._prepare_features(features)

        # Get prediction
        predicted_mov = float(self.model.predict([feature_vector])[0])

        # Calculate confidence based on feature completeness
        features_present = sum(1 for f in FEATURE_NAMES if f in features and features[f] is not None)
        feature_completeness = features_present / len(FEATURE_NAMES)

        return MOVPrediction(
            predicted_mov=predicted_mov,
            mov_std=self.mov_std,
            confidence=feature_completeness,
            features_used=[f for f in FEATURE_NAMES if f in features],
        )

    def _baseline_prediction(self, features: dict[str, float]) -> MOVPrediction:
        """
        Generate baseline prediction when model is not trained.

        Uses a blend of recent (L10) and season-long net ratings:
        - 60% weight on L10 (recency matters)
        - 40% weight on season (overall talent level)

        MOV ≈ (home_blended_net - away_blended_net) * 0.3 + home_court_advantage
        Home court advantage in NBA is roughly 2-3 points.
        """
        HOME_COURT_ADV = 2.5
        L10_WEIGHT = 0.6
        SEASON_WEIGHT = 0.4

        def get_blended_net(prefix: str) -> float:
            """Get blended net rating for a team (L10 + season)."""
            net_l10 = features.get(f"{prefix}_net_rtg_10")
            net_season = features.get(f"{prefix}_net_rtg_season")

            # If we have both, blend them
            if net_l10 is not None and net_season is not None:
                return L10_WEIGHT * net_l10 + SEASON_WEIGHT * net_season

            # If only one is available, use it
            if net_l10 is not None:
                return net_l10
            if net_season is not None:
                return net_season

            # Fallback to calculating from ortg/drtg
            ortg = features.get(f"{prefix}_ortg_10") or features.get(f"{prefix}_ortg_season")
            drtg = features.get(f"{prefix}_drtg_10") or features.get(f"{prefix}_drtg_season")
            if ortg is not None and drtg is not None:
                return ortg - drtg

            return 0.0  # Neutral if no data

        home_net = get_blended_net("home")
        away_net = get_blended_net("away")

        # Simple MOV estimate
        # Each point of net rating differential ≈ 0.3 points per game
        predicted_mov = (home_net - away_net) * 0.3 + HOME_COURT_ADV

        # Adjust for rest
        home_rest = features.get("home_rest_days", 1)
        away_rest = features.get("away_rest_days", 1)
        rest_adj = (home_rest - away_rest) * 0.5  # Rest advantage
        predicted_mov += rest_adj

        # Back-to-back penalty
        if features.get("home_b2b"):
            predicted_mov -= 2.0
        if features.get("away_b2b"):
            predicted_mov += 2.0

        features_used = [f for f in FEATURE_NAMES if f in features]

        return MOVPrediction(
            predicted_mov=predicted_mov,
            mov_std=DEFAULT_MOV_STD,
            confidence=0.5,  # Lower confidence for baseline
            features_used=features_used,
        )

    def _prepare_features(self, features: dict[str, float]) -> list[float]:
        """Prepare feature vector in correct order."""
        return [features.get(f, 0.0) or 0.0 for f in FEATURE_NAMES]

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_split: float = 0.2,
        **lgb_params,
    ) -> dict[str, Any]:
        """
        Train the MOV model.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (home margin)
            validation_split: Fraction of data to use for validation
            **lgb_params: Additional LightGBM parameters

        Returns:
            Training metrics
        """
        if not HAS_LIGHTGBM:
            raise ImportError("LightGBM is required for training. Install with: pip install lightgbm")

        # Split data
        n_val = int(len(X) * validation_split)
        X_train, X_val = X[:-n_val], X[-n_val:]
        y_train, y_val = y[:-n_val], y[-n_val:]

        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES)
        val_data = lgb.Dataset(X_val, label=y_val, feature_name=FEATURE_NAMES, reference=train_data)

        # Default parameters (tuned for NBA MOV prediction)
        default_params = {
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
        default_params.update(lgb_params)

        # Train
        callbacks = [
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100),
        ]

        self.model = lgb.train(
            default_params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, val_data],
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )

        self.is_trained = True

        # Calculate MOV std from validation predictions
        val_preds = self.model.predict(X_val)
        residuals = y_val - val_preds
        self.mov_std = float(np.std(residuals))

        # Calculate metrics
        train_preds = self.model.predict(X_train)
        metrics = {
            "train_rmse": float(np.sqrt(np.mean((y_train - train_preds) ** 2))),
            "val_rmse": float(np.sqrt(np.mean((y_val - val_preds) ** 2))),
            "mov_std": self.mov_std,
            "n_train": len(X_train),
            "n_val": len(X_val),
            "best_iteration": self.model.best_iteration,
        }

        logger.info("MOV model trained", **metrics)
        return metrics

    def save(self, path: str | Path) -> None:
        """Save model to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "model": self.model,
            "mov_std": self.mov_std,
            "is_trained": self.is_trained,
            "feature_names": FEATURE_NAMES,
        }

        with open(path, "wb") as f:
            pickle.dump(model_data, f)

        logger.info("MOV model saved", path=str(path))

    def load(self, path: str | Path) -> None:
        """Load model from file."""
        path = Path(path)

        if not path.exists():
            logger.warning("Model file not found", path=str(path))
            return

        with open(path, "rb") as f:
            model_data = pickle.load(f)

        self.model = model_data["model"]
        self.mov_std = model_data.get("mov_std", DEFAULT_MOV_STD)
        self.is_trained = model_data.get("is_trained", True)

        logger.info("MOV model loaded", path=str(path))

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance scores."""
        if not self.is_trained or self.model is None:
            return {}

        importance = self.model.feature_importance(importance_type="gain")
        return dict(zip(FEATURE_NAMES, importance))
