"""Spread Model V2 - Uses ATS tendencies, home/away splits, and situational features."""

import pickle
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import structlog

logger = structlog.get_logger()

# Feature names expected by the model (must match training)
# Note: Percentage features are CENTERED around 0.5 to avoid bias
SPREAD_V2_FEATURE_NAMES = [
    # Basic rolling stats (6 features)
    'home_ppg', 'home_opp_ppg', 'home_net_ppg',
    'away_ppg', 'away_opp_ppg', 'away_net_ppg',
    # Scoring variance (2 features)
    'home_scoring_std', 'away_scoring_std',
    # Win rate L10 - CENTERED (2 features)
    'home_win_pct_l10_centered', 'away_win_pct_l10_centered',
    # ATS tendencies - CENTERED (2 features)
    'home_ats_pct_centered', 'away_ats_pct_centered',
    # Home/away performance - CENTERED (4 features)
    'home_home_win_pct_centered', 'home_away_win_pct_centered',
    'away_home_win_pct_centered', 'away_away_win_pct_centered',
    # Rest/fatigue (5 features)
    'home_rest_days', 'away_rest_days',
    'rest_advantage', 'home_b2b', 'away_b2b',
    # Schedule density (2 features)
    'home_games_last_7', 'away_games_last_7',
]

# Original feature names (before centering) - used to fetch from team_stats
SPREAD_V2_RAW_FEATURES = [
    'home_ppg', 'home_opp_ppg', 'home_net_ppg',
    'away_ppg', 'away_opp_ppg', 'away_net_ppg',
    'home_scoring_std', 'away_scoring_std',
    'home_win_pct_l10', 'away_win_pct_l10',
    'home_ats_pct_l10', 'away_ats_pct_l10',
    'home_home_win_pct', 'home_away_win_pct',
    'away_home_win_pct', 'away_away_win_pct',
    'home_rest_days', 'away_rest_days',
    'rest_advantage', 'home_b2b', 'away_b2b',
    'home_games_last_7', 'away_games_last_7',
]

# Features that need centering (subtract 0.5)
FEATURES_TO_CENTER = [
    'home_win_pct_l10', 'away_win_pct_l10',
    'home_ats_pct_l10', 'away_ats_pct_l10',
    'home_home_win_pct', 'home_away_win_pct',
    'away_home_win_pct', 'away_away_win_pct',
]

# Minimum edge threshold for spread bets (from backtesting: 4+ points = 52.6% win rate)
SPREAD_MIN_EDGE_POINTS = 4.0


@dataclass
class SpreadV2Prediction:
    """Result of spread model v2 prediction."""
    predicted_mov: float  # Predicted home margin (positive = home wins by this much)
    mov_std: float  # Standard deviation for probability conversion
    edge_points: float  # Difference between our prediction and the spread
    confidence: float  # Feature completeness (0-1)
    meets_threshold: bool  # True if edge >= SPREAD_MIN_EDGE_POINTS


class SpreadModelV2:
    """
    Spread prediction model v2 with ATS tendencies and situational features.

    Key improvements over v1:
    - Uses ATS % (teams that cover often tend to regress)
    - Uses home/away performance splits
    - Uses schedule fatigue (games in last 7 days)

    Backtesting showed 52.6% win rate at 4+ point edges (above 52.4% breakeven).
    """

    def __init__(self, model_path: str | Path | None = None):
        """Initialize spread model v2."""
        self.model = None
        self.mov_std = 14.0  # Default from training
        self.is_trained = False
        self.feature_names = SPREAD_V2_FEATURE_NAMES

        if model_path:
            self.load(model_path)

    def predict(self, features: dict[str, float], spread_line: float | None = None) -> SpreadV2Prediction:
        """
        Predict home margin of victory.

        Args:
            features: Dictionary of feature values (with home_/away_ prefixes)
            spread_line: Current spread line (negative = home favored)
                        Used to calculate edge

        Returns:
            SpreadV2Prediction with predicted MOV and edge
        """
        if not self.is_trained or self.model is None:
            return self._baseline_prediction(features, spread_line)

        # Prepare feature vector
        feature_vector = self._prepare_features(features)

        # Check for NaN/Inf values
        if np.any(np.isnan(feature_vector)) or np.any(np.isinf(feature_vector)):
            logger.warning("Invalid feature values, using baseline", features=features)
            return self._baseline_prediction(features, spread_line)

        # Get prediction
        predicted_mov = float(self.model.predict([feature_vector])[0])

        # Calculate edge vs spread
        edge_points = 0.0
        if spread_line is not None:
            # spread_line is from home perspective: negative = home favored
            # edge_points = predicted_mov - (-spread_line) = predicted_mov + spread_line
            # If we predict home wins by 8 and spread is -5 (home favored by 5),
            # our edge is 8 - 5 = 3 points (we think home wins by more than spread says)
            edge_points = predicted_mov - (-spread_line)

        # Calculate confidence based on feature completeness
        features_present = sum(1 for f in self.feature_names if f in features and features.get(f) is not None)
        confidence = features_present / len(self.feature_names)

        return SpreadV2Prediction(
            predicted_mov=predicted_mov,
            mov_std=self.mov_std,
            edge_points=abs(edge_points),
            confidence=confidence,
            meets_threshold=abs(edge_points) >= SPREAD_MIN_EDGE_POINTS,
        )

    def _baseline_prediction(self, features: dict[str, float], spread_line: float | None) -> SpreadV2Prediction:
        """Generate baseline prediction when model not available."""
        # Use net PPG differential as simple estimate
        home_net = features.get('home_net_ppg', 0) or features.get('home_net_rtg_10', 0) or 0
        away_net = features.get('away_net_ppg', 0) or features.get('away_net_rtg_10', 0) or 0

        # Home court advantage ~2.5 points
        predicted_mov = (home_net - away_net) + 2.5

        # Adjust for rest
        home_rest = features.get('home_rest_days', 1) or 1
        away_rest = features.get('away_rest_days', 1) or 1
        predicted_mov += (home_rest - away_rest) * 0.5

        # B2B penalty
        if features.get('home_b2b'):
            predicted_mov -= 2.0
        if features.get('away_b2b'):
            predicted_mov += 2.0

        edge_points = 0.0
        if spread_line is not None:
            edge_points = predicted_mov - (-spread_line)

        return SpreadV2Prediction(
            predicted_mov=predicted_mov,
            mov_std=14.0,
            edge_points=abs(edge_points),
            confidence=0.3,  # Low confidence for baseline
            meets_threshold=abs(edge_points) >= SPREAD_MIN_EDGE_POINTS,
        )

    def _prepare_features(self, features: dict[str, float]) -> list[float]:
        """Prepare feature vector in correct order, centering percentage features."""
        vector = []

        # Map from centered feature names to raw feature names
        centered_to_raw = {
            'home_win_pct_l10_centered': 'home_win_pct_l10',
            'away_win_pct_l10_centered': 'away_win_pct_l10',
            'home_ats_pct_centered': 'home_ats_pct_l10',
            'away_ats_pct_centered': 'away_ats_pct_l10',
            'home_home_win_pct_centered': 'home_home_win_pct',
            'home_away_win_pct_centered': 'home_away_win_pct',
            'away_home_win_pct_centered': 'away_home_win_pct',
            'away_away_win_pct_centered': 'away_away_win_pct',
        }

        for f in self.feature_names:
            # Check if this is a centered feature
            if f in centered_to_raw:
                raw_name = centered_to_raw[f]
                val = features.get(raw_name)
                if val is None:
                    val = 0.5  # Default to 50%
                # Center around 0.5
                val = float(val) - 0.5
            else:
                val = features.get(f)
                if val is None:
                    # Use sensible defaults
                    if 'ppg' in f:
                        val = 110.0
                    elif 'std' in f:
                        val = 10.0
                    elif 'rest' in f:
                        val = 1
                    elif 'b2b' in f:
                        val = 0
                    elif 'games_last_7' in f:
                        val = 3
                    elif 'advantage' in f:
                        val = 0
                    else:
                        val = 0.0
            vector.append(float(val))
        return vector

    def load(self, path: str | Path) -> None:
        """Load model from pickle file."""
        path = Path(path)
        if not path.exists():
            logger.warning("Spread model v2 not found", path=str(path))
            return

        try:
            with open(path, 'rb') as f:
                model_data = pickle.load(f)

            self.model = model_data['model']
            self.mov_std = model_data.get('mov_std', 14.0)
            self.feature_names = model_data.get('feature_names', SPREAD_V2_FEATURE_NAMES)
            self.is_trained = True

            logger.info(
                "Loaded spread model v2",
                mov_std=self.mov_std,
                training_games=model_data.get('training_games'),
                avg_mae=model_data.get('avg_mae'),
            )
        except Exception as e:
            logger.error("Failed to load spread model v2", error=str(e))

    def save(self, path: str | Path) -> None:
        """Save model to pickle file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'model': self.model,
            'mov_std': self.mov_std,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained,
            'model_type': 'ridge_spread_v2',
        }

        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info("Saved spread model v2", path=str(path))


# Singleton instance
_spread_model_v2: SpreadModelV2 | None = None

# Model path
MODEL_DIR = Path(__file__).parent.parent.parent.parent / "models"
SPREAD_V2_MODEL_PATH = MODEL_DIR / "spread_model_v2.pkl"


def get_spread_model_v2() -> SpreadModelV2:
    """Get or create the spread model v2 singleton."""
    global _spread_model_v2
    if _spread_model_v2 is None:
        _spread_model_v2 = SpreadModelV2(SPREAD_V2_MODEL_PATH)
    return _spread_model_v2
