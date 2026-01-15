"""Probability calibration layer using Isotonic and Platt scaling."""

import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import Literal

import numpy as np
import structlog
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss

logger = structlog.get_logger()


@dataclass
class CalibratedProbability:
    """Result of probability calibration."""

    raw_prob: float
    calibrated_prob: float
    market_type: str
    calibration_method: str
    reliability_score: float | None = None


@dataclass
class CalibrationMetrics:
    """Metrics for calibration quality."""

    brier_score_before: float
    brier_score_after: float
    log_loss_before: float
    log_loss_after: float
    calibration_slope: float
    n_samples: int


class CalibrationLayer:
    """
    Probability calibration using Isotonic Regression or Platt Scaling.

    Transforms raw model probabilities to calibrated probabilities that
    better reflect true outcome frequencies.

    Maintains separate calibrators for each market type.
    """

    def __init__(self, method: Literal["isotonic", "platt", "sigmoid"] = "isotonic"):
        """
        Initialize calibration layer.

        Args:
            method: Calibration method ('isotonic', 'platt', or 'sigmoid')
                    Note: 'sigmoid' is an alias for 'platt' (Platt scaling)
        """
        # 'sigmoid' is an alias for 'platt'
        self.method = "platt" if method == "sigmoid" else method
        self.calibrators: dict[str, IsotonicRegression | LogisticRegression] = {}
        self.metrics: dict[str, CalibrationMetrics] = {}
        self.is_fitted: dict[str, bool] = {}

    def fit(
        self,
        y_pred_proba: np.ndarray,
        y_true: np.ndarray,
        market_type: str = "default",
    ) -> CalibrationMetrics:
        """
        Fit calibrator for a specific market type.

        Args:
            y_pred_proba: Raw predicted probabilities
            y_true: Actual binary outcomes (0 or 1)
            market_type: Market type identifier

        Returns:
            CalibrationMetrics for before/after comparison
        """
        # Calculate pre-calibration metrics
        brier_before = brier_score_loss(y_true, y_pred_proba)
        logloss_before = log_loss(y_true, np.clip(y_pred_proba, 1e-10, 1 - 1e-10))

        # Fit calibrator
        if self.method == "isotonic":
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(y_pred_proba, y_true)
        else:  # platt
            calibrator = LogisticRegression(C=1.0, solver="lbfgs")
            calibrator.fit(y_pred_proba.reshape(-1, 1), y_true)

        self.calibrators[market_type] = calibrator
        self.is_fitted[market_type] = True

        # Get calibrated predictions
        calibrated = self.transform(y_pred_proba, market_type)

        # Calculate post-calibration metrics
        brier_after = brier_score_loss(y_true, calibrated)
        logloss_after = log_loss(y_true, np.clip(calibrated, 1e-10, 1 - 1e-10))

        # Calculate calibration slope (reliability)
        prob_true, prob_pred = calibration_curve(y_true, calibrated, n_bins=10, strategy="uniform")
        if len(prob_pred) > 1:
            slope = np.polyfit(prob_pred, prob_true, 1)[0]
        else:
            slope = 1.0

        metrics = CalibrationMetrics(
            brier_score_before=float(brier_before),
            brier_score_after=float(brier_after),
            log_loss_before=float(logloss_before),
            log_loss_after=float(logloss_after),
            calibration_slope=float(slope),
            n_samples=len(y_true),
        )

        self.metrics[market_type] = metrics

        logger.info(
            "Calibrator fitted",
            market_type=market_type,
            method=self.method,
            brier_improvement=brier_before - brier_after,
            slope=slope,
        )

        return metrics

    def transform(
        self,
        y_pred_proba: np.ndarray | float,
        market_type: str = "default",
    ) -> np.ndarray | float:
        """
        Transform raw probabilities to calibrated probabilities.

        Args:
            y_pred_proba: Raw probabilities (single value or array)
            market_type: Market type to use calibrator for

        Returns:
            Calibrated probabilities
        """
        is_scalar = np.isscalar(y_pred_proba)
        proba = np.atleast_1d(y_pred_proba)

        # If no calibrator fitted, return original
        if market_type not in self.calibrators:
            # Try default calibrator
            if "default" in self.calibrators:
                market_type = "default"
            else:
                return float(proba[0]) if is_scalar else proba

        calibrator = self.calibrators[market_type]

        if self.method == "isotonic":
            calibrated = calibrator.predict(proba)
        else:  # platt
            calibrated = calibrator.predict_proba(proba.reshape(-1, 1))[:, 1]

        # Apply regression to mean - pull extreme probabilities back toward 0.50
        # This prevents overconfident predictions from calibration overfitting
        # A regression_strength of 0.20 means: 0.80 becomes 0.80*0.80 + 0.50*0.20 = 0.74
        REGRESSION_STRENGTH = 0.20
        calibrated = calibrated * (1 - REGRESSION_STRENGTH) + 0.50 * REGRESSION_STRENGTH

        # Ensure bounds
        calibrated = np.clip(calibrated, 0.001, 0.999)

        return float(calibrated[0]) if is_scalar else calibrated

    def calibrate(
        self,
        raw_prob: float,
        market_type: str = "default",
    ) -> CalibratedProbability:
        """
        Calibrate a single probability value.

        Args:
            raw_prob: Raw model probability
            market_type: Type of market

        Returns:
            CalibratedProbability with full context
        """
        calibrated = self.transform(raw_prob, market_type)

        # Get reliability score if available
        reliability = None
        if market_type in self.metrics:
            reliability = self.metrics[market_type].calibration_slope

        return CalibratedProbability(
            raw_prob=raw_prob,
            calibrated_prob=float(calibrated),
            market_type=market_type,
            calibration_method=self.method,
            reliability_score=reliability,
        )

    def save(self, path: str | Path) -> None:
        """Save calibrators to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "method": self.method,
            "calibrators": self.calibrators,
            "metrics": self.metrics,
            "is_fitted": self.is_fitted,
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

        logger.info("Calibration layer saved", path=str(path))

    def load(self, path: str | Path) -> None:
        """Load calibrators from file."""
        path = Path(path)

        if not path.exists():
            logger.warning("Calibration file not found", path=str(path))
            return

        with open(path, "rb") as f:
            data = pickle.load(f)

        self.method = data["method"]
        self.calibrators = data["calibrators"]
        self.metrics = data.get("metrics", {})
        self.is_fitted = data.get("is_fitted", {})

        logger.info(
            "Calibration layer loaded",
            path=str(path),
            market_types=list(self.calibrators.keys()),
        )

    def get_calibration_curve(
        self,
        y_pred_proba: np.ndarray,
        y_true: np.ndarray,
        n_bins: int = 10,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Get calibration curve data for visualization.

        Args:
            y_pred_proba: Predicted probabilities
            y_true: Actual outcomes
            n_bins: Number of bins

        Returns:
            Tuple of (mean_predicted_prob, actual_frequency) per bin
        """
        return calibration_curve(y_true, y_pred_proba, n_bins=n_bins, strategy="uniform")


def create_default_calibrators() -> CalibrationLayer:
    """
    Create calibration layer with reasonable default parameters.

    For production use, calibrators should be fitted on historical data.
    This provides a starting point that doesn't distort probabilities too much.
    """
    layer = CalibrationLayer(method="isotonic")

    # Create synthetic training data for each market type
    # These represent typical calibration patterns observed in sports betting
    np.random.seed(42)

    for market_type in ["spread", "moneyline", "total"]:
        # Generate synthetic well-calibrated data
        # In practice, this would come from historical predictions
        n_samples = 1000
        raw_probs = np.random.beta(2, 2, n_samples)  # Symmetric around 0.5

        # Outcomes follow the probabilities (well-calibrated baseline)
        outcomes = (np.random.random(n_samples) < raw_probs).astype(int)

        # Fit calibrator
        layer.fit(raw_probs, outcomes, market_type)

    return layer
