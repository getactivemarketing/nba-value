"""Confidence multiplier calculation."""

import numpy as np
from dataclasses import dataclass


@dataclass
class ConfidenceComponents:
    """Components of the confidence multiplier."""

    ensemble_agreement: float
    calibration_reliability: float
    injury_certainty: float
    segment_reliability: float | None  # Only used in Algo A
    final_multiplier: float


def compute_confidence_multiplier(
    ensemble_std: float,
    market_type: str,
    raw_edge: float,
    injury_certainty: float = 1.0,
    algorithm: str = "a",
) -> ConfidenceComponents:
    """
    Compute confidence multiplier from component factors.

    Components:
    - Ensemble Agreement: Lower std = higher agreement = more confidence
    - Calibration Reliability: Historical accuracy for this market type
    - Injury Certainty: Certainty of injury information
    - Segment Reliability: Historical accuracy for this edge band (Algo A only)

    Args:
        ensemble_std: Standard deviation of ensemble predictions
        market_type: Type of market
        raw_edge: Raw edge value for segment lookup
        injury_certainty: Injury information certainty (0-1)
        algorithm: Which algorithm ('a' or 'b')

    Returns:
        ConfidenceComponents with all factors and final multiplier
    """
    # Ensemble Agreement Score
    # Lower std = more agreement among models
    # Map std of [0, 0.1] to agreement of [1.5, 0.5]
    ensemble_agreement = float(np.clip(1.5 - (ensemble_std * 10), 0.5, 1.5))

    # Calibration Reliability
    # TODO: Load from calibration_metrics table based on market_type
    # For now, use reasonable defaults
    calibration_reliability = {
        "spread": 1.1,
        "moneyline": 1.0,
        "total": 0.95,
        "prop": 0.85,
    }.get(market_type, 1.0)

    # Injury Certainty
    # Scale from [0, 1] to [0.7, 1.2]
    injury_factor = 0.7 + (injury_certainty * 0.5)

    if algorithm == "a":
        # Algorithm A: 4 components, 0-2 range each
        # Segment reliability based on edge band
        edge_band = _get_edge_band(raw_edge)
        segment_reliability = _get_segment_reliability(market_type, edge_band)

        # Combine (multiplicative)
        final = (
            ensemble_agreement
            * calibration_reliability
            * injury_factor
            * segment_reliability
        )
    else:
        # Algorithm B: 3 components, 0.5-1.5 range
        segment_reliability = None
        final = ensemble_agreement * calibration_reliability * injury_factor
        # Clamp to [0.5, 1.5] for Algo B
        final = float(np.clip(final, 0.5, 1.5))

    return ConfidenceComponents(
        ensemble_agreement=ensemble_agreement,
        calibration_reliability=calibration_reliability,
        injury_certainty=injury_factor,
        segment_reliability=segment_reliability,
        final_multiplier=final,
    )


def _get_edge_band(raw_edge: float) -> str:
    """Classify raw edge into bands."""
    if raw_edge < 0.02:
        return "tiny"
    elif raw_edge < 0.05:
        return "small"
    elif raw_edge < 0.10:
        return "medium"
    else:
        return "large"


def _get_segment_reliability(market_type: str, edge_band: str) -> float:
    """
    Get historical reliability for a market type + edge band combination.

    TODO: Load from database based on historical evaluation.
    """
    # Placeholder reliability matrix
    # Higher edge bands historically less reliable (regression to mean)
    reliability_matrix = {
        "spread": {"tiny": 0.9, "small": 1.0, "medium": 1.1, "large": 0.95},
        "moneyline": {"tiny": 0.85, "small": 1.0, "medium": 1.05, "large": 0.9},
        "total": {"tiny": 0.8, "small": 0.95, "medium": 1.0, "large": 0.85},
        "prop": {"tiny": 0.7, "small": 0.85, "medium": 0.9, "large": 0.75},
    }

    market_reliability = reliability_matrix.get(market_type, {})
    return market_reliability.get(edge_band, 1.0)
