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
    home_injury_score: float = 0.0,
    away_injury_score: float = 0.0,
    is_home_bet: bool = True,
    algorithm: str = "a",
) -> ConfidenceComponents:
    """
    Compute confidence multiplier from component factors.

    Components:
    - Ensemble Agreement: Lower std = higher agreement = more confidence
    - Calibration Reliability: Historical accuracy for this market type
    - Injury Adjustment: Adjust based on injury differential
    - Segment Reliability: Historical accuracy for this edge band (Algo A only)

    Args:
        ensemble_std: Standard deviation of ensemble predictions
        market_type: Type of market
        raw_edge: Raw edge value for segment lookup
        home_injury_score: Home team injury severity (0-1, higher = more injured)
        away_injury_score: Away team injury severity (0-1, higher = more injured)
        is_home_bet: True if betting on home team/over, False for away/under
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

    # Injury Adjustment
    # For spread/ML: adjust based on which team we're betting on
    # If betting on injured team, reduce confidence
    # If betting against injured team, increase confidence
    #
    # For totals: injuries generally lead to lower scoring, so:
    # - Over bets: high injury = less confidence
    # - Under bets: high injury = more confidence
    injury_factor = _compute_injury_adjustment(
        home_injury_score,
        away_injury_score,
        is_home_bet,
        market_type,
    )

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


def _compute_injury_adjustment(
    home_injury_score: float,
    away_injury_score: float,
    is_home_bet: bool,
    market_type: str,
) -> float:
    """
    Compute injury-based confidence adjustment.

    Logic:
    - For spread/moneyline bets:
      - Betting ON an injured team → reduce confidence
      - Betting AGAINST an injured team → increase confidence
      - The adjustment is proportional to the injury differential

    - For totals:
      - More injuries → likely lower scoring
      - Over bets: penalize when either team is heavily injured
      - Under bets: boost when either team is heavily injured

    Args:
        home_injury_score: 0-1, higher = more injured
        away_injury_score: 0-1, higher = more injured
        is_home_bet: True for home team bets / over, False for away / under
        market_type: spread, moneyline, or total

    Returns:
        Adjustment factor (0.7-1.3 range)
    """
    if market_type == "total":
        # For totals: combine both teams' injuries
        total_injury = (home_injury_score + away_injury_score) / 2

        if is_home_bet:  # Over bet
            # More injuries = less scoring = less confidence in over
            # total_injury of 0 → 1.0, total_injury of 1.0 → 0.7
            return float(np.clip(1.0 - (total_injury * 0.3), 0.7, 1.0))
        else:  # Under bet
            # More injuries = less scoring = more confidence in under
            # total_injury of 0 → 1.0, total_injury of 1.0 → 1.2
            return float(np.clip(1.0 + (total_injury * 0.2), 1.0, 1.2))

    else:  # spread or moneyline
        # Injury differential: positive = opponent more injured = good for bet
        if is_home_bet:
            # Betting on home team
            injury_diff = away_injury_score - home_injury_score
        else:
            # Betting on away team
            injury_diff = home_injury_score - away_injury_score

        # Map injury_diff from [-1, 1] to [0.7, 1.3]
        # injury_diff = 0 → 1.0 (neutral)
        # injury_diff = 0.5 (opponent 50% more injured) → 1.15
        # injury_diff = -0.5 (we're betting on 50% more injured team) → 0.85
        adjustment = 1.0 + (injury_diff * 0.3)

        return float(np.clip(adjustment, 0.7, 1.3))
