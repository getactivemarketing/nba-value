"""Value score calculation for MLB betting markets."""

import structlog
from dataclasses import dataclass
from typing import Literal

logger = structlog.get_logger()


@dataclass
class MLBValueResult:
    """Result of value calculation for an MLB bet."""

    market_type: str  # moneyline, runline, total
    bet_type: str  # home_ml, away_ml, home_rl, away_rl, over, under
    team: str | None  # Team abbreviation for ML/RL bets
    line: float | None  # Runline spread or total line

    # Probabilities
    model_prob: float  # Our model's probability
    market_prob: float  # Implied from odds

    # Edge calculation
    raw_edge: float  # model_prob - market_prob
    edge_pct: float  # (raw_edge / market_prob) * 100

    # Value scoring
    value_score: float  # 0-100 composite score
    confidence: str  # high, medium, low

    # Odds
    odds_decimal: float
    odds_american: int

    # Is this a recommended bet?
    is_value_bet: bool


class MLBValueCalculator:
    """
    Calculate value scores for MLB betting markets.

    Value score formula (similar to NBA):
    1. Calculate edge = model_prob - market_prob
    2. Calculate edge_pct = (edge / market_prob) * 100
    3. Apply confidence multiplier based on model certainty
    4. Apply market quality factor
    5. Scale to 0-100

    Thresholds:
    - 65+: Strong value bet (recommended)
    - 55-64: Moderate value
    - <55: Low/no value
    """

    # Minimum edge required to consider a bet
    MIN_EDGE = 0.02  # 2%

    # Value score thresholds
    STRONG_VALUE_THRESHOLD = 65
    MODERATE_VALUE_THRESHOLD = 55

    # Edge to value score scaling
    EDGE_SCALE_FACTOR = 400  # Multiply edge_pct to get base score

    @classmethod
    def calculate_value(
        cls,
        market_type: str,
        bet_type: str,
        model_prob: float,
        market_prob: float,
        odds_decimal: float,
        team: str | None = None,
        line: float | None = None,
        model_confidence: float = 0.5,
    ) -> MLBValueResult:
        """
        Calculate value score for a single bet.

        Args:
            market_type: "moneyline", "runline", or "total"
            bet_type: Specific bet type (e.g., "home_ml", "over")
            model_prob: Model's probability of the outcome
            market_prob: Market implied probability (1 / odds_decimal)
            odds_decimal: Decimal odds offered
            team: Team abbreviation (for ML/RL bets)
            line: Spread or total line
            model_confidence: Model's confidence in prediction (0-1)

        Returns:
            MLBValueResult with value score and recommendation
        """
        # Calculate raw edge
        raw_edge = model_prob - market_prob

        # Calculate edge percentage
        if market_prob > 0:
            edge_pct = (raw_edge / market_prob) * 100
        else:
            edge_pct = 0.0

        # Base value score from edge
        # 5% edge = 20 points, 10% edge = 40 points, 15% edge = 60 points
        base_score = edge_pct * 4.0

        # Confidence multiplier (0.8 - 1.2)
        confidence_multiplier = 0.8 + (model_confidence * 0.4)

        # Market type adjustment
        # Moneylines have slightly higher variance, so reduce score
        market_multiplier = 1.0
        if market_type == "moneyline":
            market_multiplier = 0.95
        elif market_type == "total":
            market_multiplier = 0.90  # Totals are harder to predict

        # Calculate final value score
        value_score = base_score * confidence_multiplier * market_multiplier

        # Add bonus for strong favorites with edge (less variance)
        if model_prob > 0.65 and raw_edge > 0.03:
            value_score += 5

        # Clamp to 0-100
        value_score = max(0, min(100, value_score))

        # Determine confidence level
        if raw_edge >= 0.08 and model_confidence >= 0.6:
            confidence = "high"
        elif raw_edge >= 0.04 and model_confidence >= 0.4:
            confidence = "medium"
        else:
            confidence = "low"

        # Is it a value bet?
        is_value = value_score >= cls.MODERATE_VALUE_THRESHOLD and raw_edge >= cls.MIN_EDGE

        # Convert to American odds
        odds_american = cls.decimal_to_american(odds_decimal)

        return MLBValueResult(
            market_type=market_type,
            bet_type=bet_type,
            team=team,
            line=line,
            model_prob=model_prob,
            market_prob=market_prob,
            raw_edge=raw_edge,
            edge_pct=edge_pct,
            value_score=round(value_score, 1),
            confidence=confidence,
            odds_decimal=odds_decimal,
            odds_american=odds_american,
            is_value_bet=is_value,
        )

    @classmethod
    def find_best_value(
        cls,
        values: list[MLBValueResult],
    ) -> MLBValueResult | None:
        """
        Find the best value bet from a list.

        Args:
            values: List of value results

        Returns:
            Best value result or None if no value bets
        """
        value_bets = [v for v in values if v.is_value_bet]
        if not value_bets:
            return None

        return max(value_bets, key=lambda v: v.value_score)

    @classmethod
    def get_recommendation(cls, value: MLBValueResult) -> str:
        """
        Get a text recommendation for a value bet.

        Args:
            value: Value result

        Returns:
            Recommendation string
        """
        if value.value_score >= 80:
            strength = "Strong"
        elif value.value_score >= 70:
            strength = "Good"
        elif value.value_score >= 65:
            strength = "Moderate"
        else:
            strength = "Lean"

        if value.market_type == "moneyline":
            return f"{strength} {value.team} ML ({value.odds_american:+d})"
        elif value.market_type == "runline":
            return f"{strength} {value.team} {value.line:+.1f} ({value.odds_american:+d})"
        elif value.market_type == "total":
            direction = "Over" if "over" in value.bet_type else "Under"
            return f"{strength} {direction} {value.line} ({value.odds_american:+d})"
        else:
            return f"{strength} value on {value.bet_type}"

    @staticmethod
    def decimal_to_american(decimal_odds: float) -> int:
        """Convert decimal odds to American format."""
        if decimal_odds >= 2.0:
            return int((decimal_odds - 1) * 100)
        else:
            return int(-100 / (decimal_odds - 1))

    @staticmethod
    def american_to_decimal(american_odds: int) -> float:
        """Convert American odds to decimal format."""
        if american_odds > 0:
            return 1 + (american_odds / 100)
        else:
            return 1 + (100 / abs(american_odds))

    @staticmethod
    def devig_odds(
        odds1: float,
        odds2: float,
        method: Literal["multiplicative", "additive"] = "multiplicative",
    ) -> tuple[float, float]:
        """
        Remove vig from two-way market odds.

        Args:
            odds1: First outcome decimal odds
            odds2: Second outcome decimal odds
            method: Devigging method

        Returns:
            Tuple of true probabilities
        """
        prob1 = 1 / odds1
        prob2 = 1 / odds2
        total = prob1 + prob2  # Includes vig

        if method == "multiplicative":
            # Scale probabilities proportionally
            true_prob1 = prob1 / total
            true_prob2 = prob2 / total
        else:
            # Additive method (subtract equal vig from each)
            vig = (total - 1) / 2
            true_prob1 = prob1 - vig
            true_prob2 = prob2 - vig

        return true_prob1, true_prob2


def calculate_kelly_fraction(
    model_prob: float,
    odds_decimal: float,
    kelly_fraction: float = 0.25,
) -> float:
    """
    Calculate Kelly Criterion bet size.

    Args:
        model_prob: Model's probability of winning
        odds_decimal: Decimal odds offered
        kelly_fraction: Fraction of full Kelly (default 1/4 Kelly)

    Returns:
        Recommended bet as fraction of bankroll
    """
    # Kelly formula: f* = (bp - q) / b
    # where b = decimal odds - 1, p = win prob, q = lose prob
    b = odds_decimal - 1
    p = model_prob
    q = 1 - p

    kelly = (b * p - q) / b

    # Apply fractional Kelly for risk management
    kelly = kelly * kelly_fraction

    # Never recommend negative or excessive bets
    return max(0, min(kelly, 0.05))  # Cap at 5% of bankroll
