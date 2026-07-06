"""Value score calculation for MLB betting markets."""

import math
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

    # Unclamped selection metric: edge_pct * confidence * market multipliers.
    # Used to rank candidates; value_score is display-only.
    sort_score: float = 0.0


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

    # Minimum edge required to consider a bet.
    # Tightened from 0.02 -> 0.10 on 2026-04-28 after backtest showed bets with
    # raw_edge < 0.10 had a 20% win rate over 15 graded bets.
    # 2026-07-06: stays 0.10 — the 0.10-0.15 bucket is the best performer
    # since May 23 (56.9% WR, +17.9u over 72 bets). Do not raise.
    MIN_EDGE = 0.10

    # Sanity cap on edge_pct. Real sports-betting edges rarely exceed ~20%;
    # anything above 80% means the model is wildly miscalibrated for that
    # game ("model blowup") — skip the bet.
    MAX_EDGE_PCT = 80.0

    # Maximum decimal odds for a runline pick. Backtest (Apr 3-27) showed RL bets
    # with odds in 2.5-3.0 had 40% win rate over 35 bets (-2.5u). Filter applied
    # in scorer._calculate_market_values.
    MAX_RUNLINE_ODDS = 2.5

    # Value score thresholds
    STRONG_VALUE_THRESHOLD = 65
    MODERATE_VALUE_THRESHOLD = 55

    # Gate scale: the pre-2026-07-06 score formula (edge_pct * 4.0 * multipliers,
    # clamped) is kept verbatim as the QUALIFICATION gate so the proven +160u
    # season pick set is preserved exactly. (Was a dead `= 400` constant; the
    # formula hardcoded 4.0.)
    EDGE_SCALE_FACTOR = 4.0

    # Display score: market-regress the edge 50% toward the market, then
    # compress with tanh so scores spread 30-97 instead of piling up at 100
    # (65% of picks scored exactly 100 before this change). Display only —
    # selection uses sort_score, qualification uses the legacy gate.
    MARKET_REGRESSION_WEIGHT = 0.50
    DISPLAY_TANH_SCALE = 20.0

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

        # Confidence multiplier (0.8 - 1.2)
        confidence_multiplier = 0.8 + (model_confidence * 0.4)

        # Market type adjustment. 0.95/0.90 retained; an 0.80 ML penalty was
        # backtested 2026-07-06 and REJECTED (-8u vs 0.95 on 586 picks).
        market_multiplier = 1.0
        if market_type == "moneyline":
            market_multiplier = 0.95
        elif market_type == "total":
            market_multiplier = 0.90

        # --- Qualification gate: legacy formula, kept verbatim -------------
        gate_score = edge_pct * cls.EDGE_SCALE_FACTOR * confidence_multiplier * market_multiplier
        if model_prob > 0.65 and raw_edge > 0.03:
            gate_score += 5
        gate_score = max(0, min(100, gate_score))

        # --- Selection metric: unclamped ------------------------------------
        # The clamp made large edges tie at 100 and max() silently preferred
        # whichever market was added to the candidate list first (ML).
        sort_score = edge_pct * confidence_multiplier * market_multiplier

        # --- Display score: regressed + tanh-compressed ----------------------
        blended_edge_pct = edge_pct * (1.0 - cls.MARKET_REGRESSION_WEIGHT)
        value_score = (
            100.0
            * math.tanh(blended_edge_pct / cls.DISPLAY_TANH_SCALE)
            * confidence_multiplier
            * market_multiplier
        )
        # Favorite bonus uses the regressed probability so it tracks the
        # displayed edge (mirrors b004564).
        adjusted_model_prob = market_prob + raw_edge * (1.0 - cls.MARKET_REGRESSION_WEIGHT)
        if adjusted_model_prob > 0.65 and raw_edge > 0.03:
            value_score += 5
        value_score = max(0, min(100, value_score))

        # Determine confidence level
        if raw_edge >= 0.08 and model_confidence >= 0.6:
            confidence = "high"
        elif raw_edge >= 0.04 and model_confidence >= 0.4:
            confidence = "medium"
        else:
            confidence = "low"

        # Qualification: legacy gate + MIN_EDGE + blowup cap.
        # Use small epsilon for raw_edge comparison to handle floating point
        # precision (e.g., 0.60 - 0.50 gives 0.09999... due to binary representation)
        is_value = (
            gate_score >= cls.MODERATE_VALUE_THRESHOLD
            and raw_edge >= cls.MIN_EDGE - 1e-10
            and edge_pct <= cls.MAX_EDGE_PCT
        )

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
            sort_score=round(sort_score, 2),
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

        return max(value_bets, key=lambda v: v.sort_score)

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
