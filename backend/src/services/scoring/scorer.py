"""Main scoring service that integrates MOV model, calibration, and Value Score algorithms."""

from dataclasses import dataclass
from decimal import Decimal
from datetime import datetime, timezone

import structlog

from src.services.ml.mov_model import MOVModel, MOVPrediction
from src.services.ml.calibration import CalibrationLayer
from src.services.ml.probability import (
    mov_to_spread_prob,
    mov_to_moneyline_prob,
    devig_two_way_odds,
)
from src.services.scoring.algorithm_a import compute_value_score_algo_a, AlgoAResult
from src.services.scoring.algorithm_b import compute_value_score_algo_b, AlgoBResult
from src.services.scoring.confidence import compute_confidence_multiplier, ConfidenceComponents
from src.services.scoring.market_quality import compute_market_quality, MarketQualityResult

logger = structlog.get_logger()


@dataclass
class ScoringInput:
    """Input data for scoring a market."""

    # Game identification
    game_id: str
    market_type: str  # spread, moneyline, total

    # Market details
    outcome_label: str  # e.g., "home_spread", "away_ml"
    line: float | None  # Spread line or total line
    odds_decimal: float  # Current odds for this outcome
    opposite_odds: float  # Odds for opposite outcome (for de-vigging)

    # Team features (for MOV prediction)
    home_features: dict[str, float]
    away_features: dict[str, float]

    # Context
    tip_time: datetime
    book: str | None = None
    injury_certainty: float = 1.0


@dataclass
class ScoringResult:
    """Complete scoring result with both algorithms."""

    # Input context
    game_id: str
    market_type: str
    outcome_label: str

    # MOV prediction
    predicted_mov: float
    mov_confidence: float

    # Probabilities
    p_raw: float  # Raw model probability
    p_calibrated: float  # Calibrated probability (p_true)
    p_market: float  # De-vigged market probability

    # Raw edge
    raw_edge: float

    # Algorithm A results
    algo_a: AlgoAResult
    algo_a_confidence: ConfidenceComponents

    # Algorithm B results
    algo_b: AlgoBResult
    algo_b_confidence: ConfidenceComponents

    # Market quality
    market_quality: MarketQualityResult

    # Metadata
    calc_time: datetime
    time_to_tip_minutes: int


class ScoringService:
    """
    Main scoring service for computing Value Scores.

    Orchestrates:
    1. MOV prediction from team features
    2. Probability derivation from MOV
    3. Calibration of raw probabilities
    4. De-vigging of market odds
    5. Value Score calculation using both algorithms
    """

    def __init__(
        self,
        mov_model: MOVModel | None = None,
        calibration_layer: CalibrationLayer | None = None,
    ):
        """
        Initialize scoring service.

        Args:
            mov_model: Pre-trained MOV model (uses baseline if None)
            calibration_layer: Calibration layer (uses identity if None)
        """
        self.mov_model = mov_model or MOVModel()
        self.calibration = calibration_layer or CalibrationLayer()

    def score_market(self, input_data: ScoringInput) -> ScoringResult:
        """
        Score a single market opportunity.

        Args:
            input_data: Complete input data for scoring

        Returns:
            ScoringResult with both algorithm scores
        """
        calc_time = datetime.now(timezone.utc)

        # Calculate time to tip
        time_to_tip = int((input_data.tip_time - calc_time).total_seconds() / 60)

        # Combine features for MOV prediction
        features = self._combine_features(
            input_data.home_features,
            input_data.away_features,
        )

        # Step 1: Get MOV prediction
        mov_pred = self.mov_model.predict(features)

        # Step 2: Convert MOV to probability based on market type
        p_raw = self._mov_to_probability(
            mov_pred,
            input_data.market_type,
            input_data.line,
            input_data.outcome_label,
        )

        # Step 3: Calibrate probability
        p_calibrated = self.calibration.transform(p_raw, input_data.market_type)
        if isinstance(p_calibrated, float):
            p_true = p_calibrated
        else:
            p_true = float(p_calibrated)

        # Step 4: De-vig market odds to get market probability
        # devig_two_way_odds returns (prob1, prob2) where prob1 is for odds_decimal
        # odds_decimal is already for the outcome we're betting on (home/away/over/under)
        # So p_market directly represents the market's probability for this outcome
        # NO FLIP NEEDED - the devig already gives us the correct probability
        p_market, _ = devig_two_way_odds(
            input_data.odds_decimal,
            input_data.opposite_odds,
        )

        # Step 5: Calculate raw edge
        raw_edge = p_true - p_market

        # Step 6: Calculate confidence multipliers
        ensemble_std = 0.05  # Placeholder - would come from ensemble model

        confidence_a = compute_confidence_multiplier(
            ensemble_std=ensemble_std,
            market_type=input_data.market_type,
            raw_edge=raw_edge,
            injury_certainty=input_data.injury_certainty,
            algorithm="a",
        )

        confidence_b = compute_confidence_multiplier(
            ensemble_std=ensemble_std,
            market_type=input_data.market_type,
            raw_edge=raw_edge,
            injury_certainty=input_data.injury_certainty,
            algorithm="b",
        )

        # Step 7: Calculate market quality
        market_quality = compute_market_quality(
            odds_decimal=input_data.odds_decimal,
            market_type=input_data.market_type,
            time_to_tip_minutes=time_to_tip,
            book=input_data.book,
        )

        # Step 8: Compute Value Scores
        algo_a_result = compute_value_score_algo_a(
            p_true=p_true,
            p_market=p_market,
            market_type=input_data.market_type,
            confidence=confidence_a.final_multiplier,
            market_quality=market_quality.final_score,
        )

        algo_b_result = compute_value_score_algo_b(
            p_true=p_true,
            p_market=p_market,
            market_type=input_data.market_type,
            confidence=confidence_b.final_multiplier,
            market_quality=market_quality.final_score,
        )

        return ScoringResult(
            game_id=input_data.game_id,
            market_type=input_data.market_type,
            outcome_label=input_data.outcome_label,
            predicted_mov=mov_pred.predicted_mov,
            mov_confidence=mov_pred.confidence,
            p_raw=p_raw,
            p_calibrated=p_true,
            p_market=p_market,
            raw_edge=raw_edge,
            algo_a=algo_a_result,
            algo_a_confidence=confidence_a,
            algo_b=algo_b_result,
            algo_b_confidence=confidence_b,
            market_quality=market_quality,
            calc_time=calc_time,
            time_to_tip_minutes=time_to_tip,
        )

    def _combine_features(
        self,
        home_features: dict[str, float],
        away_features: dict[str, float],
    ) -> dict[str, float]:
        """Combine home and away features into single feature dict."""
        features = {}

        # Add home features with prefix
        for key, value in home_features.items():
            if not key.startswith("home_"):
                features[f"home_{key}"] = value
            else:
                features[key] = value

        # Add away features with prefix
        for key, value in away_features.items():
            if not key.startswith("away_"):
                features[f"away_{key}"] = value
            else:
                features[key] = value

        return features

    def _mov_to_probability(
        self,
        mov_pred: MOVPrediction,
        market_type: str,
        line: float | None,
        outcome_label: str,
    ) -> float:
        """Convert MOV prediction to probability for specific market."""
        is_home_perspective = "home" in outcome_label.lower() or "over" in outcome_label.lower()

        if market_type == "spread":
            if line is None:
                line = 0.0
            # For spread, line is from home perspective
            prob = mov_to_spread_prob(
                mov_pred.predicted_mov,
                line,
                mov_pred.mov_std,
            )
            return prob if is_home_perspective else (1 - prob)

        elif market_type == "moneyline":
            prob = mov_to_moneyline_prob(
                mov_pred.predicted_mov,
                mov_pred.mov_std,
            )
            return prob if is_home_perspective else (1 - prob)

        elif market_type == "total":
            # For totals, we'd need a separate total model
            # For now, return 0.5 (neutral)
            return 0.5

        else:
            # Unknown market type
            return 0.5

    def score_all_markets_for_game(
        self,
        game_id: str,
        home_features: dict[str, float],
        away_features: dict[str, float],
        markets: list[dict],
        tip_time: datetime,
    ) -> list[ScoringResult]:
        """
        Score all markets for a single game.

        Args:
            game_id: Game identifier
            home_features: Home team stats
            away_features: Away team stats
            markets: List of market dicts with keys: market_type, outcome_label, line, odds_decimal, opposite_odds
            tip_time: Game start time

        Returns:
            List of ScoringResult for each market
        """
        results = []

        for market in markets:
            try:
                input_data = ScoringInput(
                    game_id=game_id,
                    market_type=market["market_type"],
                    outcome_label=market["outcome_label"],
                    line=market.get("line"),
                    odds_decimal=market["odds_decimal"],
                    opposite_odds=market.get("opposite_odds", market["odds_decimal"]),
                    home_features=home_features,
                    away_features=away_features,
                    tip_time=tip_time,
                    book=market.get("book"),
                )

                result = self.score_market(input_data)
                results.append(result)

            except Exception as e:
                logger.error(
                    "Failed to score market",
                    game_id=game_id,
                    market_type=market.get("market_type"),
                    error=str(e),
                )

        return results


# Singleton instance for service
_scoring_service: ScoringService | None = None


def get_scoring_service() -> ScoringService:
    """Get or create the scoring service singleton."""
    global _scoring_service
    if _scoring_service is None:
        _scoring_service = ScoringService()
    return _scoring_service
