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
    mov_to_total_prob,
    estimate_game_total,
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
        mov_model=None,  # MOVModel or RidgeMOVModel
        calibration_layer: CalibrationLayer | None = None,
    ):
        """
        Initialize scoring service.

        Args:
            mov_model: Pre-trained MOV model (uses baseline if None)
            calibration_layer: Calibration layer (uses identity if None)
        """
        self.mov_model = mov_model if mov_model is not None else MOVModel()
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
            home_features=input_data.home_features,
            away_features=input_data.away_features,
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
        home_features: dict[str, float] | None = None,
        away_features: dict[str, float] | None = None,
    ) -> float:
        """Convert MOV prediction to probability for specific market."""
        is_home_perspective = "home" in outcome_label.lower() or "over" in outcome_label.lower()

        if market_type == "spread":
            if line is None:
                line = 0.0

            # For spread betting, mov_to_spread_prob assumes line is from home perspective
            # The database stores lines from each team's perspective:
            # - home_spread: line is from home perspective (use directly)
            # - away_spread: line is from away perspective (need to negate)
            #
            # Example: ORL @ WAS, ORL is -7.5 favorite
            # - home_spread (WAS) has line = +7.5
            # - away_spread (ORL) has line = -7.5
            #
            # To calculate P(away covers), we need:
            # 1. Convert away line to home perspective: negate it (-7.5 → +7.5)
            # 2. Calculate P(home covers +7.5) using mov_to_spread_prob
            # 3. Return 1 - P(home covers) = P(away covers)

            if is_home_perspective:
                # Home spread: line already in home perspective
                prob = mov_to_spread_prob(
                    mov_pred.predicted_mov,
                    line,
                    mov_pred.mov_std,
                )
                return prob
            else:
                # Away spread: line is from away perspective, negate for home perspective
                home_line = -line  # Convert away line to home equivalent
                home_prob = mov_to_spread_prob(
                    mov_pred.predicted_mov,
                    home_line,
                    mov_pred.mov_std,
                )
                return 1 - home_prob  # P(away covers) = 1 - P(home covers)

        elif market_type == "moneyline":
            prob = mov_to_moneyline_prob(
                mov_pred.predicted_mov,
                mov_pred.mov_std,
            )
            return prob if is_home_perspective else (1 - prob)

        elif market_type == "total":
            # Use pace and efficiency model to estimate total points
            if home_features is None or away_features is None:
                return 0.5  # Neutral if no features

            # Extract pace and efficiency metrics (use L10 or season)
            home_pace = home_features.get("home_pace_10") or home_features.get("home_pace_season", 100.0)
            away_pace = away_features.get("away_pace_10") or away_features.get("away_pace_season", 100.0)
            home_ortg = home_features.get("home_ortg_10") or home_features.get("home_ortg_season", 110.0)
            home_drtg = home_features.get("home_drtg_10") or home_features.get("home_drtg_season", 110.0)
            away_ortg = away_features.get("away_ortg_10") or away_features.get("away_ortg_season", 110.0)
            away_drtg = away_features.get("away_drtg_10") or away_features.get("away_drtg_season", 110.0)

            # Fallback to PPG-based estimate if pace/efficiency not available
            if home_pace == 100.0 and away_pace == 100.0:
                # Use raw PPG as simpler estimate
                home_ppg = home_features.get("home_ppg_10") or home_features.get("home_ppg_season", 110.0)
                away_ppg = away_features.get("away_ppg_10") or away_features.get("away_ppg_season", 110.0)
                home_opp_ppg = home_features.get("home_opp_ppg_10") or home_features.get("home_opp_ppg_season", 110.0)
                away_opp_ppg = away_features.get("away_opp_ppg_10") or away_features.get("away_opp_ppg_season", 110.0)

                # Estimate: home scores = avg(home_ppg, away_opp_ppg), similar for away
                home_pts_est = (home_ppg + away_opp_ppg) / 2
                away_pts_est = (away_ppg + home_opp_ppg) / 2
            else:
                # Use pace/efficiency model
                home_pts_est, away_pts_est = estimate_game_total(
                    home_pace=home_pace,
                    away_pace=away_pace,
                    home_ortg=home_ortg,
                    home_drtg=home_drtg,
                    away_ortg=away_ortg,
                    away_drtg=away_drtg,
                )

            if line is None:
                return 0.5

            # Calculate over probability
            # Use 12.0 as std for total (typical game-to-game variance)
            prob = mov_to_total_prob(
                home_total_estimate=home_pts_est,
                away_total_estimate=away_pts_est,
                total_line=line,
                total_std=12.0,
            )

            # prob is P(over), so for "under" we need 1 - prob
            return prob if is_home_perspective else (1 - prob)

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

# Model paths
from pathlib import Path
MODEL_DIR = Path(__file__).parent.parent.parent.parent / "models"
MOV_MODEL_PATH = MODEL_DIR / "mov_model.pkl"
CALIBRATION_PATH = MODEL_DIR / "calibration.pkl"


def load_trained_models() -> tuple:
    """Load trained MOV model and calibration layer if available."""
    mov_model = None
    calibration = None

    # Try to load MOV model
    if MOV_MODEL_PATH.exists():
        import pickle
        try:
            with open(MOV_MODEL_PATH, 'rb') as f:
                model_data = pickle.load(f)

            if model_data.get('model_type') == 'ridge':
                # Ridge model - create wrapper
                from src.services.ml.train_model import RidgeMOVModel
                mov_model = RidgeMOVModel(
                    model_data['model'],
                    model_data['training_features'],
                    model_data['mov_std'],
                )
                logger.info("Loaded Ridge MOV model", mov_std=model_data['mov_std'])
            else:
                # LightGBM model
                mov_model = MOVModel()
                mov_model.load(MOV_MODEL_PATH)
                logger.info("Loaded LightGBM MOV model")
        except Exception as e:
            logger.warning(f"Failed to load MOV model: {e}")

    # Try to load calibration
    if CALIBRATION_PATH.exists():
        try:
            calibration = CalibrationLayer()
            calibration.load(CALIBRATION_PATH)
            logger.info("Loaded calibration layer", market_types=list(calibration.calibrators.keys()))
        except Exception as e:
            logger.warning(f"Failed to load calibration: {e}")

    return mov_model, calibration


def get_scoring_service() -> ScoringService:
    """Get or create the scoring service singleton."""
    global _scoring_service
    if _scoring_service is None:
        mov_model, calibration = load_trained_models()
        _scoring_service = ScoringService(
            mov_model=mov_model,
            calibration_layer=calibration,
        )
    return _scoring_service
