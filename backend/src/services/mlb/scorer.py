"""MLB game scoring and prediction service."""

import structlog
import joblib
import numpy as np
from pathlib import Path
from datetime import datetime, date, timezone
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from src.models import MLBGame, MLBMarket, MLBPrediction, MLBPredictionSnapshot
from src.services.mlb.features import MLBFeatureCalculator, MLBGameFeatures
from src.services.mlb.value_calculator import MLBValueCalculator, MLBValueResult

logger = structlog.get_logger()


@dataclass
class MLBGamePrediction:
    """Complete prediction for an MLB game."""

    game_id: str
    home_team: str
    away_team: str
    game_date: date
    game_time: datetime | None

    # Model predictions
    predicted_run_diff: float  # Positive = home favored
    predicted_total: float
    p_home_win: float
    p_away_win: float

    # Runline probabilities (standard -1.5/+1.5)
    p_home_cover: float  # Home wins by 2+
    p_away_cover: float  # Away wins or loses by 1

    # Total probabilities
    p_over: float
    p_under: float

    # Best value bets by market
    best_ml: MLBValueResult | None = None
    best_rl: MLBValueResult | None = None
    best_total: MLBValueResult | None = None

    # Overall best bet
    best_bet: MLBValueResult | None = None

    # Pitcher info
    home_starter_name: str | None = None
    away_starter_name: str | None = None

    # Features used
    features: MLBGameFeatures | None = None


class MLBScorer:
    """
    Score MLB games and generate predictions.

    Uses:
    1. Run differential model (LightGBM) to predict home run margin
    2. Totals model to predict combined runs
    3. Convert predictions to probabilities
    4. Compare to market odds to find value
    """

    # Standard MLB runline
    RUNLINE = 1.5

    # Historical MLB averages for fallback
    AVG_RUNS_PER_TEAM = 4.5
    HOME_WIN_ADVANTAGE = 0.04  # ~54% home win rate historically

    # Default model paths
    DEFAULT_RUN_DIFF_MODEL = "models/mlb_run_diff_v1.joblib"
    DEFAULT_TOTALS_MODEL = "models/mlb_totals_v1.joblib"

    # Feature names expected by the trained model (must match training order)
    MODEL_FEATURE_NAMES = [
        "home_runs_per_game", "away_runs_per_game",
        "home_ops", "away_ops",
        "home_avg", "away_avg",
        "home_obp", "away_obp",
        "home_slg", "away_slg",
        "home_era", "away_era",
        "home_whip", "away_whip",
        "home_starter_era", "away_starter_era",
        "home_starter_whip", "away_starter_whip",
        "home_starter_k9", "away_starter_k9",
        "home_starter_bb9", "away_starter_bb9",
        "home_starter_ip", "away_starter_ip",
        "park_factor",
        "offense_diff", "starter_era_diff", "team_era_diff",
    ]

    def __init__(
        self,
        session: AsyncSession,
        run_diff_model_path: str | None = None,
        totals_model_path: str | None = None,
    ):
        self.session = session
        self.feature_calculator = MLBFeatureCalculator(session)
        self.value_calculator = MLBValueCalculator()

        # Load models - use defaults if not specified
        self.run_diff_model = None
        self.run_diff_model_data = None
        self.totals_model = None
        self.totals_model_data = None

        run_diff_path = Path(run_diff_model_path or self.DEFAULT_RUN_DIFF_MODEL)
        totals_path = Path(totals_model_path or self.DEFAULT_TOTALS_MODEL)

        if run_diff_path.exists():
            self.run_diff_model_data = joblib.load(run_diff_path)
            self.run_diff_model = self.run_diff_model_data.get("model")
            logger.info("Loaded run differential model", path=str(run_diff_path))

        if totals_path.exists():
            self.totals_model_data = joblib.load(totals_path)
            self.totals_model = self.totals_model_data.get("model")
            logger.info("Loaded totals model", path=str(totals_path))

    def _build_model_feature_vector(self, features: MLBGameFeatures) -> np.ndarray:
        """
        Build feature vector matching the trained model's expected features.

        The model was trained on specific features in a specific order.
        This method maps MLBGameFeatures to that expected format.
        """
        # Map features to training format with defaults
        vector = [
            features.home_runs_per_game or self.AVG_RUNS_PER_TEAM,
            features.away_runs_per_game or self.AVG_RUNS_PER_TEAM,
            features.home_ops or 0.720,  # League average OPS
            features.away_ops or 0.720,
            0.250,  # home_avg - default batting average (not in MLBGameFeatures)
            0.250,  # away_avg
            0.320,  # home_obp - default OBP
            0.320,  # away_obp
            0.400,  # home_slg - default SLG
            0.400,  # away_slg
            features.home_bullpen_era or 4.00,  # team ERA (using bullpen as proxy)
            features.away_bullpen_era or 4.00,
            1.30,  # home_whip - team WHIP default
            1.30,  # away_whip
            features.home_starter_era or 4.00,
            features.away_starter_era or 4.00,
            features.home_starter_whip or 1.25,
            features.away_starter_whip or 1.25,
            features.home_starter_k_rate or 8.5,  # K/9
            features.away_starter_k_rate or 8.5,
            features.home_starter_bb_rate or 3.0,  # BB/9
            features.away_starter_bb_rate or 3.0,
            100.0,  # home_starter_ip - innings pitched (default)
            100.0,  # away_starter_ip
            features.park_factor,
            features.offense_matchup_edge or 0.0,  # offense_diff
            features.starter_era_diff or 0.0,
            (features.away_bullpen_era or 4.0) - (features.home_bullpen_era or 4.0),  # team_era_diff
        ]
        return np.array([vector])

    async def score_game(self, game: MLBGame) -> MLBGamePrediction:
        """
        Generate prediction for a single game.

        Args:
            game: MLBGame object

        Returns:
            MLBGamePrediction with all predictions and value analysis
        """
        # Calculate features
        features = await self.feature_calculator.calculate_game_features(game)

        # Predict run differential
        if self.run_diff_model:
            feature_vector = self._build_model_feature_vector(features)
            predicted_run_diff = float(self.run_diff_model.predict(feature_vector)[0])
        else:
            # Fallback: use run differential difference + home advantage
            predicted_run_diff = self._estimate_run_diff(features)

        # Predict total
        if self.totals_model:
            feature_vector = self._build_model_feature_vector(features)
            predicted_total = float(self.totals_model.predict(feature_vector)[0])
        else:
            predicted_total = self._estimate_total(features)

        # Convert run diff to win probability
        p_home_win = self._run_diff_to_win_prob(predicted_run_diff)
        p_away_win = 1 - p_home_win

        # Calculate runline probabilities
        # Home cover = home wins by 2+
        p_home_cover = self._run_diff_to_cover_prob(predicted_run_diff, self.RUNLINE)
        p_away_cover = 1 - p_home_cover

        # Get pitcher names
        home_starter_name = None
        away_starter_name = None
        if game.home_starter:
            home_starter_name = game.home_starter.player_name
        if game.away_starter:
            away_starter_name = game.away_starter.player_name

        prediction = MLBGamePrediction(
            game_id=game.game_id,
            home_team=game.home_team,
            away_team=game.away_team,
            game_date=game.game_date,
            game_time=game.game_time,
            predicted_run_diff=round(predicted_run_diff, 2),
            predicted_total=round(predicted_total, 1),
            p_home_win=round(p_home_win, 3),
            p_away_win=round(p_away_win, 3),
            p_home_cover=round(p_home_cover, 3),
            p_away_cover=round(p_away_cover, 3),
            p_over=0.5,  # Will be updated with market context
            p_under=0.5,
            home_starter_name=home_starter_name,
            away_starter_name=away_starter_name,
            features=features,
        )

        # Get markets and calculate value
        await self._calculate_market_values(prediction, game)

        return prediction

    def _estimate_run_diff(self, features: MLBGameFeatures) -> float:
        """Estimate run differential without ML model."""
        run_diff = 0.0

        # Base: home advantage
        run_diff += 0.3  # ~0.3 runs home advantage

        # Team quality
        if features.run_diff_diff is not None:
            run_diff += features.run_diff_diff * 0.5

        # Pitcher matchup
        if features.starter_era_diff is not None:
            # Away ERA - Home ERA, positive = home has better starter
            # Each 1.0 ERA difference ~ 0.5 runs
            run_diff += features.starter_era_diff * 0.5

        if features.starter_quality_diff is not None:
            # Add small pitcher quality adjustment
            run_diff += features.starter_quality_diff * 0.02

        # Park factor adjustment
        # Higher park factor slightly favors home team (more runs overall)
        if features.park_factor > 1.0:
            run_diff += (features.park_factor - 1.0) * 2

        return run_diff

    def _estimate_total(self, features: MLBGameFeatures) -> float:
        """Estimate total runs without ML model."""
        # Start with league average
        total = self.AVG_RUNS_PER_TEAM * 2

        # Adjust for team offense
        home_rpg = features.home_runs_per_game or self.AVG_RUNS_PER_TEAM
        away_rpg = features.away_runs_per_game or self.AVG_RUNS_PER_TEAM
        team_avg = (home_rpg + away_rpg) / 2
        total = team_avg * 2

        # Adjust for pitchers
        if features.home_starter_era and features.away_starter_era:
            avg_era = (features.home_starter_era + features.away_starter_era) / 2
            # Higher ERA = more runs
            era_adjustment = (avg_era - 4.0) * 0.5
            total += era_adjustment

        # Park and weather factors
        total *= features.park_factor
        total *= features.weather_factor

        return max(5.0, min(15.0, total))  # Reasonable bounds

    def _run_diff_to_win_prob(self, run_diff: float) -> float:
        """
        Convert predicted run differential to win probability.

        Uses logistic function calibrated on historical data.
        Each run of expected margin ~ 15% win probability shift
        """
        # Logistic function: P(home win) = 1 / (1 + exp(-k * run_diff))
        # k = 0.5 gives reasonable spread
        import math
        k = 0.5
        p = 1 / (1 + math.exp(-k * run_diff))
        return max(0.05, min(0.95, p))  # Clamp to reasonable range

    def _run_diff_to_cover_prob(self, run_diff: float, spread: float) -> float:
        """
        Calculate probability of covering a spread.

        Args:
            run_diff: Predicted run differential (positive = home favored)
            spread: The spread to cover (e.g., -1.5 for home favorite)

        Returns:
            Probability of home team covering
        """
        # Adjust run diff by spread
        # Home covers -1.5 if they win by 2+
        adjusted_diff = run_diff - spread

        # Use same logistic conversion
        import math
        k = 0.5
        p = 1 / (1 + math.exp(-k * adjusted_diff))
        return max(0.05, min(0.95, p))

    def _total_to_over_prob(self, predicted_total: float, line: float) -> float:
        """Calculate probability of over given predicted total and line."""
        import math
        diff = predicted_total - line
        # More conservative for totals
        k = 0.4
        p = 1 / (1 + math.exp(-k * diff))
        return max(0.1, min(0.9, p))

    async def _calculate_market_values(
        self,
        prediction: MLBGamePrediction,
        game: MLBGame,
    ) -> None:
        """Calculate value scores for all markets."""
        # Get markets for this game
        stmt = select(MLBMarket).where(MLBMarket.game_id == game.game_id)
        result = await self.session.execute(stmt)
        markets = result.scalars().all()

        all_values = []

        for market in markets:
            if market.market_type == "moneyline":
                # Moneyline value
                if market.home_odds and market.away_odds:
                    home_prob, away_prob = MLBValueCalculator.devig_odds(
                        float(market.home_odds), float(market.away_odds)
                    )

                    # Home ML value
                    home_value = MLBValueCalculator.calculate_value(
                        market_type="moneyline",
                        bet_type="home_ml",
                        model_prob=prediction.p_home_win,
                        market_prob=home_prob,
                        odds_decimal=float(market.home_odds),
                        team=game.home_team,
                    )
                    all_values.append(home_value)

                    # Away ML value
                    away_value = MLBValueCalculator.calculate_value(
                        market_type="moneyline",
                        bet_type="away_ml",
                        model_prob=prediction.p_away_win,
                        market_prob=away_prob,
                        odds_decimal=float(market.away_odds),
                        team=game.away_team,
                    )
                    all_values.append(away_value)

                    # Find best ML
                    ml_values = [home_value, away_value]
                    prediction.best_ml = MLBValueCalculator.find_best_value(ml_values)

            elif market.market_type == "runline":
                if market.home_odds and market.away_odds:
                    home_prob, away_prob = MLBValueCalculator.devig_odds(
                        float(market.home_odds), float(market.away_odds)
                    )

                    # Home runline value
                    home_value = MLBValueCalculator.calculate_value(
                        market_type="runline",
                        bet_type="home_rl",
                        model_prob=prediction.p_home_cover,
                        market_prob=home_prob,
                        odds_decimal=float(market.home_odds),
                        team=game.home_team,
                        line=float(market.line) if market.line else -1.5,
                    )
                    all_values.append(home_value)

                    # Away runline value
                    away_value = MLBValueCalculator.calculate_value(
                        market_type="runline",
                        bet_type="away_rl",
                        model_prob=prediction.p_away_cover,
                        market_prob=away_prob,
                        odds_decimal=float(market.away_odds),
                        team=game.away_team,
                        line=abs(float(market.line)) if market.line else 1.5,
                    )
                    all_values.append(away_value)

                    rl_values = [home_value, away_value]
                    prediction.best_rl = MLBValueCalculator.find_best_value(rl_values)

            elif market.market_type == "total":
                if market.over_odds and market.under_odds and market.line:
                    line = float(market.line)
                    over_prob, under_prob = MLBValueCalculator.devig_odds(
                        float(market.over_odds), float(market.under_odds)
                    )

                    # Calculate over/under probabilities
                    p_over = self._total_to_over_prob(prediction.predicted_total, line)
                    p_under = 1 - p_over
                    prediction.p_over = round(p_over, 3)
                    prediction.p_under = round(p_under, 3)

                    # Over value
                    over_value = MLBValueCalculator.calculate_value(
                        market_type="total",
                        bet_type="over",
                        model_prob=p_over,
                        market_prob=over_prob,
                        odds_decimal=float(market.over_odds),
                        line=line,
                    )
                    all_values.append(over_value)

                    # Under value
                    under_value = MLBValueCalculator.calculate_value(
                        market_type="total",
                        bet_type="under",
                        model_prob=p_under,
                        market_prob=under_prob,
                        odds_decimal=float(market.under_odds),
                        line=line,
                    )
                    all_values.append(under_value)

                    total_values = [over_value, under_value]
                    prediction.best_total = MLBValueCalculator.find_best_value(total_values)

        # Find overall best bet
        prediction.best_bet = MLBValueCalculator.find_best_value(all_values)

    async def score_games(
        self,
        game_date: date | None = None,
    ) -> list[MLBGamePrediction]:
        """
        Score all games for a date.

        Args:
            game_date: Date to score (defaults to today)

        Returns:
            List of predictions
        """
        if game_date is None:
            game_date = date.today()

        # Get scheduled games
        stmt = select(MLBGame).where(
            and_(
                MLBGame.game_date == game_date,
                MLBGame.status == "scheduled",
            )
        )
        result = await self.session.execute(stmt)
        games = result.scalars().all()

        predictions = []
        for game in games:
            try:
                prediction = await self.score_game(game)
                predictions.append(prediction)
            except Exception as e:
                logger.warning(
                    "Failed to score game",
                    game_id=game.game_id,
                    error=str(e),
                )

        logger.info("Scored MLB games", count=len(predictions), date=game_date.isoformat())
        return predictions

    async def save_predictions(
        self,
        predictions: list[MLBGamePrediction],
    ) -> int:
        """
        Save predictions to database.

        Args:
            predictions: List of predictions

        Returns:
            Number of predictions saved
        """
        from sqlalchemy.dialects.postgresql import insert

        count = 0
        for pred in predictions:
            # Determine recommendation
            recommendation = None
            if pred.best_bet and pred.best_bet.is_value_bet:
                recommendation = MLBValueCalculator.get_recommendation(pred.best_bet)

            # Save main prediction for each market type
            for market_type in ["moneyline", "runline", "total"]:
                stmt = insert(MLBPrediction).values(
                    game_id=pred.game_id,
                    market_type=market_type,
                    predicted_run_diff=pred.predicted_run_diff,
                    predicted_total=pred.predicted_total,
                    p_home_win=pred.p_home_win,
                    p_away_win=pred.p_away_win,
                    p_home_cover=pred.p_home_cover,
                    p_away_cover=pred.p_away_cover,
                    p_over=pred.p_over,
                    p_under=pred.p_under,
                    recommendation=recommendation if market_type == "moneyline" else None,
                    model_version="v1",
                ).on_conflict_do_update(
                    index_elements=["game_id", "market_type"],
                    set_={
                        "predicted_run_diff": pred.predicted_run_diff,
                        "predicted_total": pred.predicted_total,
                        "p_home_win": pred.p_home_win,
                        "p_away_win": pred.p_away_win,
                        "created_at": datetime.now(timezone.utc),
                    },
                )
                await self.session.execute(stmt)
                count += 1

        await self.session.commit()
        logger.info("Saved predictions", count=count)
        return count


async def run_scoring(
    session: AsyncSession,
    game_date: date | None = None,
    run_diff_model_path: str | None = None,
    totals_model_path: str | None = None,
) -> list[MLBGamePrediction]:
    """
    Run full scoring pipeline for a date.

    Args:
        session: Database session
        game_date: Date to score
        run_diff_model_path: Optional path to run diff model
        totals_model_path: Optional path to totals model

    Returns:
        List of predictions
    """
    scorer = MLBScorer(
        session,
        run_diff_model_path=run_diff_model_path,
        totals_model_path=totals_model_path,
    )
    predictions = await scorer.score_games(game_date)
    await scorer.save_predictions(predictions)
    return predictions
