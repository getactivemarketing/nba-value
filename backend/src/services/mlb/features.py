"""Feature engineering for MLB run differential model."""

import structlog
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from sqlalchemy import select, and_, desc
from sqlalchemy.ext.asyncio import AsyncSession

from src.models import (
    MLBGame, MLBTeamStats, MLBPitcherStats, MLBPitcher, MLBGameContext,
)
from src.services.mlb.pitcher_quality import PitcherQualityScorer, PitcherMetrics

logger = structlog.get_logger()


@dataclass
class MLBGameFeatures:
    """Complete feature set for an MLB game prediction."""

    game_id: str
    game_date: date
    home_team: str
    away_team: str

    # Pitcher features
    home_starter_era: float | None = None
    away_starter_era: float | None = None
    home_starter_whip: float | None = None
    away_starter_whip: float | None = None
    home_starter_k_rate: float | None = None
    away_starter_k_rate: float | None = None
    home_starter_bb_rate: float | None = None
    away_starter_bb_rate: float | None = None
    home_starter_quality: float | None = None
    away_starter_quality: float | None = None
    starter_quality_diff: float | None = None
    starter_era_diff: float | None = None

    # Team offensive features
    home_runs_per_game: float | None = None
    away_runs_per_game: float | None = None
    home_runs_per_game_l10: float | None = None
    away_runs_per_game_l10: float | None = None
    home_ops: float | None = None
    away_ops: float | None = None

    # Team defensive features
    home_runs_allowed: float | None = None
    away_runs_allowed: float | None = None
    home_bullpen_era: float | None = None
    away_bullpen_era: float | None = None

    # Team run differential
    home_run_diff: float | None = None
    away_run_diff: float | None = None
    run_diff_diff: float | None = None  # Home run diff - Away run diff

    # Win percentages
    home_win_pct: float | None = None
    away_win_pct: float | None = None
    home_home_win_pct: float | None = None  # Win % at home
    away_away_win_pct: float | None = None  # Win % on road

    # Recent form
    home_last_10: str | None = None
    away_last_10: str | None = None
    home_last_10_win_pct: float | None = None
    away_last_10_win_pct: float | None = None

    # Rest and schedule
    home_rest_days: int | None = None
    away_rest_days: int | None = None
    rest_advantage: int | None = None

    # Context features
    park_factor: float = 1.0
    temperature: int | None = None
    wind_factor: float | None = None
    weather_factor: float = 1.0
    is_dome: bool = False

    # Derived features
    pitcher_matchup_edge: float | None = None  # Quality diff weighted
    offense_matchup_edge: float | None = None  # Offensive diff
    total_run_environment: float | None = None  # Park + weather combined

    # Metadata
    has_home_starter: bool = False
    has_away_starter: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for ML model input."""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith("_") and k not in ("game_id", "game_date", "home_team", "away_team")
        }

    def get_feature_vector(self) -> list[float]:
        """Get ordered feature vector for model prediction."""
        features = [
            self.home_starter_era or 4.0,
            self.away_starter_era or 4.0,
            self.home_starter_whip or 1.25,
            self.away_starter_whip or 1.25,
            self.home_starter_k_rate or 8.5,
            self.away_starter_k_rate or 8.5,
            self.home_starter_quality or 50.0,
            self.away_starter_quality or 50.0,
            self.starter_quality_diff or 0.0,
            self.starter_era_diff or 0.0,
            self.home_runs_per_game or 4.5,
            self.away_runs_per_game or 4.5,
            self.home_runs_per_game_l10 or 4.5,
            self.away_runs_per_game_l10 or 4.5,
            self.home_run_diff or 0.0,
            self.away_run_diff or 0.0,
            self.run_diff_diff or 0.0,
            self.home_win_pct or 0.5,
            self.away_win_pct or 0.5,
            self.home_home_win_pct or 0.5,
            self.away_away_win_pct or 0.5,
            self.home_last_10_win_pct or 0.5,
            self.away_last_10_win_pct or 0.5,
            float(self.home_rest_days or 1),
            float(self.away_rest_days or 1),
            float(self.rest_advantage or 0),
            self.park_factor,
            float(self.temperature or 70),
            self.weather_factor,
            1.0 if self.is_dome else 0.0,
            self.pitcher_matchup_edge or 0.0,
            self.offense_matchup_edge or 0.0,
            self.total_run_environment or 1.0,
        ]
        return features

    @classmethod
    def get_feature_names(cls) -> list[str]:
        """Get ordered feature names matching get_feature_vector()."""
        return [
            "home_starter_era",
            "away_starter_era",
            "home_starter_whip",
            "away_starter_whip",
            "home_starter_k_rate",
            "away_starter_k_rate",
            "home_starter_quality",
            "away_starter_quality",
            "starter_quality_diff",
            "starter_era_diff",
            "home_runs_per_game",
            "away_runs_per_game",
            "home_runs_per_game_l10",
            "away_runs_per_game_l10",
            "home_run_diff",
            "away_run_diff",
            "run_diff_diff",
            "home_win_pct",
            "away_win_pct",
            "home_home_win_pct",
            "away_away_win_pct",
            "home_last_10_win_pct",
            "away_last_10_win_pct",
            "home_rest_days",
            "away_rest_days",
            "rest_advantage",
            "park_factor",
            "temperature",
            "weather_factor",
            "is_dome",
            "pitcher_matchup_edge",
            "offense_matchup_edge",
            "total_run_environment",
        ]


class MLBFeatureCalculator:
    """Calculates features for MLB games."""

    def __init__(self, session: AsyncSession):
        self.session = session
        self.pitcher_scorer = PitcherQualityScorer()

    async def calculate_game_features(self, game: MLBGame) -> MLBGameFeatures:
        """
        Calculate all features for a game.

        Args:
            game: MLBGame object

        Returns:
            MLBGameFeatures with all calculated features
        """
        features = MLBGameFeatures(
            game_id=game.game_id,
            game_date=game.game_date,
            home_team=game.home_team,
            away_team=game.away_team,
        )

        # Get pitcher stats
        if game.home_starter_id:
            home_pitcher_stats = await self._get_pitcher_stats(
                game.home_starter_id, game.game_date
            )
            if home_pitcher_stats:
                features.home_starter_era = float(home_pitcher_stats.era) if home_pitcher_stats.era else None
                features.home_starter_whip = float(home_pitcher_stats.whip) if home_pitcher_stats.whip else None
                features.home_starter_k_rate = float(home_pitcher_stats.k_per_9) if home_pitcher_stats.k_per_9 else None
                features.home_starter_bb_rate = float(home_pitcher_stats.bb_per_9) if home_pitcher_stats.bb_per_9 else None
                features.home_starter_quality = float(home_pitcher_stats.quality_score) if home_pitcher_stats.quality_score else None
                features.has_home_starter = True

        if game.away_starter_id:
            away_pitcher_stats = await self._get_pitcher_stats(
                game.away_starter_id, game.game_date
            )
            if away_pitcher_stats:
                features.away_starter_era = float(away_pitcher_stats.era) if away_pitcher_stats.era else None
                features.away_starter_whip = float(away_pitcher_stats.whip) if away_pitcher_stats.whip else None
                features.away_starter_k_rate = float(away_pitcher_stats.k_per_9) if away_pitcher_stats.k_per_9 else None
                features.away_starter_bb_rate = float(away_pitcher_stats.bb_per_9) if away_pitcher_stats.bb_per_9 else None
                features.away_starter_quality = float(away_pitcher_stats.quality_score) if away_pitcher_stats.quality_score else None
                features.has_away_starter = True

        # Calculate pitcher matchup features
        if features.home_starter_quality and features.away_starter_quality:
            features.starter_quality_diff = features.home_starter_quality - features.away_starter_quality
            # Positive = home has better starter
            features.pitcher_matchup_edge = features.starter_quality_diff / 10.0  # Scaled

        if features.home_starter_era and features.away_starter_era:
            # Positive = away has higher ERA (home advantage)
            features.starter_era_diff = features.away_starter_era - features.home_starter_era

        # Get team stats
        home_stats = await self._get_team_stats(game.home_team, game.game_date)
        away_stats = await self._get_team_stats(game.away_team, game.game_date)

        if home_stats:
            features.home_runs_per_game = float(home_stats.runs_per_game) if home_stats.runs_per_game else None
            features.home_runs_per_game_l10 = float(home_stats.runs_per_game_l10) if home_stats.runs_per_game_l10 else None
            features.home_runs_allowed = float(home_stats.runs_allowed_per_game) if home_stats.runs_allowed_per_game else None
            features.home_run_diff = float(home_stats.run_diff_per_game) if home_stats.run_diff_per_game else None
            features.home_win_pct = float(home_stats.win_pct) if home_stats.win_pct else None
            features.home_home_win_pct = float(home_stats.home_win_pct) if home_stats.home_win_pct else None
            features.home_ops = float(home_stats.ops) if home_stats.ops else None
            features.home_bullpen_era = float(home_stats.bullpen_era) if home_stats.bullpen_era else None
            features.home_last_10 = home_stats.last_10_record
            features.home_rest_days = home_stats.days_rest

            # Calculate L10 win %
            if home_stats.last_10_wins is not None:
                total = (home_stats.last_10_wins or 0) + (home_stats.last_10_losses or 0)
                if total > 0:
                    features.home_last_10_win_pct = home_stats.last_10_wins / total

        if away_stats:
            features.away_runs_per_game = float(away_stats.runs_per_game) if away_stats.runs_per_game else None
            features.away_runs_per_game_l10 = float(away_stats.runs_per_game_l10) if away_stats.runs_per_game_l10 else None
            features.away_runs_allowed = float(away_stats.runs_allowed_per_game) if away_stats.runs_allowed_per_game else None
            features.away_run_diff = float(away_stats.run_diff_per_game) if away_stats.run_diff_per_game else None
            features.away_win_pct = float(away_stats.win_pct) if away_stats.win_pct else None
            features.away_away_win_pct = float(away_stats.away_win_pct) if away_stats.away_win_pct else None
            features.away_ops = float(away_stats.ops) if away_stats.ops else None
            features.away_bullpen_era = float(away_stats.bullpen_era) if away_stats.bullpen_era else None
            features.away_last_10 = away_stats.last_10_record
            features.away_rest_days = away_stats.days_rest

            if away_stats.last_10_wins is not None:
                total = (away_stats.last_10_wins or 0) + (away_stats.last_10_losses or 0)
                if total > 0:
                    features.away_last_10_win_pct = away_stats.last_10_wins / total

        # Calculate run diff difference
        if features.home_run_diff is not None and features.away_run_diff is not None:
            features.run_diff_diff = features.home_run_diff - features.away_run_diff

        # Calculate rest advantage
        if features.home_rest_days is not None and features.away_rest_days is not None:
            features.rest_advantage = features.home_rest_days - features.away_rest_days

        # Calculate offense matchup edge
        if features.home_runs_per_game and features.away_runs_per_game:
            features.offense_matchup_edge = features.home_runs_per_game - features.away_runs_per_game

        # Get game context (park, weather)
        context = await self._get_game_context(game.game_id)
        if context:
            features.park_factor = float(context.park_factor) if context.park_factor else 1.0
            features.temperature = context.temperature
            features.is_dome = context.is_dome
            features.weather_factor = float(context.weather_factor) if context.weather_factor else 1.0

            # Combined run environment
            features.total_run_environment = features.park_factor * features.weather_factor

        return features

    async def _get_pitcher_stats(
        self,
        pitcher_id: int,
        before_date: date,
    ) -> MLBPitcherStats | None:
        """Get most recent pitcher stats before the game date."""
        stmt = select(MLBPitcherStats).where(
            and_(
                MLBPitcherStats.pitcher_id == pitcher_id,
                MLBPitcherStats.stat_date < before_date,
            )
        ).order_by(desc(MLBPitcherStats.stat_date)).limit(1)

        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def _get_team_stats(
        self,
        team_abbr: str,
        before_date: date,
    ) -> MLBTeamStats | None:
        """Get most recent team stats before the game date."""
        stmt = select(MLBTeamStats).where(
            and_(
                MLBTeamStats.team_abbr == team_abbr,
                MLBTeamStats.stat_date < before_date,
            )
        ).order_by(desc(MLBTeamStats.stat_date)).limit(1)

        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def _get_game_context(self, game_id: str) -> MLBGameContext | None:
        """Get game context (venue, weather)."""
        stmt = select(MLBGameContext).where(MLBGameContext.game_id == game_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def calculate_features_batch(
        self,
        games: list[MLBGame],
    ) -> list[MLBGameFeatures]:
        """Calculate features for multiple games."""
        features = []
        for game in games:
            try:
                game_features = await self.calculate_game_features(game)
                features.append(game_features)
            except Exception as e:
                logger.warning(
                    "Failed to calculate features",
                    game_id=game.game_id,
                    error=str(e),
                )
        return features


async def build_training_data(
    session: AsyncSession,
    start_date: date,
    end_date: date,
) -> list[dict]:
    """
    Build training data from historical games.

    Args:
        session: Database session
        start_date: Start of training period
        end_date: End of training period

    Returns:
        List of dicts with features and target (run differential)
    """
    calculator = MLBFeatureCalculator(session)

    # Get completed games in date range
    stmt = select(MLBGame).where(
        and_(
            MLBGame.game_date >= start_date,
            MLBGame.game_date <= end_date,
            MLBGame.status == "final",
            MLBGame.home_score.isnot(None),
            MLBGame.away_score.isnot(None),
        )
    ).order_by(MLBGame.game_date)

    result = await session.execute(stmt)
    games = result.scalars().all()

    training_data = []
    for game in games:
        features = await calculator.calculate_game_features(game)

        # Target: home run differential
        run_diff = game.home_score - game.away_score
        total_runs = game.home_score + game.away_score

        row = features.to_dict()
        row["target_run_diff"] = run_diff
        row["target_total"] = total_runs
        row["home_won"] = 1 if run_diff > 0 else 0
        row["game_id"] = game.game_id
        row["game_date"] = game.game_date.isoformat()

        training_data.append(row)

    logger.info(
        "Built training data",
        games=len(training_data),
        start=start_date.isoformat(),
        end=end_date.isoformat(),
    )

    return training_data
