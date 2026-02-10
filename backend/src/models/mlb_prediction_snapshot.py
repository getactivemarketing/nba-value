"""MLB Prediction snapshot model for tracking picks before games start."""

from datetime import datetime, date
from decimal import Decimal

from sqlalchemy import String, Integer, Numeric, DateTime, Boolean, Text, Date
from sqlalchemy.orm import Mapped, mapped_column

from src.database import Base


class MLBPredictionSnapshot(Base):
    """Stores MLB model predictions before each game for performance tracking."""

    __tablename__ = "mlb_prediction_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    game_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    snapshot_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    # Game info
    home_team: Mapped[str] = mapped_column(String(5), nullable=False)
    away_team: Mapped[str] = mapped_column(String(5), nullable=False)
    game_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    game_date: Mapped[date | None] = mapped_column(Date, nullable=True, index=True)

    # Starting pitchers
    home_starter_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    away_starter_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    home_starter_era: Mapped[Decimal | None] = mapped_column(Numeric(5, 2), nullable=True)
    away_starter_era: Mapped[Decimal | None] = mapped_column(Numeric(5, 2), nullable=True)

    # Winner prediction
    predicted_winner: Mapped[str] = mapped_column(String(5), nullable=False)
    winner_probability: Mapped[Decimal] = mapped_column(Numeric(5, 3), nullable=False)
    winner_confidence: Mapped[str] = mapped_column(String(20), nullable=False)  # high/medium/low

    # Run differential prediction
    predicted_run_diff: Mapped[Decimal | None] = mapped_column(Numeric(5, 2), nullable=True)
    predicted_total: Mapped[Decimal | None] = mapped_column(Numeric(5, 1), nullable=True)

    # Best moneyline bet
    best_ml_team: Mapped[str | None] = mapped_column(String(5), nullable=True)
    best_ml_odds: Mapped[Decimal | None] = mapped_column(Numeric(6, 3), nullable=True)
    best_ml_value_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    best_ml_edge: Mapped[Decimal | None] = mapped_column(Numeric(5, 2), nullable=True)

    # Best runline bet
    best_rl_team: Mapped[str | None] = mapped_column(String(5), nullable=True)
    best_rl_line: Mapped[Decimal | None] = mapped_column(Numeric(5, 1), nullable=True)
    best_rl_odds: Mapped[Decimal | None] = mapped_column(Numeric(6, 3), nullable=True)
    best_rl_value_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    best_rl_edge: Mapped[Decimal | None] = mapped_column(Numeric(5, 2), nullable=True)

    # Best total bet
    best_total_direction: Mapped[str | None] = mapped_column(String(10), nullable=True)  # over/under
    best_total_line: Mapped[Decimal | None] = mapped_column(Numeric(5, 1), nullable=True)
    best_total_odds: Mapped[Decimal | None] = mapped_column(Numeric(6, 3), nullable=True)
    best_total_value_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    best_total_edge: Mapped[Decimal | None] = mapped_column(Numeric(5, 2), nullable=True)

    # Overall best bet (highest value score across all markets)
    best_bet_type: Mapped[str | None] = mapped_column(String(20), nullable=True)  # moneyline/runline/total
    best_bet_team: Mapped[str | None] = mapped_column(String(10), nullable=True)
    best_bet_line: Mapped[Decimal | None] = mapped_column(Numeric(5, 1), nullable=True)
    best_bet_odds: Mapped[Decimal | None] = mapped_column(Numeric(6, 3), nullable=True)
    best_bet_value_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    best_bet_edge: Mapped[Decimal | None] = mapped_column(Numeric(5, 2), nullable=True)

    # Park and weather context
    venue_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    park_factor: Mapped[Decimal | None] = mapped_column(Numeric(4, 2), nullable=True)
    temperature: Mapped[int | None] = mapped_column(Integer, nullable=True)
    is_dome: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    # Line movement tracking
    opening_ml_home: Mapped[Decimal | None] = mapped_column(Numeric(6, 3), nullable=True)
    current_ml_home: Mapped[Decimal | None] = mapped_column(Numeric(6, 3), nullable=True)
    opening_total: Mapped[Decimal | None] = mapped_column(Numeric(5, 1), nullable=True)
    current_total: Mapped[Decimal | None] = mapped_column(Numeric(5, 1), nullable=True)

    # Actual results (filled in after game)
    actual_winner: Mapped[str | None] = mapped_column(String(5), nullable=True)
    home_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_score: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Grading (filled in after game)
    winner_correct: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    best_ml_result: Mapped[str | None] = mapped_column(String(20), nullable=True)  # win/loss
    best_ml_profit: Mapped[Decimal | None] = mapped_column(Numeric(8, 2), nullable=True)
    best_rl_result: Mapped[str | None] = mapped_column(String(20), nullable=True)  # win/loss/push
    best_rl_profit: Mapped[Decimal | None] = mapped_column(Numeric(8, 2), nullable=True)
    best_total_result: Mapped[str | None] = mapped_column(String(20), nullable=True)  # win/loss/push
    best_total_profit: Mapped[Decimal | None] = mapped_column(Numeric(8, 2), nullable=True)
    best_bet_result: Mapped[str | None] = mapped_column(String(20), nullable=True)
    best_bet_profit: Mapped[Decimal | None] = mapped_column(Numeric(8, 2), nullable=True)

    # Key factors (JSON array)
    factors: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        nullable=False
    )
