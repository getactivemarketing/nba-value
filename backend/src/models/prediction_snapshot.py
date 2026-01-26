"""Prediction snapshot model for tracking picks before games start."""

from datetime import datetime, date
from decimal import Decimal

from sqlalchemy import String, Integer, Numeric, DateTime, Boolean, Text, Date
from sqlalchemy.orm import Mapped, mapped_column

from src.database import Base


class PredictionSnapshot(Base):
    """Stores our model's predictions before each game for performance tracking."""

    __tablename__ = "prediction_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    game_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    snapshot_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    # Game info
    home_team: Mapped[str] = mapped_column(String(10), nullable=False)
    away_team: Mapped[str] = mapped_column(String(10), nullable=False)
    tip_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    game_date: Mapped[date | None] = mapped_column(Date, nullable=True, index=True)

    # Winner prediction
    predicted_winner: Mapped[str] = mapped_column(String(10), nullable=False)
    winner_probability: Mapped[Decimal] = mapped_column(Numeric(5, 2), nullable=False)
    winner_confidence: Mapped[str] = mapped_column(String(20), nullable=False)  # high/medium/low

    # Best value bet (our recommended play)
    best_bet_type: Mapped[str | None] = mapped_column(String(20), nullable=True)  # spread/total/moneyline
    best_bet_team: Mapped[str | None] = mapped_column(String(10), nullable=True)
    best_bet_line: Mapped[Decimal | None] = mapped_column(Numeric(5, 1), nullable=True)
    best_bet_value_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    best_bet_edge: Mapped[Decimal | None] = mapped_column(Numeric(5, 2), nullable=True)
    best_bet_odds: Mapped[Decimal | None] = mapped_column(Numeric(6, 3), nullable=True)

    # Closing lines (filled in after game)
    closing_spread: Mapped[Decimal | None] = mapped_column(Numeric(5, 1), nullable=True)
    closing_total: Mapped[Decimal | None] = mapped_column(Numeric(5, 1), nullable=True)

    # Actual results (filled in after game)
    actual_winner: Mapped[str | None] = mapped_column(String(10), nullable=True)
    home_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_score: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Grading (filled in after game)
    winner_correct: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    best_bet_result: Mapped[str | None] = mapped_column(String(20), nullable=True)  # win/loss/push
    best_bet_profit: Mapped[Decimal | None] = mapped_column(Numeric(8, 2), nullable=True)  # profit on $100 bet

    # Key factors that drove the prediction
    factors: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON array of factor strings

    # Best total bet (separate from spread bet)
    best_total_direction: Mapped[str | None] = mapped_column(String(10), nullable=True)  # over/under
    best_total_line: Mapped[Decimal | None] = mapped_column(Numeric(5, 1), nullable=True)
    best_total_value_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    best_total_edge: Mapped[Decimal | None] = mapped_column(Numeric(5, 2), nullable=True)
    best_total_odds: Mapped[Decimal | None] = mapped_column(Numeric(6, 3), nullable=True)
    best_total_result: Mapped[str | None] = mapped_column(String(20), nullable=True)  # win/loss/push
    best_total_profit: Mapped[Decimal | None] = mapped_column(Numeric(8, 2), nullable=True)

    # Line movement tracking (captured at snapshot)
    opening_spread: Mapped[Decimal | None] = mapped_column(Numeric(5, 1), nullable=True)
    current_spread: Mapped[Decimal | None] = mapped_column(Numeric(5, 1), nullable=True)
    spread_movement: Mapped[Decimal | None] = mapped_column(Numeric(5, 1), nullable=True)
    opening_total: Mapped[Decimal | None] = mapped_column(Numeric(5, 1), nullable=True)
    current_total: Mapped[Decimal | None] = mapped_column(Numeric(5, 1), nullable=True)
    total_movement: Mapped[Decimal | None] = mapped_column(Numeric(5, 1), nullable=True)
    line_movement_direction: Mapped[str | None] = mapped_column(String(20), nullable=True)  # toward_home/toward_away

    # Algorithm comparison (A=edge-based, B=combined-edge)
    algo_a_value_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    algo_a_edge_score: Mapped[Decimal | None] = mapped_column(Numeric(5, 2), nullable=True)
    algo_a_confidence: Mapped[str | None] = mapped_column(String(20), nullable=True)
    algo_b_value_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    algo_b_combined_edge: Mapped[Decimal | None] = mapped_column(Numeric(5, 2), nullable=True)
    algo_b_confidence: Mapped[str | None] = mapped_column(String(20), nullable=True)
    active_algorithm: Mapped[str | None] = mapped_column(String(10), nullable=True)  # a or b

    # Algorithm grading results
    algo_a_bet_result: Mapped[str | None] = mapped_column(String(20), nullable=True)
    algo_b_bet_result: Mapped[str | None] = mapped_column(String(20), nullable=True)
    algo_a_profit: Mapped[Decimal | None] = mapped_column(Numeric(8, 2), nullable=True)
    algo_b_profit: Mapped[Decimal | None] = mapped_column(Numeric(8, 2), nullable=True)

    # Injury context (for backtesting injury model)
    home_injury_score: Mapped[Decimal | None] = mapped_column(Numeric(5, 2), nullable=True)
    away_injury_score: Mapped[Decimal | None] = mapped_column(Numeric(5, 2), nullable=True)
    home_ppg_lost: Mapped[Decimal | None] = mapped_column(Numeric(5, 1), nullable=True)
    away_ppg_lost: Mapped[Decimal | None] = mapped_column(Numeric(5, 1), nullable=True)
    injury_edge: Mapped[Decimal | None] = mapped_column(Numeric(5, 2), nullable=True)  # away - home

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        nullable=False
    )
