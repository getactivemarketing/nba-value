"""Game result model for historical tracking."""

from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import String, Integer, Numeric, Date, Boolean
from sqlalchemy.orm import Mapped, mapped_column

from src.database import Base


class GameResult(Base):
    """Historical game result with closing lines and outcomes."""

    __tablename__ = "game_results"

    game_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    game_date: Mapped[date] = mapped_column(Date, nullable=False)
    home_team_id: Mapped[str] = mapped_column(String(10), nullable=False)
    away_team_id: Mapped[str] = mapped_column(String(10), nullable=False)

    # Final Scores
    home_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_score: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Closing Lines
    closing_spread: Mapped[Decimal | None] = mapped_column(Numeric(4, 1), nullable=True)
    closing_total: Mapped[Decimal | None] = mapped_column(Numeric(5, 1), nullable=True)

    # Model Predictions
    predicted_winner: Mapped[str | None] = mapped_column(String(10), nullable=True)
    predicted_winner_prob: Mapped[Decimal | None] = mapped_column(Numeric(5, 2), nullable=True)
    predicted_spread_pick: Mapped[str | None] = mapped_column(String(10), nullable=True)
    predicted_total_pick: Mapped[str | None] = mapped_column(String(10), nullable=True)
    model_projected_total: Mapped[Decimal | None] = mapped_column(Numeric(5, 1), nullable=True)

    # Actual Results
    actual_winner: Mapped[str | None] = mapped_column(String(10), nullable=True)
    spread_result: Mapped[str | None] = mapped_column(String(20), nullable=True)
    total_result: Mapped[str | None] = mapped_column(String(20), nullable=True)

    # Correctness tracking
    winner_correct: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    spread_correct: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    total_correct: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
