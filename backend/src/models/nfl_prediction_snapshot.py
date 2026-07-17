"""Frozen NFL pre-game predictions + graded results (flat $100 units)."""
from datetime import datetime, date
from sqlalchemy import String, Integer, Float, DateTime, Date
from sqlalchemy.orm import Mapped, mapped_column
from src.database import Base


class NFLPredictionSnapshot(Base):
    __tablename__ = "nfl_prediction_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    game_id: Mapped[str] = mapped_column(String(20), nullable=False, unique=True, index=True)
    snapshot_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    home_team: Mapped[str] = mapped_column(String(5), nullable=False)
    away_team: Mapped[str] = mapped_column(String(5), nullable=False)
    kickoff_utc: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    game_date: Mapped[date | None] = mapped_column(Date, nullable=True, index=True)

    predicted_margin: Mapped[float | None] = mapped_column(Float, nullable=True)
    predicted_total: Mapped[float | None] = mapped_column(Float, nullable=True)

    # best per market (spread + ML are shadow; total is live)
    best_spread_team: Mapped[str | None] = mapped_column(String(5), nullable=True)
    best_spread_line: Mapped[float | None] = mapped_column(Float, nullable=True)
    best_spread_odds: Mapped[float | None] = mapped_column(Float, nullable=True)
    best_spread_value_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    best_spread_edge: Mapped[float | None] = mapped_column(Float, nullable=True)

    best_ml_team: Mapped[str | None] = mapped_column(String(5), nullable=True)
    best_ml_odds: Mapped[float | None] = mapped_column(Float, nullable=True)
    best_ml_value_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    best_ml_edge: Mapped[float | None] = mapped_column(Float, nullable=True)

    best_total_direction: Mapped[str | None] = mapped_column(String(10), nullable=True)  # over|under
    best_total_line: Mapped[float | None] = mapped_column(Float, nullable=True)
    best_total_odds: Mapped[float | None] = mapped_column(Float, nullable=True)
    best_total_value_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    best_total_edge: Mapped[float | None] = mapped_column(Float, nullable=True)

    best_bet_type: Mapped[str | None] = mapped_column(String(20), nullable=True)  # spread|moneyline|total
    best_bet_team: Mapped[str | None] = mapped_column(String(10), nullable=True)
    best_bet_line: Mapped[float | None] = mapped_column(Float, nullable=True)
    best_bet_odds: Mapped[float | None] = mapped_column(Float, nullable=True)
    best_bet_value_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    best_bet_edge: Mapped[float | None] = mapped_column(Float, nullable=True)

    # graded
    actual_winner: Mapped[str | None] = mapped_column(String(5), nullable=True)
    home_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    actual_margin: Mapped[int | None] = mapped_column(Integer, nullable=True)
    actual_total: Mapped[int | None] = mapped_column(Integer, nullable=True)
    best_spread_result: Mapped[str | None] = mapped_column(String(20), nullable=True)
    best_spread_profit: Mapped[float | None] = mapped_column(Float, nullable=True)
    best_ml_result: Mapped[str | None] = mapped_column(String(20), nullable=True)
    best_ml_profit: Mapped[float | None] = mapped_column(Float, nullable=True)
    best_total_result: Mapped[str | None] = mapped_column(String(20), nullable=True)
    best_total_profit: Mapped[float | None] = mapped_column(Float, nullable=True)
    best_bet_result: Mapped[str | None] = mapped_column(String(20), nullable=True)
    best_bet_profit: Mapped[float | None] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
