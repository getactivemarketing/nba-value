"""MLB Prediction database model."""

from datetime import datetime
from decimal import Decimal

from sqlalchemy import String, Integer, DateTime, Numeric, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from src.database import Base


class MLBPrediction(Base):
    """MLB model predictions."""

    __tablename__ = "mlb_predictions"

    prediction_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    game_id: Mapped[str] = mapped_column(String(50), ForeignKey("mlb_games.game_id"), nullable=False, index=True)

    # Market type: moneyline, runline, total
    market_type: Mapped[str] = mapped_column(String(20), nullable=False)

    # Model outputs
    predicted_run_diff: Mapped[Decimal | None] = mapped_column(Numeric(5, 2), nullable=True)  # Home - Away
    predicted_total: Mapped[Decimal | None] = mapped_column(Numeric(5, 1), nullable=True)

    # Win probabilities
    p_home_win: Mapped[Decimal | None] = mapped_column(Numeric(4, 3), nullable=True)
    p_away_win: Mapped[Decimal | None] = mapped_column(Numeric(4, 3), nullable=True)

    # Runline probabilities
    p_home_cover: Mapped[Decimal | None] = mapped_column(Numeric(4, 3), nullable=True)
    p_away_cover: Mapped[Decimal | None] = mapped_column(Numeric(4, 3), nullable=True)

    # Total probabilities
    p_over: Mapped[Decimal | None] = mapped_column(Numeric(4, 3), nullable=True)
    p_under: Mapped[Decimal | None] = mapped_column(Numeric(4, 3), nullable=True)

    # Market comparison
    market_implied_prob: Mapped[Decimal | None] = mapped_column(Numeric(4, 3), nullable=True)

    # Edge calculation
    edge: Mapped[Decimal | None] = mapped_column(Numeric(6, 2), nullable=True)
    edge_pct: Mapped[Decimal | None] = mapped_column(Numeric(6, 2), nullable=True)

    # Value scoring
    value_score: Mapped[Decimal | None] = mapped_column(Numeric(5, 1), nullable=True)
    confidence: Mapped[str | None] = mapped_column(String(20), nullable=True)  # high, medium, low

    # Best bet recommendation
    recommendation: Mapped[str | None] = mapped_column(String(30), nullable=True)  # e.g., "home_ml", "over"

    # Model metadata
    model_version: Mapped[str | None] = mapped_column(String(20), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow
    )
