"""Value Score database model."""

from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from sqlalchemy import BigInteger, String, Numeric, DateTime, ForeignKey, Integer, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database import Base

if TYPE_CHECKING:
    from src.models.market import Market
    from src.models.prediction import ModelPrediction


class ValueScore(Base):
    """Computed Value Score for a market (stores both algorithms)."""

    __tablename__ = "value_scores"

    value_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    prediction_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("model_predictions.prediction_id"), nullable=False
    )
    market_id: Mapped[str] = mapped_column(
        String(50), ForeignKey("markets.market_id"), nullable=False
    )

    calc_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    # Algorithm A (Idea 1 style) outputs
    algo_a_edge_score: Mapped[Decimal | None] = mapped_column(Numeric(6, 4), nullable=True)
    algo_a_confidence: Mapped[Decimal | None] = mapped_column(Numeric(5, 3), nullable=True)
    algo_a_market_quality: Mapped[Decimal | None] = mapped_column(Numeric(5, 3), nullable=True)
    algo_a_value_score: Mapped[Decimal | None] = mapped_column(Numeric(6, 2), nullable=True)

    # Algorithm B (Idea 2 style) outputs
    algo_b_combined_edge: Mapped[Decimal | None] = mapped_column(Numeric(6, 4), nullable=True)
    algo_b_confidence: Mapped[Decimal | None] = mapped_column(Numeric(5, 3), nullable=True)
    algo_b_market_quality: Mapped[Decimal | None] = mapped_column(Numeric(5, 3), nullable=True)
    algo_b_value_score: Mapped[Decimal | None] = mapped_column(Numeric(6, 2), nullable=True)

    # Active algorithm flag
    active_algorithm: Mapped[str] = mapped_column(String(1), default="A")

    # Context
    time_to_tip_minutes: Mapped[int | None] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow
    )

    # Relationships
    market: Mapped["Market"] = relationship("Market", back_populates="value_scores")
    prediction: Mapped["ModelPrediction"] = relationship(
        "ModelPrediction", back_populates="value_scores"
    )

    __table_args__ = (
        Index("idx_scores_market_calc", "market_id", "calc_time"),
        Index("idx_scores_algo_a", "algo_a_value_score"),
        Index("idx_scores_algo_b", "algo_b_value_score"),
    )
