"""Model prediction database model."""

from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from sqlalchemy import BigInteger, String, Numeric, DateTime, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database import Base

if TYPE_CHECKING:
    from src.models.market import Market
    from src.models.score import ValueScore


class ModelPrediction(Base):
    """ML model prediction for a market."""

    __tablename__ = "model_predictions"

    prediction_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    market_id: Mapped[str] = mapped_column(
        String(50), ForeignKey("markets.market_id"), nullable=False
    )

    prediction_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    # Ensemble outputs
    p_ensemble_mean: Mapped[Decimal] = mapped_column(Numeric(5, 4), nullable=False)
    p_ensemble_std: Mapped[Decimal | None] = mapped_column(Numeric(5, 4), nullable=True)

    # Calibrated probability
    p_true: Mapped[Decimal] = mapped_column(Numeric(5, 4), nullable=False)

    # Market probability (de-vigged)
    p_market: Mapped[Decimal] = mapped_column(Numeric(5, 4), nullable=False)

    # Edge calculation
    raw_edge: Mapped[Decimal] = mapped_column(Numeric(6, 4), nullable=False)
    edge_band: Mapped[str | None] = mapped_column(String(20), nullable=True)

    # Feature snapshot reference
    features_snapshot_id: Mapped[str | None] = mapped_column(String(50), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow
    )

    # Relationships
    market: Mapped["Market"] = relationship("Market", back_populates="predictions")
    value_scores: Mapped[list["ValueScore"]] = relationship(
        "ValueScore", back_populates="prediction"
    )
