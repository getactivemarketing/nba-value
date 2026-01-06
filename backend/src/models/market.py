"""Market database model."""

from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from sqlalchemy import String, Numeric, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database import Base

if TYPE_CHECKING:
    from src.models.game import Game
    from src.models.prediction import ModelPrediction
    from src.models.score import ValueScore


class Market(Base):
    """Betting market for a game."""

    __tablename__ = "markets"

    market_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    game_id: Mapped[str] = mapped_column(
        String(50), ForeignKey("games.game_id"), nullable=False
    )

    market_type: Mapped[str] = mapped_column(String(20), nullable=False)
    outcome_label: Mapped[str] = mapped_column(String(100), nullable=False)

    line: Mapped[Decimal | None] = mapped_column(Numeric(10, 2), nullable=True)
    odds_decimal: Mapped[Decimal] = mapped_column(Numeric(5, 3), nullable=False)
    book: Mapped[str | None] = mapped_column(String(50), nullable=True)

    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    game: Mapped["Game"] = relationship("Game", back_populates="markets")
    predictions: Mapped[list["ModelPrediction"]] = relationship(
        "ModelPrediction", back_populates="market"
    )
    value_scores: Mapped[list["ValueScore"]] = relationship(
        "ValueScore", back_populates="market"
    )

    @property
    def odds_american(self) -> int:
        """Convert decimal odds to American format."""
        decimal = float(self.odds_decimal)
        if decimal >= 2.0:
            return int((decimal - 1) * 100)
        else:
            return int(-100 / (decimal - 1))
