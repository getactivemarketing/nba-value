"""MLB Market database model for betting lines."""

from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from sqlalchemy import String, Integer, DateTime, Numeric, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database import Base

if TYPE_CHECKING:
    from src.models.mlb_game import MLBGame


class MLBMarket(Base):
    """MLB betting market lines."""

    __tablename__ = "mlb_markets"

    market_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    game_id: Mapped[str] = mapped_column(String(50), ForeignKey("mlb_games.game_id"), nullable=False, index=True)

    # Market type: moneyline, runline, total
    market_type: Mapped[str] = mapped_column(String(20), nullable=False)

    # Line (for runline: +/- 1.5, for total: over/under number)
    line: Mapped[Decimal | None] = mapped_column(Numeric(5, 1), nullable=True)

    # Odds in decimal format (e.g., 1.91 for -110)
    home_odds: Mapped[Decimal | None] = mapped_column(Numeric(6, 3), nullable=True)
    away_odds: Mapped[Decimal | None] = mapped_column(Numeric(6, 3), nullable=True)
    over_odds: Mapped[Decimal | None] = mapped_column(Numeric(6, 3), nullable=True)
    under_odds: Mapped[Decimal | None] = mapped_column(Numeric(6, 3), nullable=True)

    # Sportsbook source
    book: Mapped[str | None] = mapped_column(String(50), nullable=True)

    # Timestamps
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow
    )

    # Relationship
    game: Mapped["MLBGame"] = relationship("MLBGame", back_populates="markets")

    @property
    def home_odds_american(self) -> int | None:
        """Convert decimal odds to American format."""
        if self.home_odds is None:
            return None
        odds = float(self.home_odds)
        if odds >= 2.0:
            return int((odds - 1) * 100)
        else:
            return int(-100 / (odds - 1))

    @property
    def away_odds_american(self) -> int | None:
        """Convert decimal odds to American format."""
        if self.away_odds is None:
            return None
        odds = float(self.away_odds)
        if odds >= 2.0:
            return int((odds - 1) * 100)
        else:
            return int(-100 / (odds - 1))

    @property
    def home_implied_probability(self) -> float | None:
        """Calculate implied probability from home odds."""
        if self.home_odds is None:
            return None
        return 1.0 / float(self.home_odds)

    @property
    def away_implied_probability(self) -> float | None:
        """Calculate implied probability from away odds."""
        if self.away_odds is None:
            return None
        return 1.0 / float(self.away_odds)
