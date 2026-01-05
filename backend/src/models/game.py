"""Game database model."""

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import String, Integer, DateTime, Date
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database import Base

if TYPE_CHECKING:
    from src.models.market import Market


class Game(Base):
    """NBA game information."""

    __tablename__ = "games"

    game_id: Mapped[str] = mapped_column(String(50), primary_key=True)
    league: Mapped[str] = mapped_column(String(10), default="NBA")
    season: Mapped[int] = mapped_column(Integer, nullable=False)
    game_date: Mapped[datetime] = mapped_column(Date, nullable=False)
    tip_time_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    home_team_id: Mapped[str] = mapped_column(String(10), nullable=False)
    away_team_id: Mapped[str] = mapped_column(String(10), nullable=False)

    home_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_score: Mapped[int | None] = mapped_column(Integer, nullable=True)

    status: Mapped[str] = mapped_column(String(20), default="scheduled")

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    markets: Mapped[list["Market"]] = relationship("Market", back_populates="game")

    @property
    def home_margin(self) -> int | None:
        """Calculate home team margin of victory."""
        if self.home_score is not None and self.away_score is not None:
            return self.home_score - self.away_score
        return None
