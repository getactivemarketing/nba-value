"""MLB Game database model."""

from datetime import datetime, date
from typing import TYPE_CHECKING

from sqlalchemy import String, Integer, DateTime, Date, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database import Base

if TYPE_CHECKING:
    from src.models.mlb_market import MLBMarket
    from src.models.mlb_pitcher import MLBPitcher


class MLBGame(Base):
    """MLB game information."""

    __tablename__ = "mlb_games"

    game_id: Mapped[str] = mapped_column(String(50), primary_key=True)
    game_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    game_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Teams
    home_team: Mapped[str] = mapped_column(String(5), nullable=False)
    away_team: Mapped[str] = mapped_column(String(5), nullable=False)

    # Starting pitchers
    home_starter_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("mlb_pitchers.pitcher_id"), nullable=True
    )
    away_starter_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("mlb_pitchers.pitcher_id"), nullable=True
    )

    # Game status
    status: Mapped[str] = mapped_column(String(20), default="scheduled")  # scheduled, in_progress, final
    inning: Mapped[int | None] = mapped_column(Integer, nullable=True)
    inning_state: Mapped[str | None] = mapped_column(String(10), nullable=True)  # top, middle, bottom

    # Final scores
    home_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_score: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # MLB Stats API ID for data fetching
    external_id: Mapped[str | None] = mapped_column(String(50), nullable=True, index=True)

    # Season info
    season: Mapped[int | None] = mapped_column(Integer, nullable=True)
    game_type: Mapped[str | None] = mapped_column(String(5), nullable=True)  # R=regular, P=playoff, S=spring

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    markets: Mapped[list["MLBMarket"]] = relationship("MLBMarket", back_populates="game")
    home_starter: Mapped["MLBPitcher | None"] = relationship(
        "MLBPitcher", foreign_keys=[home_starter_id]
    )
    away_starter: Mapped["MLBPitcher | None"] = relationship(
        "MLBPitcher", foreign_keys=[away_starter_id]
    )

    @property
    def run_differential(self) -> int | None:
        """Calculate home team run differential."""
        if self.home_score is not None and self.away_score is not None:
            return self.home_score - self.away_score
        return None

    @property
    def total_runs(self) -> int | None:
        """Calculate total runs scored."""
        if self.home_score is not None and self.away_score is not None:
            return self.home_score + self.away_score
        return None
