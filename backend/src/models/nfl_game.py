"""NFL game database model."""
from datetime import datetime
from sqlalchemy import String, Integer, DateTime, Boolean
from sqlalchemy.orm import Mapped, mapped_column
from src.database import Base


class NFLGame(Base):
    __tablename__ = "nfl_games"

    game_id: Mapped[str] = mapped_column(String(20), primary_key=True)  # nflverse game_id
    season: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    week: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    season_type: Mapped[str] = mapped_column(String(4), default="REG")  # REG / POST

    home_team: Mapped[str] = mapped_column(String(5), nullable=False)
    away_team: Mapped[str] = mapped_column(String(5), nullable=False)
    kickoff_utc: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    home_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_score: Mapped[int | None] = mapped_column(Integer, nullable=True)

    roof: Mapped[str | None] = mapped_column(String(12), nullable=True)      # dome/outdoors/closed/open
    surface: Mapped[str | None] = mapped_column(String(20), nullable=True)
    neutral_site: Mapped[bool] = mapped_column(Boolean, default=False)

    home_qb: Mapped[str | None] = mapped_column(String(50), nullable=True)
    home_qb_id: Mapped[str | None] = mapped_column(String(20), nullable=True)
    away_qb: Mapped[str | None] = mapped_column(String(50), nullable=True)
    away_qb_id: Mapped[str | None] = mapped_column(String(20), nullable=True)

    is_divisional: Mapped[bool] = mapped_column(Boolean, default=False)
    is_primetime: Mapped[bool] = mapped_column(Boolean, default=False)
    primetime_slot: Mapped[str | None] = mapped_column(String(4), nullable=True)  # TNF/SNF/MNF

    status: Mapped[str] = mapped_column(String(20), default="scheduled")

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )

    @property
    def margin(self) -> int | None:
        if self.home_score is not None and self.away_score is not None:
            return self.home_score - self.away_score
        return None

    @property
    def total_points(self) -> int | None:
        if self.home_score is not None and self.away_score is not None:
            return self.home_score + self.away_score
        return None
