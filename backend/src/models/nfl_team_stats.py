"""NFL rolling team form, one row per team per through-week (point-in-time)."""
from datetime import datetime
from sqlalchemy import String, Integer, Float, DateTime, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column
from src.database import Base


class NFLTeamStats(Base):
    __tablename__ = "nfl_team_stats"
    __table_args__ = (
        UniqueConstraint("team", "season", "through_week", name="uq_nfl_team_stats_team_season_week"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    team: Mapped[str] = mapped_column(String(5), nullable=False, index=True)
    season: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    through_week: Mapped[int] = mapped_column(Integer, nullable=False, index=True)

    off_epa_play: Mapped[float | None] = mapped_column(Float, nullable=True)
    def_epa_play: Mapped[float | None] = mapped_column(Float, nullable=True)
    pass_epa: Mapped[float | None] = mapped_column(Float, nullable=True)
    rush_epa: Mapped[float | None] = mapped_column(Float, nullable=True)
    success_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    pace: Mapped[float | None] = mapped_column(Float, nullable=True)
    power_rating: Mapped[float | None] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
