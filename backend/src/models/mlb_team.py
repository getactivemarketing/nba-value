"""MLB Team database model."""

from datetime import datetime

from sqlalchemy import String, DateTime
from sqlalchemy.orm import Mapped, mapped_column

from src.database import Base


class MLBTeam(Base):
    """MLB team information."""

    __tablename__ = "mlb_teams"

    team_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    team_abbr: Mapped[str] = mapped_column(String(5), unique=True, nullable=False)  # e.g., "NYY", "LAD"
    team_name: Mapped[str] = mapped_column(String(100), nullable=False)  # "New York Yankees"
    league: Mapped[str] = mapped_column(String(2), nullable=False)  # "AL", "NL"
    division: Mapped[str] = mapped_column(String(10), nullable=False)  # "East", "Central", "West"

    # External API ID from MLB Stats API
    external_id: Mapped[int | None] = mapped_column(nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )
