"""MLB Pitcher database model."""

from datetime import datetime

from sqlalchemy import String, DateTime, Integer
from sqlalchemy.orm import Mapped, mapped_column

from src.database import Base


class MLBPitcher(Base):
    """MLB pitcher information."""

    __tablename__ = "mlb_pitchers"

    pitcher_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    player_name: Mapped[str] = mapped_column(String(100), nullable=False)
    team_abbr: Mapped[str | None] = mapped_column(String(5), nullable=True)  # Current team
    throws: Mapped[str | None] = mapped_column(String(1), nullable=True)  # "L" or "R"

    # External ID from MLB Stats API
    external_id: Mapped[str | None] = mapped_column(String(50), nullable=True, index=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )
