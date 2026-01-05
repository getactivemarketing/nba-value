"""Team database model."""

from datetime import datetime

from sqlalchemy import String, DateTime, Integer
from sqlalchemy.orm import Mapped, mapped_column

from src.database import Base


class Team(Base):
    """NBA team information."""

    __tablename__ = "teams"

    team_id: Mapped[str] = mapped_column(String(10), primary_key=True)  # e.g., "LAL", "BOS"
    external_id: Mapped[int | None] = mapped_column(Integer, nullable=True)  # BALLDONTLIE ID

    full_name: Mapped[str] = mapped_column(String(100), nullable=False)  # "Los Angeles Lakers"
    abbreviation: Mapped[str] = mapped_column(String(10), nullable=False)  # "LAL"
    city: Mapped[str] = mapped_column(String(50), nullable=False)
    name: Mapped[str] = mapped_column(String(50), nullable=False)  # "Lakers"
    conference: Mapped[str] = mapped_column(String(10), nullable=False)  # "West"
    division: Mapped[str] = mapped_column(String(20), nullable=False)  # "Pacific"

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )
