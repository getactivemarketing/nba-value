"""NFL team database model."""
from datetime import datetime
from sqlalchemy import String, DateTime
from sqlalchemy.orm import Mapped, mapped_column
from src.database import Base


class NFLTeam(Base):
    __tablename__ = "nfl_teams"

    abbr: Mapped[str] = mapped_column(String(5), primary_key=True)
    name: Mapped[str] = mapped_column(String(50), nullable=False)
    conference: Mapped[str | None] = mapped_column(String(3), nullable=True)
    division: Mapped[str | None] = mapped_column(String(20), nullable=True)
    primary_color: Mapped[str | None] = mapped_column(String(7), nullable=True)
    secondary_color: Mapped[str | None] = mapped_column(String(7), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
