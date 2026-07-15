"""NFL per-game situational context."""
from datetime import datetime
from sqlalchemy import String, Integer, Float, DateTime, Boolean, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column
from src.database import Base


class NFLGameContext(Base):
    __tablename__ = "nfl_game_context"

    game_id: Mapped[str] = mapped_column(
        String(20), ForeignKey("nfl_games.game_id"), primary_key=True
    )
    home_rest_days: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_rest_days: Mapped[int | None] = mapped_column(Integer, nullable=True)
    wind_mph: Mapped[float | None] = mapped_column(Float, nullable=True)
    temp_f: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_dome: Mapped[bool] = mapped_column(Boolean, default=False)

    # Best-effort candidate features (may be noisy; see spec).
    home_starters_out: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_starters_out: Mapped[int | None] = mapped_column(Integer, nullable=True)
    home_playoff_stakes: Mapped[str | None] = mapped_column(String(12), nullable=True)  # alive/clinched/eliminated
    away_playoff_stakes: Mapped[str | None] = mapped_column(String(12), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
