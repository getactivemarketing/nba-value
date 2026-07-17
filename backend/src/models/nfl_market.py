"""NFL betting market odds rows."""
from datetime import datetime
from sqlalchemy import String, Integer, Float, DateTime, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column
from src.database import Base


class NFLMarket(Base):
    __tablename__ = "nfl_markets"

    market_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    game_id: Mapped[str] = mapped_column(String(20), ForeignKey("nfl_games.game_id"), nullable=False, index=True)
    market_type: Mapped[str] = mapped_column(String(20), nullable=False)  # spread|moneyline|total
    line: Mapped[float | None] = mapped_column(Float, nullable=True)      # spread (home fav +) or total
    home_odds: Mapped[float | None] = mapped_column(Float, nullable=True)  # decimal
    away_odds: Mapped[float | None] = mapped_column(Float, nullable=True)
    over_odds: Mapped[float | None] = mapped_column(Float, nullable=True)
    under_odds: Mapped[float | None] = mapped_column(Float, nullable=True)
    book: Mapped[str | None] = mapped_column(String(50), nullable=True)
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
