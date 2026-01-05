"""Odds snapshot database model for time-series tracking."""

from datetime import datetime
from decimal import Decimal

from sqlalchemy import String, DateTime, Numeric, Integer, Index, BigInteger
from sqlalchemy.orm import Mapped, mapped_column

from src.database import Base


class OddsSnapshot(Base):
    """Point-in-time snapshot of odds from a sportsbook.

    Stores historical odds for CLV (Closing Line Value) calculation
    and odds movement analysis.
    """

    __tablename__ = "odds_snapshots"

    snapshot_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    # Game reference (external ID from odds API)
    game_id: Mapped[str] = mapped_column(String(50), nullable=False)

    # Market identification
    market_type: Mapped[str] = mapped_column(String(20), nullable=False)  # spread, moneyline, total
    book_key: Mapped[str] = mapped_column(String(50), nullable=False)  # fanduel, draftkings, etc.

    # Snapshot timing
    snapshot_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    minutes_to_tip: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Spread market
    home_spread: Mapped[Decimal | None] = mapped_column(Numeric(5, 2), nullable=True)
    home_spread_odds: Mapped[Decimal | None] = mapped_column(Numeric(6, 3), nullable=True)
    away_spread: Mapped[Decimal | None] = mapped_column(Numeric(5, 2), nullable=True)
    away_spread_odds: Mapped[Decimal | None] = mapped_column(Numeric(6, 3), nullable=True)

    # Moneyline market (decimal odds)
    home_ml_odds: Mapped[Decimal | None] = mapped_column(Numeric(6, 3), nullable=True)
    away_ml_odds: Mapped[Decimal | None] = mapped_column(Numeric(6, 3), nullable=True)

    # Totals market
    total_line: Mapped[Decimal | None] = mapped_column(Numeric(5, 2), nullable=True)
    over_odds: Mapped[Decimal | None] = mapped_column(Numeric(6, 3), nullable=True)
    under_odds: Mapped[Decimal | None] = mapped_column(Numeric(6, 3), nullable=True)

    # Computed implied probabilities (de-vigged)
    home_spread_prob: Mapped[Decimal | None] = mapped_column(Numeric(5, 4), nullable=True)
    home_ml_prob: Mapped[Decimal | None] = mapped_column(Numeric(5, 4), nullable=True)
    over_prob: Mapped[Decimal | None] = mapped_column(Numeric(5, 4), nullable=True)

    # Is this the closing line?
    is_closing_line: Mapped[bool] = mapped_column(default=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow
    )

    __table_args__ = (
        Index("idx_odds_game_time", "game_id", "snapshot_time"),
        Index("idx_odds_game_book", "game_id", "book_key"),
        Index("idx_odds_closing", "game_id", "is_closing_line"),
        Index("idx_odds_snapshot_time", "snapshot_time"),
    )
