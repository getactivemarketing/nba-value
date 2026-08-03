"""Append-only NFL odds history.

`nfl_markets` is MUTABLE-CURRENT — overwritten on every refresh. This is its
append-only counterpart: one row per (game, book) each time that book's prices
move, which is what makes line movement, closing-line value, and the
market-intelligence layer possible.

Added BEFORE the 2026 season opens, deliberately. The MLB equivalent arrived
2026-07-31, four months into the season, and everything before it is
permanently unrecoverable.

Column names mirror `mlb_odds_snapshots` where the concept is shared so one
market-intelligence service can read across verticals; NFL adds a second
priced line (spread AND total), which MLB's runline/total split does not.

See `docs/engineering/05-api-data-contracts.md` §3 (durability classes).
"""

from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Index,
    Integer,
    Numeric,
    String,
)
from sqlalchemy.orm import Mapped, mapped_column

from src.database import Base


class NFLOddsSnapshot(Base):
    """Point-in-time NFL prices from one sportsbook, written only on change."""

    __tablename__ = "nfl_odds_snapshots"

    snapshot_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    game_id: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    book: Mapped[str] = mapped_column(String(50), nullable=False)

    snapshot_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    # Negative once the game has started — the sign is what keeps an in-game
    # price from being mistaken for a close.
    minutes_to_kickoff: Mapped[int | None] = mapped_column(Integer, nullable=True)

    ml_home: Mapped[Decimal | None] = mapped_column(Numeric(7, 3), nullable=True)
    ml_away: Mapped[Decimal | None] = mapped_column(Numeric(7, 3), nullable=True)

    # SIGNED home spread: -3.5 means the home team lays 3.5.
    spread_line: Mapped[Decimal | None] = mapped_column(Numeric(5, 1), nullable=True)
    spread_home_odds: Mapped[Decimal | None] = mapped_column(Numeric(7, 3), nullable=True)
    spread_away_odds: Mapped[Decimal | None] = mapped_column(Numeric(7, 3), nullable=True)

    total_line: Mapped[Decimal | None] = mapped_column(Numeric(5, 1), nullable=True)
    over_odds: Mapped[Decimal | None] = mapped_column(Numeric(7, 3), nullable=True)
    under_odds: Mapped[Decimal | None] = mapped_column(Numeric(7, 3), nullable=True)

    # Assigned AFTER kickoff to the last row at or before it, per (game, book).
    # Never set at insert — the NBA writer hardcoded its equivalent False and
    # it was never true for any row, which is why CLV was uncomputable.
    is_closing_line: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(), nullable=False
    )

    __table_args__ = (
        Index("ix_nfl_odds_snap_game_book_time", "game_id", "book", "snapshot_time"),
        Index("ix_nfl_odds_snap_closing", "game_id", "is_closing_line"),
    )

    def __repr__(self) -> str:
        return (
            f"<NFLOddsSnapshot {self.game_id} {self.book} @{self.snapshot_time} "
            f"spread={self.spread_line} total={self.total_line}>"
        )
