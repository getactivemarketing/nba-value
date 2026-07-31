"""Append-only MLB odds history.

`mlb_markets` is MUTABLE-CURRENT (overwritten in place on every ingest). This
table is the append-only counterpart: one row per (game, book) each time that
book's prices actually move. It is what makes line movement, closing-line
value, and the Phase 4 market-dynamics features possible.

Column names deliberately mirror the NBA `odds_snapshots` table so a single
market-intelligence service can read across verticals. `home_spread` there is
the runline here — same shape, sport-specific meaning.

See `docs/engineering/06-roadmap-2026H2.md` Phase 1 and
`docs/engineering/05-api-data-contracts.md` §3 (durability classes).
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


class MLBOddsSnapshot(Base):
    """Point-in-time MLB prices from one sportsbook, written only on change."""

    __tablename__ = "mlb_odds_snapshots"

    snapshot_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    game_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    book: Mapped[str] = mapped_column(String(50), nullable=False)

    # When this price was observed, and how far that was from first pitch.
    # minutes_to_first_pitch is negative for anything seen after the game began.
    snapshot_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    minutes_to_first_pitch: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Moneyline (decimal odds)
    ml_home: Mapped[Decimal | None] = mapped_column(Numeric(6, 3), nullable=True)
    ml_away: Mapped[Decimal | None] = mapped_column(Numeric(6, 3), nullable=True)

    # Runline. rl_line is the SIGNED home line (-1.5 = home favoured), matching
    # the 2026-07-21 sign-pairing fix in the scorer.
    rl_line: Mapped[Decimal | None] = mapped_column(Numeric(5, 1), nullable=True)
    rl_home_odds: Mapped[Decimal | None] = mapped_column(Numeric(6, 3), nullable=True)
    rl_away_odds: Mapped[Decimal | None] = mapped_column(Numeric(6, 3), nullable=True)

    # Total
    total_line: Mapped[Decimal | None] = mapped_column(Numeric(5, 1), nullable=True)
    over_odds: Mapped[Decimal | None] = mapped_column(Numeric(6, 3), nullable=True)
    under_odds: Mapped[Decimal | None] = mapped_column(Numeric(6, 3), nullable=True)

    # Set by the closing-line marker after first pitch: the last row observed
    # at or before first pitch, per (game, book). NEVER set at insert time —
    # the NBA writer hardcoded this False and it has never been true for any
    # row, which is why CLV has never been computable.
    is_closing_line: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(), nullable=False
    )

    __table_args__ = (
        # Change-detection reads the newest row per (game, book) on every ingest.
        Index("ix_mlb_odds_snap_game_book_time", "game_id", "book", "snapshot_time"),
        # Closing-line marking and CLV joins scan by game.
        Index("ix_mlb_odds_snap_closing", "game_id", "is_closing_line"),
    )

    def __repr__(self) -> str:
        return (
            f"<MLBOddsSnapshot {self.game_id} {self.book} "
            f"@{self.snapshot_time} ml={self.ml_home}/{self.ml_away}>"
        )
