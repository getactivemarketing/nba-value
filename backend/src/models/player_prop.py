"""Player prop database model for storing player prop betting lines."""

from datetime import datetime
from decimal import Decimal

from sqlalchemy import String, DateTime, Numeric, Integer, Index, BigInteger
from sqlalchemy.orm import Mapped, mapped_column

from src.database import Base


class PlayerProp(Base):
    """Player prop betting line from sportsbooks.

    Stores player prop markets like points, rebounds, assists, threes, PRA, etc.
    """

    __tablename__ = "player_props"

    prop_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    # Game reference (external ID from odds API)
    game_id: Mapped[str] = mapped_column(String(50), nullable=False)

    # Player identification
    player_name: Mapped[str] = mapped_column(String(100), nullable=False)
    player_team: Mapped[str | None] = mapped_column(String(10), nullable=True)

    # Prop details
    prop_type: Mapped[str] = mapped_column(String(30), nullable=False)  # points, rebounds, assists, threes, pra, etc.
    line: Mapped[Decimal] = mapped_column(Numeric(5, 2), nullable=False)

    # Odds (decimal format)
    over_odds: Mapped[Decimal | None] = mapped_column(Numeric(6, 3), nullable=True)
    under_odds: Mapped[Decimal | None] = mapped_column(Numeric(6, 3), nullable=True)

    # Sportsbook
    book: Mapped[str] = mapped_column(String(50), nullable=False)

    # Timing
    snapshot_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow
    )

    __table_args__ = (
        Index("idx_props_game", "game_id"),
        Index("idx_props_player", "player_name"),
        Index("idx_props_game_player", "game_id", "player_name"),
        Index("idx_props_snapshot", "snapshot_time"),
    )
