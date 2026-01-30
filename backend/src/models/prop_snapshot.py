"""Prop snapshot model for tracking prop predictions and outcomes."""

from datetime import datetime, date
from decimal import Decimal
from sqlalchemy import String, Numeric, DateTime, Date, Text, BigInteger, Index
from sqlalchemy.orm import Mapped, mapped_column

from src.models.base import Base


class PropSnapshot(Base):
    """Stores prop predictions for later evaluation."""

    __tablename__ = "prop_snapshots"

    snapshot_id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True
    )
    game_id: Mapped[str] = mapped_column(String(50), nullable=False)
    player_name: Mapped[str] = mapped_column(String(100), nullable=False)
    prop_type: Mapped[str] = mapped_column(String(30), nullable=False)
    line: Mapped[Decimal] = mapped_column(Numeric(5, 2), nullable=False)
    over_odds: Mapped[Decimal | None] = mapped_column(Numeric(6, 3), nullable=True)
    under_odds: Mapped[Decimal | None] = mapped_column(Numeric(6, 3), nullable=True)
    book: Mapped[str | None] = mapped_column(String(50), nullable=True)

    # Prediction fields
    season_avg: Mapped[Decimal | None] = mapped_column(Numeric(6, 2), nullable=True)
    edge: Mapped[Decimal | None] = mapped_column(Numeric(6, 2), nullable=True)
    edge_pct: Mapped[Decimal | None] = mapped_column(Numeric(6, 2), nullable=True)
    recommendation: Mapped[str | None] = mapped_column(String(10), nullable=True)
    value_score: Mapped[Decimal | None] = mapped_column(Numeric(5, 1), nullable=True)
    reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Timing
    snapshot_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )
    game_date: Mapped[date | None] = mapped_column(Date, nullable=True)

    # Grading (filled in after game completes)
    actual_value: Mapped[Decimal | None] = mapped_column(Numeric(6, 2), nullable=True)
    result: Mapped[str | None] = mapped_column(String(10), nullable=True)  # WIN, LOSS, PUSH
    graded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("idx_prop_snapshots_game", "game_id"),
        Index("idx_prop_snapshots_date", "game_date"),
        Index("idx_prop_snapshots_ungraded", "game_date", postgresql_where="result IS NULL"),
        Index("idx_prop_snapshots_player", "player_name"),
    )

    def __repr__(self) -> str:
        return f"<PropSnapshot {self.player_name} {self.prop_type} {self.line} ({self.recommendation})>"
