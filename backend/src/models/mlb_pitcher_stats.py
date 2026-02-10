"""MLB Pitcher rolling stats database model."""

from datetime import datetime, date
from decimal import Decimal

from sqlalchemy import String, DateTime, Date, Numeric, Integer, Index, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from src.database import Base


class MLBPitcherStats(Base):
    """Rolling pitcher statistics for a specific date.

    Stores pre-computed stats for model features.
    One row per pitcher per date.
    """

    __tablename__ = "mlb_pitcher_stats"

    stat_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    pitcher_id: Mapped[int] = mapped_column(Integer, ForeignKey("mlb_pitchers.pitcher_id"), nullable=False)
    stat_date: Mapped[date] = mapped_column(Date, nullable=False)

    # Core pitching stats
    era: Mapped[Decimal | None] = mapped_column(Numeric(5, 2), nullable=True)  # Earned Run Average
    whip: Mapped[Decimal | None] = mapped_column(Numeric(5, 3), nullable=True)  # Walks + Hits per Inning
    k_per_9: Mapped[Decimal | None] = mapped_column(Numeric(5, 2), nullable=True)  # Strikeouts per 9 innings
    bb_per_9: Mapped[Decimal | None] = mapped_column(Numeric(5, 2), nullable=True)  # Walks per 9 innings
    fip: Mapped[Decimal | None] = mapped_column(Numeric(5, 2), nullable=True)  # Fielding Independent Pitching

    # Workload
    innings_pitched: Mapped[Decimal | None] = mapped_column(Numeric(6, 1), nullable=True)
    games_started: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Advanced metrics
    k_pct: Mapped[Decimal | None] = mapped_column(Numeric(5, 3), nullable=True)  # Strikeout %
    bb_pct: Mapped[Decimal | None] = mapped_column(Numeric(5, 3), nullable=True)  # Walk %
    hr_per_9: Mapped[Decimal | None] = mapped_column(Numeric(5, 2), nullable=True)  # Home Runs per 9

    # Recent form (last 5 starts)
    era_l5: Mapped[Decimal | None] = mapped_column(Numeric(5, 2), nullable=True)
    whip_l5: Mapped[Decimal | None] = mapped_column(Numeric(5, 3), nullable=True)

    # Derived composite score (0-100)
    quality_score: Mapped[Decimal | None] = mapped_column(Numeric(5, 2), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow
    )

    __table_args__ = (
        Index("idx_mlb_pitcher_stats_pitcher", "pitcher_id"),
        Index("idx_mlb_pitcher_stats_date", "stat_date"),
        # Unique constraint for pitcher + date
        Index("idx_mlb_pitcher_stats_unique", "pitcher_id", "stat_date", unique=True),
    )
