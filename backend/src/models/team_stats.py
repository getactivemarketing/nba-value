"""Team rolling stats database model."""

from datetime import datetime, date
from decimal import Decimal

from sqlalchemy import String, DateTime, Date, Numeric, Integer, Index
from sqlalchemy.orm import Mapped, mapped_column

from src.database import Base


class TeamStats(Base):
    """Rolling team statistics for a specific date.

    Stores pre-computed rolling averages for model features.
    One row per team per date.
    """

    __tablename__ = "team_stats"

    # Composite primary key: team_id + stat_date
    team_id: Mapped[str] = mapped_column(String(10), primary_key=True)
    stat_date: Mapped[date] = mapped_column(Date, primary_key=True)

    # Games played context
    games_played: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Offensive Rating (points per 100 possessions) - rolling windows
    ortg_5: Mapped[Decimal | None] = mapped_column(Numeric(6, 2), nullable=True)  # Last 5 games
    ortg_10: Mapped[Decimal | None] = mapped_column(Numeric(6, 2), nullable=True)  # Last 10 games
    ortg_20: Mapped[Decimal | None] = mapped_column(Numeric(6, 2), nullable=True)  # Last 20 games
    ortg_season: Mapped[Decimal | None] = mapped_column(Numeric(6, 2), nullable=True)  # Season avg

    # Defensive Rating (opponent points per 100 possessions) - rolling windows
    drtg_5: Mapped[Decimal | None] = mapped_column(Numeric(6, 2), nullable=True)
    drtg_10: Mapped[Decimal | None] = mapped_column(Numeric(6, 2), nullable=True)
    drtg_20: Mapped[Decimal | None] = mapped_column(Numeric(6, 2), nullable=True)
    drtg_season: Mapped[Decimal | None] = mapped_column(Numeric(6, 2), nullable=True)

    # Net Rating (ORtg - DRtg)
    net_rtg_5: Mapped[Decimal | None] = mapped_column(Numeric(6, 2), nullable=True)
    net_rtg_10: Mapped[Decimal | None] = mapped_column(Numeric(6, 2), nullable=True)
    net_rtg_season: Mapped[Decimal | None] = mapped_column(Numeric(6, 2), nullable=True)

    # Pace (possessions per 48 minutes)
    pace_10: Mapped[Decimal | None] = mapped_column(Numeric(5, 2), nullable=True)
    pace_season: Mapped[Decimal | None] = mapped_column(Numeric(5, 2), nullable=True)

    # Points per game
    ppg_10: Mapped[Decimal | None] = mapped_column(Numeric(5, 2), nullable=True)
    ppg_season: Mapped[Decimal | None] = mapped_column(Numeric(5, 2), nullable=True)
    opp_ppg_10: Mapped[Decimal | None] = mapped_column(Numeric(5, 2), nullable=True)
    opp_ppg_season: Mapped[Decimal | None] = mapped_column(Numeric(5, 2), nullable=True)

    # Win/Loss context
    wins: Mapped[int] = mapped_column(Integer, default=0)
    losses: Mapped[int] = mapped_column(Integer, default=0)
    win_pct_10: Mapped[Decimal | None] = mapped_column(Numeric(4, 3), nullable=True)

    # Rest and schedule context
    days_rest: Mapped[int | None] = mapped_column(Integer, nullable=True)
    is_back_to_back: Mapped[bool | None] = mapped_column(nullable=True)
    games_last_7_days: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Home/Away splits
    home_win_pct: Mapped[Decimal | None] = mapped_column(Numeric(4, 3), nullable=True)
    away_win_pct: Mapped[Decimal | None] = mapped_column(Numeric(4, 3), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow
    )

    __table_args__ = (
        Index("idx_team_stats_team", "team_id"),
        Index("idx_team_stats_date", "stat_date"),
    )
