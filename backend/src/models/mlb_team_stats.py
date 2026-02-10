"""MLB Team rolling stats database model."""

from datetime import datetime, date
from decimal import Decimal

from sqlalchemy import String, DateTime, Date, Numeric, Integer, Index
from sqlalchemy.orm import Mapped, mapped_column

from src.database import Base


class MLBTeamStats(Base):
    """Rolling team statistics for a specific date.

    Stores pre-computed rolling averages for model features.
    One row per team per date.
    """

    __tablename__ = "mlb_team_stats"

    stat_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    team_abbr: Mapped[str] = mapped_column(String(5), nullable=False)
    stat_date: Mapped[date] = mapped_column(Date, nullable=False)

    # Offensive stats
    runs_per_game: Mapped[Decimal | None] = mapped_column(Numeric(5, 2), nullable=True)
    runs_allowed_per_game: Mapped[Decimal | None] = mapped_column(Numeric(5, 2), nullable=True)
    run_diff_per_game: Mapped[Decimal | None] = mapped_column(Numeric(6, 2), nullable=True)

    # Batting metrics
    ops: Mapped[Decimal | None] = mapped_column(Numeric(5, 3), nullable=True)  # On-base + Slugging
    batting_avg: Mapped[Decimal | None] = mapped_column(Numeric(5, 3), nullable=True)
    obp: Mapped[Decimal | None] = mapped_column(Numeric(5, 3), nullable=True)  # On-base percentage
    slg: Mapped[Decimal | None] = mapped_column(Numeric(5, 3), nullable=True)  # Slugging percentage

    # Pitching staff
    era: Mapped[Decimal | None] = mapped_column(Numeric(5, 2), nullable=True)  # Team ERA
    bullpen_era: Mapped[Decimal | None] = mapped_column(Numeric(5, 2), nullable=True)

    # Win/Loss context
    wins: Mapped[int | None] = mapped_column(Integer, nullable=True)
    losses: Mapped[int | None] = mapped_column(Integer, nullable=True)
    win_pct: Mapped[Decimal | None] = mapped_column(Numeric(4, 3), nullable=True)

    # Home/Away splits
    home_wins: Mapped[int | None] = mapped_column(Integer, nullable=True)
    home_losses: Mapped[int | None] = mapped_column(Integer, nullable=True)
    home_win_pct: Mapped[Decimal | None] = mapped_column(Numeric(4, 3), nullable=True)
    away_wins: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_losses: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_win_pct: Mapped[Decimal | None] = mapped_column(Numeric(4, 3), nullable=True)

    # Recent form
    last_10_wins: Mapped[int | None] = mapped_column(Integer, nullable=True)
    last_10_losses: Mapped[int | None] = mapped_column(Integer, nullable=True)
    last_10_record: Mapped[str | None] = mapped_column(String(10), nullable=True)  # e.g., "7-3"

    # Runs - rolling windows
    runs_per_game_l10: Mapped[Decimal | None] = mapped_column(Numeric(5, 2), nullable=True)
    runs_allowed_per_game_l10: Mapped[Decimal | None] = mapped_column(Numeric(5, 2), nullable=True)

    # Rest and schedule
    days_rest: Mapped[int | None] = mapped_column(Integer, nullable=True)
    games_last_7_days: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Betting records
    ats_wins: Mapped[int | None] = mapped_column(Integer, nullable=True)  # Against the spread (runline)
    ats_losses: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ats_pushes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ou_overs: Mapped[int | None] = mapped_column(Integer, nullable=True)  # Over/Under
    ou_unders: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ou_pushes: Mapped[int | None] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow
    )

    __table_args__ = (
        Index("idx_mlb_team_stats_team", "team_abbr"),
        Index("idx_mlb_team_stats_date", "stat_date"),
        Index("idx_mlb_team_stats_unique", "team_abbr", "stat_date", unique=True),
    )
