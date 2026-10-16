# backend/src/models/edge.py
"""TruLine Edge tables. See docs/superpowers/specs/2026-10-15-edge-ev-finder-design.md §5.

edge_quotes is MUTABLE-CURRENT (one row per price key); edge_quote_history is
its append-only counterpart, written only when a key's price changes — it is
what makes line movement and closing-line value possible.
"""
from datetime import date, datetime

from sqlalchemy import (BigInteger, Boolean, Date, DateTime, Float, Index, Integer,
                        Numeric, String, Text, UniqueConstraint)
from sqlalchemy.orm import Mapped, mapped_column

from src.database import Base


class EdgeEvent(Base):
    __tablename__ = "edge_events"
    event_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    sport: Mapped[str] = mapped_column(String(40), nullable=False)
    home_team: Mapped[str] = mapped_column(String(60), nullable=False)
    away_team: Mapped[str] = mapped_column(String(60), nullable=False)
    commence_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(12), nullable=False, default="scheduled")
    completed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    home_score: Mapped[int | None] = mapped_column(Integer)
    away_score: Mapped[int | None] = mapped_column(Integer)
    last_lines_poll: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_props_poll: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class EdgeQuote(Base):
    __tablename__ = "edge_quotes"
    __table_args__ = (UniqueConstraint("event_id", "market", "player", "side", "line", "book",
                                       name="uq_edge_quotes_key"),)
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    market: Mapped[str] = mapped_column(String(48), nullable=False)
    player: Mapped[str] = mapped_column(String(80), nullable=False, default="")
    side: Mapped[str] = mapped_column(String(8), nullable=False)
    line: Mapped[float] = mapped_column(Numeric(6, 2), nullable=False)
    book: Mapped[str] = mapped_column(String(32), nullable=False)
    price: Mapped[int] = mapped_column(Integer, nullable=False)
    link: Mapped[str | None] = mapped_column(Text)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class EdgeQuoteHistory(Base):
    __tablename__ = "edge_quote_history"
    __table_args__ = (Index("ix_edge_hist_market", "event_id", "market", "player", "line", "observed_at"),)
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(String(64), nullable=False)
    market: Mapped[str] = mapped_column(String(48), nullable=False)
    player: Mapped[str] = mapped_column(String(80), nullable=False, default="")
    side: Mapped[str] = mapped_column(String(8), nullable=False)
    line: Mapped[float] = mapped_column(Numeric(6, 2), nullable=False)
    book: Mapped[str] = mapped_column(String(32), nullable=False)
    price: Mapped[int] = mapped_column(Integer, nullable=False)
    link: Mapped[str | None] = mapped_column(Text)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class EdgePlayer(Base):
    __tablename__ = "edge_players"
    odds_name: Mapped[str] = mapped_column(String(80), primary_key=True)
    bdl_player_id: Mapped[int | None] = mapped_column(Integer, index=True)
    team: Mapped[str | None] = mapped_column(String(4))
    match_method: Mapped[str] = mapped_column(String(12), nullable=False)  # exact|normalized|fuzzy|manual|unmatched
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class EdgePlayerGame(Base):
    __tablename__ = "edge_player_games"
    __table_args__ = (UniqueConstraint("bdl_player_id", "game_date", name="uq_edge_player_games"),
                      Index("ix_edge_pg_team_date", "team", "game_date"))
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    bdl_player_id: Mapped[int] = mapped_column(Integer, nullable=False)
    bdl_game_id: Mapped[int] = mapped_column(Integer, nullable=False)
    season: Mapped[int] = mapped_column(Integer, nullable=False)
    game_date: Mapped[date] = mapped_column(Date, nullable=False)
    team: Mapped[str] = mapped_column(String(4), nullable=False)
    opponent: Mapped[str] = mapped_column(String(4), nullable=False)
    is_home: Mapped[bool] = mapped_column(Boolean, nullable=False)
    minutes: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    pts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    reb: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    ast: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    fg3m: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    stl: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    blk: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    tov: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class EdgeOpportunity(Base):
    __tablename__ = "edge_opportunities"
    __table_args__ = (UniqueConstraint("event_id", "market", "player", "side", "line", "book", "price",
                                       name="uq_edge_opportunity"),)
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    market: Mapped[str] = mapped_column(String(48), nullable=False)
    player: Mapped[str] = mapped_column(String(80), nullable=False, default="")
    side: Mapped[str] = mapped_column(String(8), nullable=False)
    line: Mapped[float] = mapped_column(Numeric(6, 2), nullable=False)
    book: Mapped[str] = mapped_column(String(32), nullable=False)
    price: Mapped[int] = mapped_column(Integer, nullable=False)
    link: Mapped[str | None] = mapped_column(Text)
    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    fair_prob: Mapped[float] = mapped_column(Numeric(6, 5), nullable=False)
    ev: Mapped[float] = mapped_column(Numeric(7, 5), nullable=False)
    max_ev: Mapped[float] = mapped_column(Numeric(7, 5), nullable=False)
    n_books: Mapped[int] = mapped_column(Integer, nullable=False)
    devig_method: Mapped[str] = mapped_column(String(12), nullable=False, default="power")
    closing_done: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    closing_fair_prob: Mapped[float | None] = mapped_column(Numeric(6, 5))
    clv: Mapped[float | None] = mapped_column(Numeric(7, 5))
    clv_note: Mapped[str | None] = mapped_column(String(80))
    result: Mapped[str | None] = mapped_column(String(8))  # win|loss|push|void
    profit_units: Mapped[float | None] = mapped_column(Numeric(7, 4))


class EdgeApiCall(Base):
    __tablename__ = "edge_api_calls"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    called_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    endpoint: Mapped[str] = mapped_column(String(32), nullable=False)
    cost: Mapped[int] = mapped_column(Integer, nullable=False)
    remaining: Mapped[int | None] = mapped_column(Integer)


class EdgeWorkerState(Base):
    __tablename__ = "edge_worker_state"
    key: Mapped[str] = mapped_column(String(40), primary_key=True)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class EdgeNflPlayer(Base):
    """Odds API NFL player name -> BallDontLie NFL id. Keyed with team: NFL names repeat across teams."""
    __tablename__ = "edge_nfl_players"
    odds_name: Mapped[str] = mapped_column(String(80), primary_key=True)
    team: Mapped[str] = mapped_column(String(4), primary_key=True)
    bdl_player_id: Mapped[int | None] = mapped_column(Integer, index=True)
    position: Mapped[str | None] = mapped_column(String(8))
    match_method: Mapped[str] = mapped_column(String(12), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class EdgeNflGame(Base):
    __tablename__ = "edge_nfl_games"
    bdl_game_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    season: Mapped[int] = mapped_column(Integer, nullable=False)
    week: Mapped[int] = mapped_column(Integer, nullable=False)
    game_date: Mapped[date] = mapped_column(Date, nullable=False)
    home: Mapped[str] = mapped_column(String(4), nullable=False)
    away: Mapped[str] = mapped_column(String(4), nullable=False)
    postseason: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    status_state: Mapped[str | None] = mapped_column(String(16))
    final_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    stats_loaded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    snaps_loaded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class EdgeNflPlayerGame(Base):
    __tablename__ = "edge_nfl_player_games"
    __table_args__ = (UniqueConstraint("bdl_player_id", "bdl_game_id", name="uq_edge_nfl_player_games"),)
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    bdl_player_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    bdl_game_id: Mapped[int] = mapped_column(Integer, nullable=False)
    season: Mapped[int] = mapped_column(Integer, nullable=False)
    week: Mapped[int] = mapped_column(Integer, nullable=False)
    game_date: Mapped[date] = mapped_column(Date, nullable=False)
    team: Mapped[str] = mapped_column(String(4), nullable=False)
    opponent: Mapped[str] = mapped_column(String(4), nullable=False)
    is_home: Mapped[bool] = mapped_column(Boolean, nullable=False)
    pass_att: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    pass_cmp: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    pass_yds: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    pass_tds: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    pass_int: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    rush_att: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    rush_yds: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    receptions: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    rec_yds: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class EdgeNflSnap(Base):
    """nflverse snap counts: the participation record that tells 'played, no stats' from 'inactive'."""
    __tablename__ = "edge_nfl_snaps"
    __table_args__ = (UniqueConstraint("season", "week", "team", "player_norm", name="uq_edge_nfl_snaps"),)
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    season: Mapped[int] = mapped_column(Integer, nullable=False)
    week: Mapped[int] = mapped_column(Integer, nullable=False)
    team: Mapped[str] = mapped_column(String(4), nullable=False)
    player_norm: Mapped[str] = mapped_column(String(80), nullable=False)
    snaps: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    loaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
