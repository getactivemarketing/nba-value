"""Database connection and session management."""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from src.config import settings

# Create async engine (use async_database_url for Railway compatibility)
engine = create_async_engine(
    settings.async_database_url,
    echo=settings.debug,
    pool_pre_ping=True,
    pool_size=3,
    max_overflow=2,
    pool_recycle=180,
    pool_timeout=30,
)

# Session factory
async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# Alias for convenience (context manager style)
async_session = async_session_maker


class Base(DeclarativeBase):
    """Base class for all database models."""

    pass


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for getting database sessions."""
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db() -> None:
    """Initialize database tables and run pending column migrations."""
    from sqlalchemy import text

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Add columns that create_all won't add to existing tables
    column_migrations = [
        ("mlb_games", "home_first_inning_runs", "INTEGER"),
        ("mlb_games", "away_first_inning_runs", "INTEGER"),
        ("mlb_team_stats", "first_inning_scored", "INTEGER"),
        ("mlb_team_stats", "first_inning_scoreless", "INTEGER"),
        ("mlb_team_stats", "first_inning_score_pct", "NUMERIC(4,3)"),
        ("mlb_team_stats", "first_inning_runs_avg", "NUMERIC(4,2)"),
        ("mlb_team_stats", "team_whip", "NUMERIC(5,3)"),
        ("mlb_games", "pregame_tweet_posted", "BOOLEAN DEFAULT FALSE"),
        ("mlb_games", "first_inning_tweet_posted", "BOOLEAN DEFAULT FALSE"),
        ("mlb_games", "final_tweet_posted", "BOOLEAN DEFAULT FALSE"),
        ("mlb_prediction_snapshots", "celebration_tweet_posted", "BOOLEAN DEFAULT FALSE"),
        ("prediction_snapshots", "celebration_tweet_posted", "BOOLEAN DEFAULT FALSE"),
    ]

    async with engine.begin() as conn:
        for table, column, col_type in column_migrations:
            try:
                await conn.execute(text(
                    f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} {col_type}"
                ))
            except Exception:
                pass  # Column already exists or table doesn't exist yet

    # Add unique constraints needed for ON CONFLICT upserts
    unique_constraints = [
        ("mlb_predictions", "uq_mlb_predictions_game_market", "game_id, market_type"),
    ]

    async with engine.begin() as conn:
        for table, name, columns in unique_constraints:
            try:
                # Check if constraint exists first (PostgreSQL doesn't have IF NOT EXISTS for constraints)
                exists_result = await conn.execute(text(
                    "SELECT 1 FROM pg_constraint WHERE conname = :name"
                ), {"name": name})
                if exists_result.scalar() is None:
                    await conn.execute(text(
                        f"ALTER TABLE {table} ADD CONSTRAINT {name} UNIQUE ({columns})"
                    ))
            except Exception:
                pass  # Constraint conflict or table doesn't exist
