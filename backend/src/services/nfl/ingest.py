"""Persist nflverse-derived DataFrames into nfl_* tables (idempotent upserts)."""
import pandas as pd
import structlog
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.models import NFLGame, NFLGameContext, NFLTeamStats

logger = structlog.get_logger()


async def upsert_games(session: AsyncSession, game_rows: list[dict]) -> int:
    for row in game_rows:
        stmt = insert(NFLGame).values(**row).on_conflict_do_update(
            index_elements=["game_id"],
            set_={k: row[k] for k in row if k != "game_id"},
        )
        await session.execute(stmt)
    logger.info("nfl_upsert_games", count=len(game_rows))
    return len(game_rows)


def _is_dome(roof) -> bool:
    """True for enclosed venues; safe against NaN/None/non-str roof values."""
    return isinstance(roof, str) and roof.strip().lower() in ("dome", "closed")


async def upsert_game_context(session: AsyncSession, sched: pd.DataFrame) -> int:
    count = 0
    for _, g in sched.iterrows():
        values = {
            "game_id": g["game_id"],
            "home_rest_days": None if pd.isna(g.get("home_rest")) else int(g["home_rest"]),
            "away_rest_days": None if pd.isna(g.get("away_rest")) else int(g["away_rest"]),
            "temp_f": None if pd.isna(g.get("temp")) else float(g["temp"]),
            "wind_mph": None if pd.isna(g.get("wind")) else float(g["wind"]),
            "is_dome": _is_dome(g.get("roof")),
        }
        stmt = insert(NFLGameContext).values(**values).on_conflict_do_update(
            index_elements=["game_id"],
            set_={k: values[k] for k in values if k != "game_id"},
        )
        await session.execute(stmt)
        count += 1
    logger.info("nfl_upsert_game_context", count=count)
    return count


async def upsert_team_stats(session: AsyncSession, stats: pd.DataFrame) -> int:
    records = stats.to_dict("records")
    for rec in records:
        clean = {k: (None if pd.isna(v) else v) for k, v in rec.items()}
        stmt = insert(NFLTeamStats).values(**clean).on_conflict_do_update(
            index_elements=["team", "season", "through_week"],
            set_={k: clean[k] for k in clean
                  if k not in ("team", "season", "through_week", "id")},
        )
        await session.execute(stmt)
    logger.info("nfl_upsert_team_stats", count=len(records))
    return len(records)
