"""Backfill NFL games + team stats from nflverse for a season range."""
import asyncio
import math
import sys

import structlog

from src.database import async_session_maker, init_db
from src.services.nfl.nfl_data import load_schedules, load_pbp, schedule_to_game_rows
from src.services.nfl.features import team_game_epa, rolling_team_stats
from src.services.nfl.ingest import upsert_games, upsert_game_context, upsert_team_stats

logger = structlog.get_logger()


def _clean_nan(row: dict) -> dict:
    """schedule_to_game_rows can leave raw NaN floats (not None) on optional
    string fields (e.g. surface, roof, QB names) when nflverse data is
    missing for a game; asyncpg rejects NaN for VARCHAR params. Sanitize
    defensively here rather than touching the already-reviewed ingest layer.
    """
    return {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in row.items()}


async def backfill_season(season: int) -> None:
    sched = load_schedules([season])
    sched = sched[sched["game_type"] == "REG"] if "game_type" in sched else sched
    game_rows = [_clean_nan(r) for r in schedule_to_game_rows(sched)]

    pbp = load_pbp([season])
    tg = team_game_epa(pbp)
    stats = rolling_team_stats(tg)

    async with async_session_maker() as session:
        await upsert_games(session, game_rows)
        await upsert_game_context(session, sched)
        await upsert_team_stats(session, stats)
        await session.commit()
    logger.info("nfl_backfill_season_done", season=season,
                games=len(game_rows), stat_rows=len(stats))


async def main(start: int, end: int) -> None:
    await init_db()   # ensure nfl_* tables exist
    for season in range(start, end + 1):
        await backfill_season(season)


if __name__ == "__main__":
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 2010
    end = int(sys.argv[2]) if len(sys.argv) > 2 else 2024
    asyncio.run(main(start, end))
