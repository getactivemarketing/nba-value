"""Backfill point-in-time pitcher stats from MLB game logs.

`mlb_pitcher_stats` held exactly one snapshot per pitcher — whatever the
season-to-date numbers were when the ingest last ran. That is unusable for
training: an April game would be fed July's ERA.

This walks each pitcher's game log and writes one cumulative row per
appearance, dated the day after it, so the existing point-in-time lookup
(`stat_date < game_date`, most recent first) returns what was actually known
at the time.

Run once to seed history; the daily ingest keeps it current going forward.

    python -m src.tasks.backfill_pitcher_history [--season 2026] [--limit N]
"""

import asyncio
import sys
from datetime import date

import structlog
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.config import settings
from src.models import MLBPitcher, MLBPitcherStats
from src.services.mlb.mlb_api import MLBStatsAPIClient
from src.services.mlb.pitcher_history import build_stat_history
from src.services.mlb.pitcher_quality import PitcherQualityScorer

logger = structlog.get_logger()


async def backfill(season: int = 2026, limit: int | None = None) -> dict:
    engine = create_async_engine(settings.async_database_url, echo=False)
    Session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    client = MLBStatsAPIClient()

    pitchers_done = rows_written = failures = 0

    async with Session() as session:
        stmt = select(MLBPitcher).where(MLBPitcher.external_id.isnot(None))
        if limit:
            stmt = stmt.limit(limit)
        pitchers = list((await session.execute(stmt)).scalars().all())

    print(f"[BACKFILL] {len(pitchers)} pitchers, season {season}", flush=True)

    for index, pitcher in enumerate(pitchers, start=1):
        try:
            logs = await client.get_pitcher_game_logs(int(pitcher.external_id), season)
            history = build_stat_history(logs)
            if not history:
                continue

            async with Session() as session:
                for row in history:
                    quality = PitcherQualityScorer.calculate_quality_score(
                        era=row["era"],
                        whip=row["whip"],
                        k_per_9=row["k_per_9"],
                        bb_per_9=row["bb_per_9"],
                    )
                    await session.execute(
                        insert(MLBPitcherStats)
                        .values(
                            pitcher_id=pitcher.pitcher_id,
                            stat_date=date.fromisoformat(row["stat_date"]),
                            era=row["era"],
                            whip=row["whip"],
                            k_per_9=row["k_per_9"],
                            bb_per_9=row["bb_per_9"],
                            fip=row["fip"],
                            innings_pitched=row["innings_pitched"],
                            games_started=row["games_started"],
                            quality_score=quality,
                        )
                        .on_conflict_do_update(
                            index_elements=["pitcher_id", "stat_date"],
                            set_={
                                "era": row["era"],
                                "whip": row["whip"],
                                "k_per_9": row["k_per_9"],
                                "bb_per_9": row["bb_per_9"],
                                "fip": row["fip"],
                                "innings_pitched": row["innings_pitched"],
                                "games_started": row["games_started"],
                                "quality_score": quality,
                            },
                        )
                    )
                    rows_written += 1
                await session.commit()

            pitchers_done += 1
            if index % 50 == 0:
                print(
                    f"[BACKFILL] {index}/{len(pitchers)} pitchers, {rows_written} rows",
                    flush=True,
                )

        except Exception as exc:
            failures += 1
            logger.warning(
                "Pitcher history backfill failed",
                pitcher_id=pitcher.pitcher_id,
                error=str(exc),
            )

    await engine.dispose()
    result = {
        "pitchers_with_history": pitchers_done,
        "rows_written": rows_written,
        "failures": failures,
    }
    print(f"[BACKFILL] done: {result}", flush=True)
    return result


if __name__ == "__main__":
    args = sys.argv[1:]
    season = 2026
    limit = None
    for i, arg in enumerate(args):
        if arg == "--season" and i + 1 < len(args):
            season = int(args[i + 1])
        elif arg == "--limit" and i + 1 < len(args):
            limit = int(args[i + 1])
    asyncio.run(backfill(season=season, limit=limit))
