"""Backfill point-in-time team rate stats from MLB game logs.

`update_team_stats` only began writing OPS/AVG/OBP/SLG/ERA/WHIP on 2026-07-31,
so all 111 days of prior mlb_team_stats history carry NULLs — 0% coverage for
six features the run-diff model was trained on with real variance.

Writes one cumulative row per game day, dated the day after, so the existing
point-in-time lookup (`stat_date < game_date`, most recent first) returns what
was actually known at the time.

UPDATE-ONLY on the rate columns: the standings-derived fields already present
on these rows (runs_per_game, wins, last_10, first-inning stats) are correct
and must not be clobbered by a partial insert.

    python -m src.tasks.backfill_team_history [--season 2026]
"""

import asyncio
import sys
from datetime import date

import structlog
from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.config import settings
from src.models import MLBTeam, MLBTeamStats
from src.services.mlb.mlb_api import MLBStatsAPIClient
from src.services.mlb.team_history import build_team_stat_history

logger = structlog.get_logger()

RATE_COLUMNS = ("batting_avg", "obp", "slg", "ops", "era", "team_whip")


async def backfill(season: int = 2026) -> dict:
    engine = create_async_engine(settings.async_database_url, echo=False)
    Session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    client = MLBStatsAPIClient()

    teams_done = rows_updated = rows_inserted = failures = 0

    async with Session() as session:
        teams = list((await session.execute(
            select(MLBTeam).where(MLBTeam.external_id.isnot(None))
        )).scalars().all())

    print(f"[BACKFILL] {len(teams)} teams, season {season}", flush=True)

    for team in teams:
        try:
            hitting, pitching = await client.get_team_game_logs(int(team.external_id), season)
            history = build_team_stat_history(hitting, pitching)
            if not history:
                continue

            async with Session() as session:
                for row in history:
                    stat_date = date.fromisoformat(row["stat_date"])
                    values = {c: row[c] for c in RATE_COLUMNS}

                    existing = (await session.execute(
                        select(MLBTeamStats.stat_id).where(
                            MLBTeamStats.team_abbr == team.team_abbr,
                            MLBTeamStats.stat_date == stat_date,
                        )
                    )).scalar_one_or_none()

                    if existing:
                        # Only the rate columns — the standings-derived fields
                        # on this row are already correct.
                        await session.execute(
                            update(MLBTeamStats)
                            .where(MLBTeamStats.stat_id == existing)
                            .values(**values)
                        )
                        rows_updated += 1
                    else:
                        await session.execute(
                            insert(MLBTeamStats).values(
                                team_abbr=team.team_abbr,
                                stat_date=stat_date,
                                **values,
                            )
                        )
                        rows_inserted += 1
                await session.commit()

            teams_done += 1
            print(f"[BACKFILL] {team.team_abbr}: {len(history)} days", flush=True)

        except Exception as exc:
            failures += 1
            logger.warning("Team history backfill failed", team=team.team_abbr, error=str(exc))

    await engine.dispose()
    result = {
        "teams": teams_done,
        "rows_updated": rows_updated,
        "rows_inserted": rows_inserted,
        "failures": failures,
    }
    print(f"[BACKFILL] done: {result}", flush=True)
    return result


if __name__ == "__main__":
    args = sys.argv[1:]
    season = 2026
    for i, a in enumerate(args):
        if a == "--season" and i + 1 < len(args):
            season = int(args[i + 1])
    asyncio.run(backfill(season=season))
