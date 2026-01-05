"""Admin API endpoints for manual tasks."""

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from src.tasks.ingestion import (
    ingest_odds,
    update_nba_stats,
    sync_game_results,
)

router = APIRouter(prefix="/admin", tags=["Admin"])


class TaskResult(BaseModel):
    """Task execution result."""
    task_id: str | None = None
    status: str
    message: str


@router.post("/ingest/odds", response_model=TaskResult)
async def trigger_odds_ingestion(background_tasks: BackgroundTasks) -> TaskResult:
    """
    Manually trigger odds ingestion from The Odds API.

    This will fetch current odds for all NBA games and store them
    in the database. Normally runs automatically every 15 minutes.
    """
    try:
        # Run synchronously for immediate feedback
        result = ingest_odds()
        return TaskResult(
            status="completed",
            message=f"Ingested odds for {result.get('games_fetched', 0)} games, "
                    f"{result.get('snapshots_stored', 0)} snapshots stored. "
                    f"API requests remaining: {result.get('api_requests_remaining', 'unknown')}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest/stats", response_model=TaskResult)
async def trigger_stats_update(background_tasks: BackgroundTasks) -> TaskResult:
    """
    Manually trigger NBA stats update from BALLDONTLIE.

    This will fetch team information and recent game results.
    Normally runs automatically once per day.
    """
    try:
        result = update_nba_stats()
        return TaskResult(
            status="completed",
            message=f"Updated {result.get('teams_updated', 0)} teams, "
                    f"fetched {result.get('games_fetched', 0)} games.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sync/results", response_model=TaskResult)
async def trigger_results_sync(background_tasks: BackgroundTasks) -> TaskResult:
    """
    Manually sync game results for completed games.

    Updates final scores for any games that have finished.
    """
    try:
        result = sync_game_results()
        return TaskResult(
            status="completed",
            message=f"Updated {result.get('games_updated', 0)} game results.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_system_status() -> dict:
    """Get current system status and stats."""
    from datetime import datetime, timezone
    from sqlalchemy import select, func
    from src.database import async_session_maker
    from src.models import Game, OddsSnapshot, Team

    async with async_session_maker() as session:
        # Count games
        games_result = await session.execute(select(func.count(Game.game_id)))
        games_count = games_result.scalar() or 0

        # Count odds snapshots
        snapshots_result = await session.execute(select(func.count(OddsSnapshot.snapshot_id)))
        snapshots_count = snapshots_result.scalar() or 0

        # Count teams
        teams_result = await session.execute(select(func.count(Team.team_id)))
        teams_count = teams_result.scalar() or 0

        # Get latest snapshot time
        latest_result = await session.execute(
            select(OddsSnapshot.snapshot_time)
            .order_by(OddsSnapshot.snapshot_time.desc())
            .limit(1)
        )
        latest_snapshot = latest_result.scalar()

    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stats": {
            "games": games_count,
            "odds_snapshots": snapshots_count,
            "teams": teams_count,
            "latest_snapshot": latest_snapshot.isoformat() if latest_snapshot else None,
        },
    }
