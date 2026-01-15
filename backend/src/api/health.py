"""Health check endpoints."""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import get_db

router = APIRouter()


@router.get("/health")
async def health_check(db: AsyncSession = Depends(get_db)) -> dict:
    """
    Health check endpoint for load balancers and monitoring.

    Checks database connectivity and returns service status.
    """
    checks = {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": {},
    }

    # Database check
    try:
        await db.execute(text("SELECT 1"))
        checks["checks"]["database"] = "ok"
    except Exception as e:
        checks["checks"]["database"] = f"error: {str(e)}"
        checks["status"] = "unhealthy"

    return checks


@router.get("/ready")
async def readiness_check() -> dict:
    """Readiness probe for Kubernetes/Railway."""
    return {"status": "ready"}


@router.get("/scheduler")
async def scheduler_status(db: AsyncSession = Depends(get_db)) -> dict:
    """
    Scheduler health endpoint - shows last run times for all scheduled tasks.

    This queries the database directly since the scheduler runs in a background
    thread and may not share memory with the API process.
    """
    from datetime import timedelta

    checks = {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tasks": {},
        "overdue": [],
    }

    # Expected intervals in minutes (with 2x buffer for "overdue")
    # Note: Some tasks may not produce new records every run (e.g., snapshot only
    # creates records for games within 45 min of tip, team_stats updates daily)
    task_checks = {
        "scoring": {"table": "value_scores", "column": "created_at", "interval": 60},
        "ingest": {"table": "markets", "column": "updated_at", "interval": 60},
        "team_stats": {"table": "team_stats", "column": "created_at", "interval": 1440},  # Daily updates are fine
        "snapshot": {"table": "prediction_snapshots", "column": "snapshot_time", "interval": 1440},  # Only creates before games
        "grading": {"table": "game_results", "column": "created_at", "interval": 1440},  # Depends on game results
    }

    now = datetime.now(timezone.utc)

    for task_name, config in task_checks.items():
        try:
            result = await db.execute(
                text(f"SELECT MAX({config['column']}) FROM {config['table']}")
            )
            last_run = result.scalar()

            if last_run:
                # Handle timezone-naive datetimes
                if last_run.tzinfo is None:
                    last_run = last_run.replace(tzinfo=timezone.utc)

                minutes_ago = (now - last_run).total_seconds() / 60
                is_healthy = minutes_ago < config["interval"]

                checks["tasks"][task_name] = {
                    "last_run": last_run.isoformat(),
                    "minutes_ago": round(minutes_ago, 1),
                    "status": "healthy" if is_healthy else "overdue",
                }

                if not is_healthy:
                    checks["overdue"].append(task_name)
            else:
                checks["tasks"][task_name] = {
                    "last_run": None,
                    "status": "no_data",
                }
        except Exception as e:
            checks["tasks"][task_name] = {
                "status": "error",
                "error": str(e),
            }

    # Overall health
    if checks["overdue"]:
        checks["status"] = "degraded"

    return checks
