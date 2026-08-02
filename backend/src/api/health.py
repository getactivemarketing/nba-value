"""Health check endpoints."""

from datetime import datetime, timezone
from functools import lru_cache

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

    # Model check. Added 2026-08-02 because nothing in the system could say
    # which model was serving or how old it was — the run-diff model ran six
    # months stale without a word, and a missing artifact silently falls back
    # to a hand-written heuristic while picks keep flowing and look normal.
    #
    # Reported as "degraded", never "unhealthy": a stale model is a real
    # problem but the service is still up, and failing the healthcheck here
    # would take production down over a monitoring signal.
    try:
        checks["checks"]["models"] = _model_check()
        if checks["checks"]["models"]["status"] == "degraded" and checks["status"] == "healthy":
            checks["status"] = "degraded"
    except Exception as e:
        checks["checks"]["models"] = {"status": "unknown", "error": str(e)}

    return checks


@lru_cache(maxsize=1)
def _model_metadata() -> tuple[dict | None, dict | None, str, str]:
    """Read artifact metadata ONCE per process.

    Cached because the first version called joblib.load() on every /health
    request — synchronous disk I/O and unpickling on the exact endpoint
    Railway polls during startup, when the event loop is already contended by
    three scheduler threads. That is a self-inflicted healthcheck failure.

    Model files are baked into the image and cannot change within a process,
    so caching costs nothing in accuracy. Age is recomputed per request from
    the cached trained_at, so staleness still advances with the clock.
    """
    import joblib
    from pathlib import Path

    from src.config import settings

    def _meta(path: str) -> dict | None:
        p = Path(path)
        if not p.exists():
            return None
        art = joblib.load(p)
        # Drop the estimator; only metadata is needed and it keeps this small.
        return {k: v for k, v in art.items() if k != "model"}

    run_diff_path = settings.mlb_run_diff_model_path
    totals_path = settings.mlb_totals_model_path
    return _meta(run_diff_path), _meta(totals_path), run_diff_path, totals_path


def _model_check() -> dict:
    """Model status from cached metadata; no disk I/O on the request path."""
    from src.services.mlb.model_status import summarize_models

    run_diff, totals, run_diff_path, totals_path = _model_metadata()
    return summarize_models(
        run_diff=run_diff,
        totals=totals,
        run_diff_path=run_diff_path,
        totals_path=totals_path,
    )


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
