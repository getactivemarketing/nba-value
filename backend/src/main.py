"""FastAPI application entry point."""

import os
import threading
import time
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.database import init_db
from src.api import health, markets, bets, evaluation, admin, trends, backtest, mlb

# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
)

logger = structlog.get_logger()

# Global scheduler thread references
_scheduler_thread = None
_mlb_scheduler_thread = None
_scheduler_should_run = False  # Flag to control watchdog lifecycle


def _run_scheduler():
    """Run the NBA scheduler in a background thread."""
    try:
        print("[SCHEDULER] Importing scheduler module...", flush=True)
        from src.tasks.scheduler import start_scheduler
        print("[SCHEDULER] Starting scheduler daemon...", flush=True)
        start_scheduler()
    except Exception as e:
        print(f"[SCHEDULER] CRASHED: {e}", flush=True)
        logger.error(f"Scheduler thread crashed: {e}")


def _run_mlb_scheduler():
    """Run the MLB scheduler in a background thread."""
    try:
        print("[MLB-SCHEDULER] Importing MLB scheduler module...", flush=True)
        from src.tasks.mlb_scheduler import start_scheduler as start_mlb_scheduler
        print("[MLB-SCHEDULER] Starting MLB scheduler daemon...", flush=True)
        start_mlb_scheduler()
    except Exception as e:
        print(f"[MLB-SCHEDULER] CRASHED: {e}", flush=True)
        logger.error(f"MLB scheduler thread crashed: {e}")


def _start_scheduler_thread():
    """Create and start a new NBA scheduler thread. Returns the thread."""
    thread = threading.Thread(target=_run_scheduler, daemon=True, name="scheduler")
    thread.start()
    return thread


def _start_mlb_scheduler_thread():
    """Create and start a new MLB scheduler thread. Returns the thread."""
    thread = threading.Thread(target=_run_mlb_scheduler, daemon=True, name="mlb-scheduler")
    thread.start()
    return thread


def _run_social_scheduler():
    """Run the social media scheduler in a background thread."""
    try:
        print("[SOCIAL-SCHEDULER] Importing social scheduler module...", flush=True)
        from src.tasks.social_scheduler import start_scheduler as start_social_scheduler
        print("[SOCIAL-SCHEDULER] Starting social scheduler daemon...", flush=True)
        start_social_scheduler()
    except Exception as e:
        print(f"[SOCIAL-SCHEDULER] CRASHED: {e}", flush=True)
        logger.error(f"Social scheduler thread crashed: {e}")


def _start_social_scheduler_thread():
    """Create and start social scheduler thread. Returns the thread."""
    thread = threading.Thread(target=_run_social_scheduler, daemon=True, name="social-scheduler")
    thread.start()
    return thread


def _scheduler_watchdog():
    """Monitor scheduler threads and restart them if they die."""
    global _scheduler_thread, _mlb_scheduler_thread
    restart_count = 0
    mlb_restart_count = 0
    max_restarts = 10  # Prevent infinite restart loops

    while _scheduler_should_run:
        time.sleep(60)  # Check every 60 seconds

        if not _scheduler_should_run:
            break

        # Watch NBA scheduler
        if _scheduler_thread is None or not _scheduler_thread.is_alive():
            restart_count += 1
            if restart_count > max_restarts:
                print(f"[SCHEDULER-WATCHDOG] Max restarts ({max_restarts}) reached for NBA scheduler. Giving up.", flush=True)
                logger.error(f"Scheduler watchdog: exceeded {max_restarts} restarts for NBA, stopping")
            else:
                print(f"[SCHEDULER-WATCHDOG] NBA scheduler thread died! Restarting (attempt {restart_count}/{max_restarts})...", flush=True)
                logger.warning(f"Scheduler watchdog restarting NBA thread (attempt {restart_count})")
                backoff = min(5 * (2 ** (restart_count - 1)), 300)
                time.sleep(backoff)
                _scheduler_thread = _start_scheduler_thread()
                print(f"[SCHEDULER-WATCHDOG] NBA scheduler thread restarted successfully", flush=True)
        else:
            restart_count = 0

        # Watch MLB scheduler
        if _mlb_scheduler_thread is None or not _mlb_scheduler_thread.is_alive():
            mlb_restart_count += 1
            if mlb_restart_count > max_restarts:
                print(f"[SCHEDULER-WATCHDOG] Max restarts ({max_restarts}) reached for MLB scheduler. Giving up.", flush=True)
                logger.error(f"Scheduler watchdog: exceeded {max_restarts} restarts for MLB, stopping")
            else:
                print(f"[SCHEDULER-WATCHDOG] MLB scheduler thread died! Restarting (attempt {mlb_restart_count}/{max_restarts})...", flush=True)
                logger.warning(f"Scheduler watchdog restarting MLB thread (attempt {mlb_restart_count})")
                backoff = min(5 * (2 ** (mlb_restart_count - 1)), 300)
                time.sleep(backoff)
                _mlb_scheduler_thread = _start_mlb_scheduler_thread()
                print(f"[SCHEDULER-WATCHDOG] MLB scheduler thread restarted successfully", flush=True)
        else:
            mlb_restart_count = 0


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan manager."""
    global _scheduler_thread, _mlb_scheduler_thread, _scheduler_should_run

    # Startup
    logger.info("Starting NBA Value Betting API", environment=settings.environment)
    await init_db()

    # Start schedulers in background threads if DATABASE_URL is set (Railway deployment)
    # Note: Railway uses postgres:// format, SQLAlchemy accepts postgresql://
    db_url = os.environ.get("DATABASE_URL", "")
    should_start_scheduler = bool(db_url) and ("postgresql://" in db_url or "postgres://" in db_url)
    print(f"[SCHEDULER] DATABASE_URL present: {bool(db_url)}, will_start: {should_start_scheduler}", flush=True)
    if should_start_scheduler:
        _scheduler_should_run = True

        _scheduler_thread = _start_scheduler_thread()
        print("[SCHEDULER] NBA background thread started", flush=True)

        _mlb_scheduler_thread = _start_mlb_scheduler_thread()
        print("[MLB-SCHEDULER] MLB background thread started", flush=True)

        _start_social_scheduler_thread()
        print("[SOCIAL-SCHEDULER] Social media thread started", flush=True)

        # Start watchdog to auto-restart schedulers if they crash
        watchdog_thread = threading.Thread(target=_scheduler_watchdog, daemon=True, name="scheduler-watchdog")
        watchdog_thread.start()
        print("[SCHEDULER-WATCHDOG] Watchdog thread started", flush=True)

    yield

    # Shutdown
    _scheduler_should_run = False
    logger.info("Shutting down NBA Value Betting API")


app = FastAPI(
    title="NBA Value Betting API",
    description="Sharp-focused betting intelligence platform with Value Score ranking",
    version="0.2.0",  # Phase 2: Model Layer
    lifespan=lifespan,
    docs_url="/docs" if not settings.is_production else None,
    redoc_url="/redoc" if not settings.is_production else None,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(markets.router, prefix=settings.api_v1_prefix, tags=["Markets"])
app.include_router(bets.router, prefix=settings.api_v1_prefix, tags=["Bets"])
app.include_router(evaluation.router, prefix=settings.api_v1_prefix, tags=["Evaluation"])
app.include_router(admin.router, prefix=settings.api_v1_prefix, tags=["Admin"])
app.include_router(trends.router, prefix=settings.api_v1_prefix, tags=["Trends"])
app.include_router(backtest.router, prefix=settings.api_v1_prefix, tags=["Backtest"])
app.include_router(mlb.router, prefix=settings.api_v1_prefix, tags=["MLB"])


@app.get("/")
async def root() -> dict:
    """Root endpoint."""
    return {
        "name": "TruLine API",
        "version": "0.4.0",
        "build": "2026-04-07-v4",
        "docs": "/docs",
    }


