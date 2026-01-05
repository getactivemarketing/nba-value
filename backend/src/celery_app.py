"""Celery application configuration."""

from celery import Celery
from celery.schedules import crontab

from src.config import settings

# Create Celery app
celery_app = Celery(
    "nba_value_betting",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["src.tasks.scoring", "src.tasks.ingestion", "src.tasks.evaluation"],
)

# Celery configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=600,  # 10 minutes max
    worker_prefetch_multiplier=1,
    task_acks_late=True,
)

# Beat schedule for periodic tasks
celery_app.conf.beat_schedule = {
    # Pre-game scoring: every 10 minutes during game hours (10am-11pm ET)
    "pre-game-scoring": {
        "task": "src.tasks.scoring.run_pre_game_scoring",
        "schedule": crontab(minute="*/10", hour="10-23"),
    },
    # Post-game evaluation: nightly at 4am ET (9am UTC)
    "post-game-evaluation": {
        "task": "src.tasks.evaluation.run_post_game_evaluation",
        "schedule": crontab(minute=0, hour=9),
    },
    # Odds ingestion: every 15 minutes
    "odds-ingestion": {
        "task": "src.tasks.ingestion.ingest_odds",
        "schedule": crontab(minute="*/15"),
    },
    # Stats update: daily at 6am ET (11am UTC)
    "stats-update": {
        "task": "src.tasks.ingestion.update_nba_stats",
        "schedule": crontab(minute=0, hour=11),
    },
    # Injury check: every 30 minutes
    "injury-check": {
        "task": "src.tasks.ingestion.check_injuries",
        "schedule": crontab(minute="*/30"),
    },
}
