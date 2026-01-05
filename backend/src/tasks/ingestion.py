"""Data ingestion tasks."""

import structlog
from src.celery_app import celery_app

logger = structlog.get_logger()


@celery_app.task(name="src.tasks.ingestion.ingest_odds")
def ingest_odds() -> dict:
    """
    Ingest current odds from The Odds API.

    Runs every 15 minutes and:
    1. Fetches NBA odds for all active games
    2. Stores snapshots in odds_snapshots table
    3. Updates current odds in markets table
    4. Triggers re-scoring if odds changed significantly
    """
    logger.info("Starting odds ingestion")

    # TODO: Implement actual odds ingestion
    # 1. Call The Odds API
    # 2. Parse response
    # 3. Store snapshots
    # 4. Update markets
    # 5. Check for significant changes

    result = {
        "games_fetched": 0,
        "markets_updated": 0,
        "api_requests_remaining": None,
        "status": "placeholder",
    }

    logger.info("Completed odds ingestion", **result)
    return result


@celery_app.task(name="src.tasks.ingestion.update_nba_stats")
def update_nba_stats() -> dict:
    """
    Update NBA team and player statistics.

    Runs daily and:
    1. Fetches latest team stats from nba_api
    2. Calculates rolling metrics (ORtg, DRtg, pace)
    3. Updates team_stats table
    4. Fetches player stats for injury impact calculation
    """
    logger.info("Starting NBA stats update")

    # TODO: Implement stats update
    # 1. Call nba_api endpoints
    # 2. Calculate rolling averages
    # 3. Store in database

    result = {
        "teams_updated": 0,
        "players_updated": 0,
        "status": "placeholder",
    }

    logger.info("Completed NBA stats update", **result)
    return result


@celery_app.task(name="src.tasks.ingestion.check_injuries")
def check_injuries() -> dict:
    """
    Check for injury updates.

    Runs every 30 minutes and:
    1. Fetches latest injury report
    2. Compares to previous state
    3. Updates injuries table
    4. Triggers re-scoring for affected games
    """
    logger.info("Starting injury check")

    # TODO: Implement injury checking
    # 1. Call injury API/scraper
    # 2. Parse injury statuses
    # 3. Update database
    # 4. Identify changed games

    result = {
        "injuries_found": 0,
        "changes_detected": 0,
        "status": "placeholder",
    }

    logger.info("Completed injury check", **result)
    return result


@celery_app.task(name="src.tasks.ingestion.backfill_historical_odds")
def backfill_historical_odds(start_date: str, end_date: str) -> dict:
    """
    Backfill historical odds data for backtesting.

    Manual task for loading historical data.
    """
    logger.info("Starting historical odds backfill", start_date=start_date, end_date=end_date)

    # TODO: Implement historical backfill

    return {"status": "placeholder", "start_date": start_date, "end_date": end_date}
