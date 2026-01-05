"""Scoring tasks for pre-game Value Score calculation."""

import structlog
from src.celery_app import celery_app

logger = structlog.get_logger()


@celery_app.task(name="src.tasks.scoring.run_pre_game_scoring")
def run_pre_game_scoring() -> dict:
    """
    Calculate Value Scores for all active markets.

    This task runs every 10 minutes during game hours and:
    1. Fetches all markets with games starting in next 24 hours
    2. Gets latest model predictions
    3. Calculates confidence and market quality factors
    4. Computes Value Scores for both algorithms
    5. Stores results in value_scores table
    """
    logger.info("Starting pre-game scoring task")

    # TODO: Implement actual scoring logic
    # 1. Query active markets
    # 2. Get latest predictions
    # 3. Calculate confidence factors
    # 4. Calculate market quality factors
    # 5. Run both scoring algorithms
    # 6. Store results

    result = {
        "markets_scored": 0,
        "errors": 0,
        "status": "placeholder",
    }

    logger.info("Completed pre-game scoring", **result)
    return result


@celery_app.task(name="src.tasks.scoring.score_single_market")
def score_single_market(market_id: str) -> dict:
    """
    Calculate Value Score for a single market.

    Useful for on-demand scoring when odds change significantly.
    """
    logger.info("Scoring single market", market_id=market_id)

    # TODO: Implement single market scoring

    return {"market_id": market_id, "status": "placeholder"}
