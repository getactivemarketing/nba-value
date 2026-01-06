"""Admin API endpoints for manual tasks."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from datetime import datetime, timezone, timedelta
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.tasks.ingestion import (
    _ingest_odds_async,
    _update_nba_stats_async,
    _sync_game_results_async,
)
from src.tasks.scoring import _run_pre_game_scoring_async
from src.tasks.stats_calculation import _calculate_team_stats_async

router = APIRouter(prefix="/admin", tags=["Admin"])


class TaskResult(BaseModel):
    """Task execution result."""
    task_id: str | None = None
    status: str
    message: str


@router.post("/ingest/odds", response_model=TaskResult)
async def trigger_odds_ingestion() -> TaskResult:
    """
    Manually trigger odds ingestion from The Odds API.

    This will fetch current odds for all NBA games and store them
    in the database. Normally runs automatically every 15 minutes.
    """
    try:
        # Run the async function directly
        result = await _ingest_odds_async()
        error_msg = result.get("error", "")
        return TaskResult(
            status=result.get("status", "unknown"),
            message=f"Ingested odds for {result.get('games_fetched', 0)} games, "
                    f"{result.get('snapshots_stored', 0)} snapshots stored. "
                    f"API requests remaining: {result.get('api_requests_remaining', 'unknown')}"
                    + (f" Error: {error_msg}" if error_msg else ""),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest/stats", response_model=TaskResult)
async def trigger_stats_update() -> TaskResult:
    """
    Manually trigger NBA stats update from BALLDONTLIE.

    This will fetch team information and recent game results.
    Normally runs automatically once per day.
    """
    try:
        result = await _update_nba_stats_async()
        error_msg = result.get("error", "")
        return TaskResult(
            status=result.get("status", "unknown"),
            message=f"Updated {result.get('teams_updated', 0)} teams, "
                    f"fetched {result.get('games_fetched', 0)} games."
                    + (f" Error: {error_msg}" if error_msg else ""),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sync/results", response_model=TaskResult)
async def trigger_results_sync() -> TaskResult:
    """
    Manually sync game results for completed games.

    Updates final scores for any games that have finished.
    """
    try:
        result = await _sync_game_results_async()
        return TaskResult(
            status=result.get("status", "unknown"),
            message=f"Updated {result.get('games_updated', 0)} game results.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/scoring/run", response_model=TaskResult)
async def trigger_scoring() -> TaskResult:
    """
    Manually trigger Value Score calculation for all active markets.

    This will:
    1. Find all games starting in the next 24 hours
    2. Calculate Value Scores for each market using both algorithms
    3. Store results in the database
    """
    try:
        result = await _run_pre_game_scoring_async()
        return TaskResult(
            status=result.get("status", "unknown"),
            message=f"Scored {result.get('markets_scored', 0)} markets "
                    f"across {result.get('games_processed', 0)} games. "
                    f"Errors: {result.get('errors', 0)}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stats/calculate", response_model=TaskResult)
async def trigger_stats_calculation() -> TaskResult:
    """
    Calculate rolling team statistics from completed games.

    This will:
    1. Query all completed games this season
    2. Calculate rolling stats for each team (record, win%, net rating, etc.)
    3. Store results in team_stats table
    """
    try:
        result = await _calculate_team_stats_async()
        return TaskResult(
            status=result.get("status", "unknown"),
            message=f"Updated stats for {result.get('teams_updated', 0)} teams. "
                    f"Errors: {result.get('errors', 0)}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/backfill/games")
async def backfill_historical_games(days: int = 30) -> TaskResult:
    """
    Backfill historical games from BallDontLie API for backtesting.

    This will:
    1. Fetch games from the past N days
    2. Store them with final scores in the games table
    3. Create placeholder game_ids matching our format
    """
    from hashlib import md5
    from src.database import async_session_maker
    from src.services.data.balldontlie import BallDontLieClient
    from src.models import Game as GameModel

    try:
        client = BallDontLieClient()
        today = datetime.now(timezone.utc).date()
        start_date = today - timedelta(days=days)

        games_data = await client.get_games(
            start_date=start_date,
            end_date=today,
        )

        games_added = 0
        games_updated = 0

        async with async_session_maker() as session:
            for game in games_data:
                # Only include completed games
                if game.status != "Final" or not game.home_team_score:
                    continue

                # Generate a consistent game_id based on date and teams
                game_key = f"{game.date}_{game.home_team.abbreviation}_{game.away_team.abbreviation}"
                game_id = md5(game_key.encode()).hexdigest()

                # Create tip time from date (assume 7pm ET for historical)
                tip_time = datetime.combine(game.date, datetime.min.time().replace(hour=23))
                tip_time = tip_time.replace(tzinfo=timezone.utc)

                game_stmt = pg_insert(GameModel).values(
                    game_id=game_id,
                    league="NBA",
                    season=2025,
                    game_date=game.date,
                    tip_time_utc=tip_time,
                    home_team_id=game.home_team.abbreviation,
                    away_team_id=game.away_team.abbreviation,
                    home_score=game.home_team_score,
                    away_score=game.away_team_score,
                    status="final",
                ).on_conflict_do_update(
                    index_elements=["game_id"],
                    set_={
                        "home_score": game.home_team_score,
                        "away_score": game.away_team_score,
                        "status": "final",
                        "updated_at": datetime.utcnow(),
                    }
                )

                result = await session.execute(game_stmt)
                if result.rowcount > 0:
                    games_added += 1

            await session.commit()

        return TaskResult(
            status="success",
            message=f"Backfilled {games_added} completed games from the past {days} days.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_system_status() -> dict:
    """Get current system status and stats."""
    from datetime import datetime, timezone
    from sqlalchemy import select, func
    from src.database import async_session_maker
    from src.models import Game, OddsSnapshot, Team, Market, ValueScore, ModelPrediction

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

        # Count markets
        markets_result = await session.execute(select(func.count(Market.market_id)))
        markets_count = markets_result.scalar() or 0

        # Count value scores
        scores_result = await session.execute(select(func.count(ValueScore.value_id)))
        scores_count = scores_result.scalar() or 0

        # Count predictions
        predictions_result = await session.execute(select(func.count(ModelPrediction.prediction_id)))
        predictions_count = predictions_result.scalar() or 0

        # Get latest snapshot time
        latest_result = await session.execute(
            select(OddsSnapshot.snapshot_time)
            .order_by(OddsSnapshot.snapshot_time.desc())
            .limit(1)
        )
        latest_snapshot = latest_result.scalar()

        # Get latest scoring time
        latest_score_result = await session.execute(
            select(ValueScore.calc_time)
            .order_by(ValueScore.calc_time.desc())
            .limit(1)
        )
        latest_score = latest_score_result.scalar()

    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stats": {
            "games": games_count,
            "markets": markets_count,
            "odds_snapshots": snapshots_count,
            "teams": teams_count,
            "value_scores": scores_count,
            "predictions": predictions_count,
            "latest_snapshot": latest_snapshot.isoformat() if latest_snapshot else None,
            "latest_scoring": latest_score.isoformat() if latest_score else None,
        },
    }
