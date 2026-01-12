"""Admin API endpoints for manual tasks."""

import os
from fastapi import APIRouter, HTTPException, Header
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

# Simple API key for cron authentication (set in Railway env vars)
CRON_API_KEY = os.environ.get("CRON_API_KEY", "nba-value-cron-2024")


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


def _verify_cron_key(x_cron_key: str | None) -> bool:
    """Verify the cron API key."""
    if not x_cron_key:
        return False
    return x_cron_key == CRON_API_KEY


@router.post("/cron/run-all")
async def cron_run_all(x_cron_key: str = Header(None)) -> dict:
    """
    Run all scheduled tasks. Called by external cron service.

    This endpoint runs:
    1. Team stats update (from BallDontLie)
    2. Odds ingestion (from The Odds API)
    3. Value score calculation
    4. Prediction snapshots
    5. Results sync and grading

    Requires X-Cron-Key header for authentication.
    """
    if not _verify_cron_key(x_cron_key):
        raise HTTPException(status_code=401, detail="Invalid or missing X-Cron-Key")

    import structlog
    logger = structlog.get_logger()

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tasks": {}
    }

    # 1. Update team stats
    try:
        logger.info("Cron: Running team stats update...")
        from src.tasks.scheduler import update_team_stats
        stats_result = update_team_stats()
        results["tasks"]["team_stats"] = stats_result
    except Exception as e:
        logger.error(f"Cron: Team stats failed: {e}")
        results["tasks"]["team_stats"] = {"status": "error", "error": str(e)}

    # 2. Ingest odds
    try:
        logger.info("Cron: Running odds ingestion...")
        from src.tasks.scheduler import ingest_odds
        odds_result = ingest_odds()
        results["tasks"]["odds_ingestion"] = odds_result
    except Exception as e:
        logger.error(f"Cron: Odds ingestion failed: {e}")
        results["tasks"]["odds_ingestion"] = {"status": "error", "error": str(e)}

    # 3. Run scoring
    try:
        logger.info("Cron: Running scoring pipeline...")
        from src.tasks.scheduler import run_scoring
        scoring_result = run_scoring()
        results["tasks"]["scoring"] = scoring_result
    except Exception as e:
        logger.error(f"Cron: Scoring failed: {e}")
        results["tasks"]["scoring"] = {"status": "error", "error": str(e)}

    # 4. Snapshot predictions
    try:
        logger.info("Cron: Running prediction snapshot...")
        from src.tasks.scheduler import run_snapshot
        snapshot_result = run_snapshot()
        results["tasks"]["snapshot"] = snapshot_result
    except Exception as e:
        logger.error(f"Cron: Snapshot failed: {e}")
        results["tasks"]["snapshot"] = {"status": "error", "error": str(e)}

    # 5. Grade predictions
    try:
        logger.info("Cron: Running prediction grading...")
        from src.tasks.scheduler import run_grading
        grading_result = run_grading()
        results["tasks"]["grading"] = grading_result
    except Exception as e:
        logger.error(f"Cron: Grading failed: {e}")
        results["tasks"]["grading"] = {"status": "error", "error": str(e)}

    # 6. Sync game results
    try:
        logger.info("Cron: Running results sync...")
        from src.tasks.scheduler import run_results_sync
        sync_result = run_results_sync()
        results["tasks"]["results_sync"] = sync_result
    except Exception as e:
        logger.error(f"Cron: Results sync failed: {e}")
        results["tasks"]["results_sync"] = {"status": "error", "error": str(e)}

    # 7. Backfill any missing snapshots
    try:
        logger.info("Cron: Backfilling missing snapshots...")
        backfill_result = await _backfill_missing_snapshots()
        results["tasks"]["backfill_snapshots"] = backfill_result
    except Exception as e:
        logger.error(f"Cron: Backfill failed: {e}")
        results["tasks"]["backfill_snapshots"] = {"status": "error", "error": str(e)}

    results["status"] = "completed"
    logger.info("Cron: All tasks complete", results=results)
    return results


async def _backfill_missing_snapshots() -> dict:
    """Backfill snapshots for completed games that have value_scores but no snapshot."""
    import psycopg2
    import json

    DB_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:wzYHkiAOkykxiPitXKBIqPJxvifFtDPI@maglev.proxy.rlwy.net:46068/railway")

    conn = psycopg2.connect(DB_URL)
    conn.autocommit = True
    cur = conn.cursor()

    # Find completed games with value_scores but no snapshot
    cur.execute("""
        SELECT DISTINCT g.game_id, g.game_date, g.home_team_id, g.away_team_id, g.tip_time_utc,
               gr.home_score, gr.away_score, gr.actual_winner, gr.spread_result, gr.total_result,
               gr.closing_spread, gr.closing_total
        FROM games g
        JOIN markets m ON m.game_id = g.game_id
        JOIN value_scores vs ON vs.market_id = m.market_id
        JOIN game_results gr ON gr.home_team_id = g.home_team_id
                            AND gr.away_team_id = g.away_team_id
                            AND gr.game_date = g.game_date
        LEFT JOIN prediction_snapshots ps ON ps.game_id = g.game_id
        WHERE ps.id IS NULL
        ORDER BY g.game_date DESC
    """)
    games = cur.fetchall()

    if not games:
        cur.close()
        conn.close()
        return {"status": "success", "games_backfilled": 0}

    created = 0
    for game in games:
        game_id, game_date, home_team, away_team, tip_time, h_score, a_score, actual_winner, spread_result, total_result, closing_spread, closing_total = game

        # Get the best value_score for this game
        cur.execute("""
            SELECT m.market_type, m.outcome_label, m.line, m.odds_decimal,
                   vs.algo_a_value_score, vs.algo_a_edge_score, vs.algo_a_confidence,
                   vs.algo_b_value_score, vs.algo_b_combined_edge, vs.algo_b_confidence,
                   mp.p_true, mp.p_market
            FROM value_scores vs
            JOIN markets m ON m.market_id = vs.market_id
            JOIN model_predictions mp ON mp.prediction_id = vs.prediction_id
            WHERE m.game_id = %s
            ORDER BY vs.algo_a_value_score DESC NULLS LAST
        """, (game_id,))
        scores = cur.fetchall()

        if not scores:
            continue

        best = scores[0]
        mtype, outcome, line, odds, algo_a_vs, algo_a_edge, algo_a_conf, algo_b_vs, algo_b_edge, algo_b_conf, p_true, p_market = best

        is_home = 'home' in outcome.lower()
        bet_team = home_team if is_home else away_team

        # Get home probability from moneyline
        home_prob = 0.5
        for s in scores:
            if s[0] == 'moneyline' and 'home' in s[1].lower():
                home_prob = float(s[10]) if s[10] else 0.5
                break

        predicted_winner = home_team if home_prob >= 0.5 else away_team
        winner_correct = predicted_winner == actual_winner

        # Grade the bet
        if mtype == 'spread':
            if is_home:
                bet_result = 'win' if spread_result == 'home_cover' else ('push' if spread_result == 'push' else 'loss')
            else:
                bet_result = 'win' if spread_result == 'away_cover' else ('push' if spread_result == 'push' else 'loss')
        elif mtype == 'moneyline':
            bet_result = 'win' if bet_team == actual_winner else 'loss'
        elif mtype == 'total':
            if 'over' in outcome.lower():
                bet_result = 'win' if total_result == 'over' else ('push' if total_result == 'push' else 'loss')
            else:
                bet_result = 'win' if total_result == 'under' else ('push' if total_result == 'push' else 'loss')
        else:
            bet_result = None

        bet_profit = 90.91 if bet_result == 'win' else (-100 if bet_result == 'loss' else 0)

        cur.execute("""
            INSERT INTO prediction_snapshots (
                game_id, snapshot_time, home_team, away_team, tip_time, game_date,
                predicted_winner, winner_probability, winner_confidence,
                best_bet_type, best_bet_team, best_bet_line, best_bet_value_score, best_bet_odds,
                actual_winner, home_score, away_score, closing_spread, closing_total,
                winner_correct, best_bet_result, best_bet_profit,
                algo_a_value_score, algo_a_edge_score, algo_a_confidence,
                algo_b_value_score, algo_b_combined_edge, algo_b_confidence,
                algo_a_bet_result, algo_b_bet_result, algo_a_profit, algo_b_profit,
                factors
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            game_id, tip_time, home_team, away_team, tip_time, game_date,
            predicted_winner, round(home_prob * 100, 1), 'medium',
            mtype, bet_team, float(line) if line else None, int(algo_a_vs) if algo_a_vs else None, float(odds) if odds else None,
            actual_winner, h_score, a_score, closing_spread, closing_total,
            winner_correct, bet_result, bet_profit,
            int(algo_a_vs) if algo_a_vs else None, float(algo_a_edge) if algo_a_edge else None, float(algo_a_conf) if algo_a_conf else None,
            int(algo_b_vs) if algo_b_vs else None, float(algo_b_edge) if algo_b_edge else None, float(algo_b_conf) if algo_b_conf else None,
            bet_result, bet_result, bet_profit, bet_profit,
            json.dumps(["Backfilled by cron"])
        ))
        created += 1

    cur.close()
    conn.close()
    return {"status": "success", "games_backfilled": created}
