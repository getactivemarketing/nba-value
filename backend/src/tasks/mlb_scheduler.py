#!/usr/bin/env python3
"""
MLB Scheduler for automated prediction tracking tasks.

This script runs scheduled tasks for MLB:
1. Sync teams (daily)
2. Ingest games from MLB Stats API (every 2 hours)
3. Update team/pitcher stats (every 2 hours)
4. Ingest weather data (every 2 hours)
5. Ingest odds from Odds API (every 30 min)
6. Run scoring/predictions (every 30 min)
7. Snapshot predictions before games (every 15 min)
8. Grade completed predictions (every hour)
9. Sync game results (every 2 hours)

Can be run as a standalone process or scheduled via cron/Railway.
"""

import asyncio
import sys
import time
from datetime import datetime, timezone, timedelta, date

import structlog
import schedule

# Add parent to path for imports
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.database import async_session

logger = structlog.get_logger()

# Track last run times for health monitoring
_last_run_times: dict[str, datetime] = {}


def log_task(message: str, **kwargs):
    """Log scheduler task output."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    extra = " ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
    print(f"[MLB-SCHEDULER] {timestamp} | {message} {extra}", flush=True)


async def sync_teams_async() -> dict:
    """Sync all MLB teams to database."""
    from src.services.mlb.ingest import MLBDataIngestor

    async with async_session() as session:
        ingestor = MLBDataIngestor(session)
        count = await ingestor.sync_teams()

    return {"teams_synced": count, "status": "success"}


def run_sync_teams():
    """Sync wrapper for team sync."""
    log_task("Syncing MLB teams...")
    try:
        result = asyncio.run(sync_teams_async())
        log_task("Teams sync complete", **result)
        _last_run_times['sync_teams'] = datetime.now(timezone.utc)
        return result
    except Exception as e:
        log_task(f"Teams sync FAILED: {e}")
        _last_run_times['sync_teams'] = datetime.now(timezone.utc)
        return {"status": "failed", "error": str(e)}


async def ingest_games_async(days_ahead: int = 7) -> dict:
    """Ingest games from MLB Stats API."""
    from src.services.mlb.ingest import MLBDataIngestor

    async with async_session() as session:
        ingestor = MLBDataIngestor(session)
        count = await ingestor.ingest_games(
            start_date=date.today(),
            end_date=date.today() + timedelta(days=days_ahead),
        )

    return {"games_ingested": count, "status": "success"}


def run_ingest_games():
    """Sync wrapper for game ingestion."""
    log_task("Ingesting MLB games...")
    try:
        result = asyncio.run(ingest_games_async())
        log_task("Games ingestion complete", **result)
        _last_run_times['ingest_games'] = datetime.now(timezone.utc)
        return result
    except Exception as e:
        log_task(f"Games ingestion FAILED: {e}")
        _last_run_times['ingest_games'] = datetime.now(timezone.utc)
        return {"status": "failed", "error": str(e)}


async def update_stats_async() -> dict:
    """Update team and pitcher statistics."""
    from src.services.mlb.ingest import MLBDataIngestor

    async with async_session() as session:
        ingestor = MLBDataIngestor(session)

        team_count = await ingestor.update_team_stats()
        pitcher_count = await ingestor.ingest_pitcher_stats()

    return {
        "teams_updated": team_count,
        "pitchers_updated": pitcher_count,
        "status": "success",
    }


def run_update_stats():
    """Sync wrapper for stats update."""
    log_task("Updating MLB stats...")
    try:
        result = asyncio.run(update_stats_async())
        log_task("Stats update complete", **result)
        _last_run_times['update_stats'] = datetime.now(timezone.utc)
        return result
    except Exception as e:
        log_task(f"Stats update FAILED: {e}")
        _last_run_times['update_stats'] = datetime.now(timezone.utc)
        return {"status": "failed", "error": str(e)}


async def ingest_weather_async() -> dict:
    """Ingest weather data for today's games."""
    from src.services.mlb.ingest import MLBDataIngestor

    async with async_session() as session:
        ingestor = MLBDataIngestor(session)
        count = await ingestor.ingest_weather()

    return {"games_with_weather": count, "status": "success"}


def run_ingest_weather():
    """Sync wrapper for weather ingestion."""
    log_task("Ingesting weather data...")
    try:
        result = asyncio.run(ingest_weather_async())
        log_task("Weather ingestion complete", **result)
        _last_run_times['ingest_weather'] = datetime.now(timezone.utc)
        return result
    except Exception as e:
        log_task(f"Weather ingestion FAILED: {e}")
        _last_run_times['ingest_weather'] = datetime.now(timezone.utc)
        return {"status": "failed", "error": str(e)}


async def ingest_odds_async() -> dict:
    """Ingest odds from The Odds API."""
    from src.services.mlb.ingest import MLBDataIngestor

    async with async_session() as session:
        ingestor = MLBDataIngestor(session)
        count = await ingestor.ingest_odds()

    return {"markets_ingested": count, "status": "success"}


def run_ingest_odds():
    """Sync wrapper for odds ingestion."""
    log_task("Ingesting MLB odds...")
    try:
        result = asyncio.run(ingest_odds_async())
        log_task("Odds ingestion complete", **result)
        _last_run_times['ingest_odds'] = datetime.now(timezone.utc)
        return result
    except Exception as e:
        log_task(f"Odds ingestion FAILED: {e}")
        _last_run_times['ingest_odds'] = datetime.now(timezone.utc)
        return {"status": "failed", "error": str(e)}


async def run_scoring_async() -> dict:
    """Run scoring pipeline for today's games."""
    from src.services.mlb.scorer import run_scoring

    async with async_session() as session:
        predictions = await run_scoring(session)

    return {
        "games_scored": len(predictions),
        "value_bets": sum(1 for p in predictions if p.best_bet and p.best_bet.is_value_bet),
        "status": "success",
    }


def run_scoring():
    """Sync wrapper for scoring."""
    log_task("Running MLB scoring...")
    try:
        result = asyncio.run(run_scoring_async())
        log_task("Scoring complete", **result)
        _last_run_times['scoring'] = datetime.now(timezone.utc)
        return result
    except Exception as e:
        log_task(f"Scoring FAILED: {e}")
        _last_run_times['scoring'] = datetime.now(timezone.utc)
        return {"status": "failed", "error": str(e)}


async def snapshot_predictions_async(hours_ahead: float = 0.75) -> dict:
    """
    Snapshot predictions for games starting soon.

    Captures predictions ~30-45 minutes before first pitch.
    """
    from sqlalchemy import select, and_
    from src.models import MLBGame, MLBPrediction, MLBPredictionSnapshot, MLBPitcher, MLBGameContext
    from src.services.mlb.scorer import MLBScorer

    now = datetime.now(timezone.utc)
    cutoff = now + timedelta(hours=hours_ahead)

    snapshots_created = 0

    async with async_session() as session:
        # Get games starting within the window
        stmt = select(MLBGame).where(
            and_(
                MLBGame.game_time > now,
                MLBGame.game_time < cutoff,
                MLBGame.status == "scheduled",
            )
        )
        result = await session.execute(stmt)
        games = result.scalars().all()

        scorer = MLBScorer(session)

        for game in games:
            # Check if already snapshotted
            existing = await session.execute(
                select(MLBPredictionSnapshot).where(
                    MLBPredictionSnapshot.game_id == game.game_id
                )
            )
            if existing.scalar_one_or_none():
                continue

            try:
                # Generate fresh prediction
                prediction = await scorer.score_game(game)

                # Get pitcher names
                home_starter_name = None
                away_starter_name = None
                home_starter_era = None
                away_starter_era = None

                if game.home_starter_id:
                    pitcher_result = await session.execute(
                        select(MLBPitcher).where(MLBPitcher.pitcher_id == game.home_starter_id)
                    )
                    pitcher = pitcher_result.scalar_one_or_none()
                    if pitcher:
                        home_starter_name = pitcher.player_name
                    if prediction.features:
                        home_starter_era = prediction.features.home_starter_era

                if game.away_starter_id:
                    pitcher_result = await session.execute(
                        select(MLBPitcher).where(MLBPitcher.pitcher_id == game.away_starter_id)
                    )
                    pitcher = pitcher_result.scalar_one_or_none()
                    if pitcher:
                        away_starter_name = pitcher.player_name
                    if prediction.features:
                        away_starter_era = prediction.features.away_starter_era

                # Get context
                context_result = await session.execute(
                    select(MLBGameContext).where(MLBGameContext.game_id == game.game_id)
                )
                context = context_result.scalar_one_or_none()

                # Create snapshot
                snapshot = MLBPredictionSnapshot(
                    game_id=game.game_id,
                    snapshot_time=now,
                    home_team=game.home_team,
                    away_team=game.away_team,
                    game_time=game.game_time,
                    game_date=game.game_date,
                    home_starter_name=home_starter_name,
                    away_starter_name=away_starter_name,
                    home_starter_era=home_starter_era,
                    away_starter_era=away_starter_era,
                    predicted_winner=game.home_team if prediction.p_home_win > 0.5 else game.away_team,
                    winner_probability=max(prediction.p_home_win, prediction.p_away_win),
                    winner_confidence="high" if max(prediction.p_home_win, prediction.p_away_win) > 0.6 else "medium",
                    predicted_run_diff=prediction.predicted_run_diff,
                    predicted_total=prediction.predicted_total,
                    venue_name=context.venue_name if context else None,
                    park_factor=context.park_factor if context else None,
                    temperature=context.temperature if context else None,
                    is_dome=context.is_dome if context else None,
                )

                # Add best bet info
                if prediction.best_ml:
                    snapshot.best_ml_team = prediction.best_ml.team
                    snapshot.best_ml_odds = prediction.best_ml.odds_decimal
                    snapshot.best_ml_value_score = int(prediction.best_ml.value_score)
                    snapshot.best_ml_edge = prediction.best_ml.raw_edge

                if prediction.best_rl:
                    snapshot.best_rl_team = prediction.best_rl.team
                    snapshot.best_rl_line = prediction.best_rl.line
                    snapshot.best_rl_odds = prediction.best_rl.odds_decimal
                    snapshot.best_rl_value_score = int(prediction.best_rl.value_score)
                    snapshot.best_rl_edge = prediction.best_rl.raw_edge

                if prediction.best_total:
                    snapshot.best_total_direction = "over" if "over" in prediction.best_total.bet_type else "under"
                    snapshot.best_total_line = prediction.best_total.line
                    snapshot.best_total_odds = prediction.best_total.odds_decimal
                    snapshot.best_total_value_score = int(prediction.best_total.value_score)
                    snapshot.best_total_edge = prediction.best_total.raw_edge

                if prediction.best_bet:
                    snapshot.best_bet_type = prediction.best_bet.market_type
                    snapshot.best_bet_team = prediction.best_bet.team
                    snapshot.best_bet_line = prediction.best_bet.line
                    snapshot.best_bet_odds = prediction.best_bet.odds_decimal
                    snapshot.best_bet_value_score = int(prediction.best_bet.value_score)
                    snapshot.best_bet_edge = prediction.best_bet.raw_edge

                session.add(snapshot)
                snapshots_created += 1

            except Exception as e:
                logger.warning(f"Failed to snapshot game {game.game_id}: {e}")

        await session.commit()

    return {"snapshots_created": snapshots_created, "games_checked": len(games), "status": "success"}


def run_snapshot():
    """Sync wrapper for prediction snapshot."""
    log_task("Running prediction snapshot...")
    try:
        result = asyncio.run(snapshot_predictions_async())
        log_task("Snapshot complete", **result)
        _last_run_times['snapshot'] = datetime.now(timezone.utc)
        return result
    except Exception as e:
        log_task(f"Snapshot FAILED: {e}")
        _last_run_times['snapshot'] = datetime.now(timezone.utc)
        return {"status": "failed", "error": str(e)}


async def grade_predictions_async() -> dict:
    """Grade completed predictions."""
    from sqlalchemy import select, and_
    from src.models import MLBGame, MLBPredictionSnapshot

    now = datetime.now(timezone.utc)
    graded = 0

    async with async_session() as session:
        # Get ungraded snapshots for completed games
        stmt = select(MLBPredictionSnapshot).where(
            and_(
                MLBPredictionSnapshot.actual_winner.is_(None),
                MLBPredictionSnapshot.game_time < now,
            )
        )
        result = await session.execute(stmt)
        snapshots = result.scalars().all()

        for snapshot in snapshots:
            # Get game result
            game_result = await session.execute(
                select(MLBGame).where(
                    and_(
                        MLBGame.game_id == snapshot.game_id,
                        MLBGame.status == "final",
                    )
                )
            )
            game = game_result.scalar_one_or_none()

            if not game or game.home_score is None:
                continue

            # Fill in results
            snapshot.home_score = game.home_score
            snapshot.away_score = game.away_score
            snapshot.actual_winner = game.home_team if game.home_score > game.away_score else game.away_team

            # Grade winner prediction
            snapshot.winner_correct = snapshot.predicted_winner == snapshot.actual_winner

            # Grade best ML bet
            if snapshot.best_ml_team:
                ml_won = snapshot.best_ml_team == snapshot.actual_winner
                snapshot.best_ml_result = "win" if ml_won else "loss"
                if ml_won and snapshot.best_ml_odds:
                    snapshot.best_ml_profit = (float(snapshot.best_ml_odds) - 1) * 100
                else:
                    snapshot.best_ml_profit = -100

            # Grade best runline bet
            if snapshot.best_rl_team and snapshot.best_rl_line:
                run_diff = game.home_score - game.away_score
                if snapshot.best_rl_team == game.home_team:
                    # Home team needs to cover (win by more than spread)
                    rl_won = run_diff > abs(float(snapshot.best_rl_line))
                else:
                    # Away team needs to cover
                    rl_won = -run_diff > abs(float(snapshot.best_rl_line)) or \
                             run_diff < -abs(float(snapshot.best_rl_line))

                if rl_won:
                    snapshot.best_rl_result = "win"
                    if snapshot.best_rl_odds:
                        snapshot.best_rl_profit = (float(snapshot.best_rl_odds) - 1) * 100
                else:
                    snapshot.best_rl_result = "loss"
                    snapshot.best_rl_profit = -100

            # Grade best total bet
            if snapshot.best_total_direction and snapshot.best_total_line:
                total_runs = game.home_score + game.away_score
                line = float(snapshot.best_total_line)

                if snapshot.best_total_direction == "over":
                    if total_runs > line:
                        snapshot.best_total_result = "win"
                        if snapshot.best_total_odds:
                            snapshot.best_total_profit = (float(snapshot.best_total_odds) - 1) * 100
                    elif total_runs < line:
                        snapshot.best_total_result = "loss"
                        snapshot.best_total_profit = -100
                    else:
                        snapshot.best_total_result = "push"
                        snapshot.best_total_profit = 0
                else:  # under
                    if total_runs < line:
                        snapshot.best_total_result = "win"
                        if snapshot.best_total_odds:
                            snapshot.best_total_profit = (float(snapshot.best_total_odds) - 1) * 100
                    elif total_runs > line:
                        snapshot.best_total_result = "loss"
                        snapshot.best_total_profit = -100
                    else:
                        snapshot.best_total_result = "push"
                        snapshot.best_total_profit = 0

            # Grade overall best bet
            if snapshot.best_bet_type:
                if snapshot.best_bet_type == "moneyline":
                    snapshot.best_bet_result = snapshot.best_ml_result
                    snapshot.best_bet_profit = snapshot.best_ml_profit
                elif snapshot.best_bet_type == "runline":
                    snapshot.best_bet_result = snapshot.best_rl_result
                    snapshot.best_bet_profit = snapshot.best_rl_profit
                elif snapshot.best_bet_type == "total":
                    snapshot.best_bet_result = snapshot.best_total_result
                    snapshot.best_bet_profit = snapshot.best_total_profit

            graded += 1

        await session.commit()

    return {"predictions_graded": graded, "status": "success"}


def run_grading():
    """Sync wrapper for grading."""
    log_task("Running prediction grading...")
    try:
        result = asyncio.run(grade_predictions_async())
        log_task("Grading complete", **result)
        _last_run_times['grading'] = datetime.now(timezone.utc)
        return result
    except Exception as e:
        log_task(f"Grading FAILED: {e}")
        _last_run_times['grading'] = datetime.now(timezone.utc)
        return {"status": "failed", "error": str(e)}


async def sync_results_async() -> dict:
    """Sync final scores for completed games."""
    from src.services.mlb.ingest import MLBDataIngestor

    async with async_session() as session:
        ingestor = MLBDataIngestor(session)
        count = await ingestor.sync_results()

    return {"games_synced": count, "status": "success"}


def run_sync_results():
    """Sync wrapper for results sync."""
    log_task("Syncing game results...")
    try:
        result = asyncio.run(sync_results_async())
        log_task("Results sync complete", **result)
        _last_run_times['sync_results'] = datetime.now(timezone.utc)

        # Auto-grade after syncing results
        if result.get("games_synced", 0) > 0:
            log_task("New results found, running grading...")
            grade_result = run_grading()
            result['grading'] = grade_result

        return result
    except Exception as e:
        log_task(f"Results sync FAILED: {e}")
        _last_run_times['sync_results'] = datetime.now(timezone.utc)
        return {"status": "failed", "error": str(e)}


def run_all():
    """Run all MLB tasks once."""
    log_task("=" * 50)
    log_task("Running all MLB scheduled tasks...")

    run_sync_teams()
    time.sleep(2)

    run_ingest_games()
    time.sleep(2)

    run_update_stats()
    time.sleep(2)

    run_ingest_weather()
    time.sleep(2)

    run_ingest_odds()
    time.sleep(2)

    run_scoring()
    time.sleep(2)

    run_snapshot()
    time.sleep(2)

    run_grading()
    time.sleep(2)

    run_sync_results()

    log_task("All MLB tasks complete")
    log_task("=" * 50)


def get_scheduler_status() -> dict:
    """Get current scheduler status for health monitoring."""
    now = datetime.now(timezone.utc)

    status = {
        "running": True,
        "last_run_times": {},
        "task_health": {},
    }

    # Expected intervals in minutes
    expected_intervals = {
        "sync_teams": 1440,  # 24 hours
        "ingest_games": 120,  # 2 hours
        "update_stats": 120,
        "ingest_weather": 120,
        "ingest_odds": 30,
        "scoring": 30,
        "snapshot": 15,
        "grading": 60,
        "sync_results": 120,
    }

    for task, last_run in _last_run_times.items():
        minutes_ago = (now - last_run).total_seconds() / 60
        status["last_run_times"][task] = {
            "last_run": last_run.isoformat(),
            "minutes_ago": round(minutes_ago, 1),
        }

        expected = expected_intervals.get(task, 60)
        is_healthy = minutes_ago < expected * 2
        status["task_health"][task] = "healthy" if is_healthy else "overdue"

    overdue_tasks = [t for t, h in status["task_health"].items() if h == "overdue"]
    status["healthy"] = len(overdue_tasks) == 0
    status["overdue_tasks"] = overdue_tasks

    return status


def run_health_check():
    """Check scheduler health and log warnings."""
    status = get_scheduler_status()

    if status["overdue_tasks"]:
        log_task(f"WARNING: Overdue tasks: {status['overdue_tasks']}")
    else:
        log_task("Health check: All MLB tasks running normally")


def start_scheduler():
    """Start the MLB scheduler loop."""
    log_task("Starting MLB prediction tracker scheduler...")

    # Run all tasks immediately on startup
    run_all()

    # Schedule recurring tasks
    schedule.every().day.at("06:00").do(run_sync_teams)  # Daily team sync
    schedule.every(2).hours.do(run_ingest_games)
    schedule.every(2).hours.do(run_update_stats)
    schedule.every(2).hours.do(run_ingest_weather)
    schedule.every(30).minutes.do(run_ingest_odds)
    schedule.every(30).minutes.do(run_scoring)
    schedule.every(15).minutes.do(run_snapshot)
    schedule.every(1).hour.do(run_grading)
    schedule.every(2).hours.do(run_sync_results)
    schedule.every(1).hour.do(run_health_check)

    log_task("MLB Scheduler configured:")
    log_task("  - Team sync: daily at 6:00 AM")
    log_task("  - Game ingestion: every 2 hours")
    log_task("  - Stats update: every 2 hours")
    log_task("  - Weather ingestion: every 2 hours")
    log_task("  - Odds ingestion: every 30 minutes")
    log_task("  - Scoring: every 30 minutes")
    log_task("  - Prediction snapshot: every 15 minutes")
    log_task("  - Grading: every 1 hour")
    log_task("  - Results sync: every 2 hours")
    log_task("  - Health monitoring: every 1 hour")

    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == 'teams':
            run_sync_teams()
        elif command == 'games':
            run_ingest_games()
        elif command == 'stats':
            run_update_stats()
        elif command == 'weather':
            run_ingest_weather()
        elif command == 'odds':
            run_ingest_odds()
        elif command == 'score':
            run_scoring()
        elif command == 'snapshot':
            run_snapshot()
        elif command == 'grade':
            run_grading()
        elif command == 'results':
            run_sync_results()
        elif command == 'all':
            run_all()
        elif command == 'daemon':
            start_scheduler()
        else:
            print(f"Unknown command: {command}")
            print("Usage: python mlb_scheduler.py [teams|games|stats|weather|odds|score|snapshot|grade|results|all|daemon]")
    else:
        run_all()
