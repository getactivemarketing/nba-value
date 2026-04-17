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

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from src.config import settings

logger = structlog.get_logger()

# Track last run times for health monitoring
_last_run_times: dict[str, datetime] = {}

# Lazy-initialized engine for the MLB scheduler thread.
# Created inside start_scheduler() to avoid blocking app startup.
# Sharing the main engine causes "Future attached to a different loop" errors
# because asyncpg connections are bound to the event loop that created them.
_mlb_engine = None
_mlb_session_factory = None


def _init_engine():
    """Initialize the MLB scheduler's DB engine. Called once from start_scheduler()."""
    global _mlb_engine, _mlb_session_factory
    if _mlb_engine is None:
        _mlb_engine = create_async_engine(
            settings.async_database_url,
            pool_pre_ping=True,
            pool_size=2,
            max_overflow=1,
            pool_recycle=180,
            pool_timeout=30,
        )
        _mlb_session_factory = async_sessionmaker(
            _mlb_engine, class_=AsyncSession, expire_on_commit=False
        )


def mlb_session():
    """Get an async session from the MLB scheduler's dedicated engine."""
    if _mlb_session_factory is None:
        _init_engine()
    return _mlb_session_factory()


def log_task(message: str, **kwargs):
    """Log scheduler task output."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    extra = " ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
    print(f"[MLB-SCHEDULER] {timestamp} | {message} {extra}", flush=True)


def _today_et() -> date:
    """Get today's date in US Eastern time.

    MLB game_date is stored as the ET date (officialDate from MLB API).
    Railway runs in UTC, so date.today() returns the wrong date after 7pm ET.
    """
    eastern = timedelta(hours=-5)  # EDT approximation (close enough for game dates)
    return (datetime.now(timezone.utc) + eastern).date()


async def sync_teams_async() -> dict:
    """Sync all MLB teams to database."""
    from src.services.mlb.ingest import MLBDataIngestor

    async with mlb_session() as session:
        ingestor = MLBDataIngestor(session)
        count = await ingestor.sync_teams()

    return {"teams_synced": count, "status": "success"}


def run_sync_teams():
    """Sync wrapper for team sync."""
    log_task("Syncing MLB teams...")
    try:
        result = _run_async(sync_teams_async())
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

    async with mlb_session() as session:
        ingestor = MLBDataIngestor(session)
        today = _today_et()
        count = await ingestor.ingest_games(
            start_date=today,
            end_date=today + timedelta(days=days_ahead),
        )

    return {"games_ingested": count, "status": "success"}


def run_ingest_games():
    """Sync wrapper for game ingestion."""
    log_task("Ingesting MLB games...")
    try:
        result = _run_async(ingest_games_async())
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

    async with mlb_session() as session:
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
        result = _run_async(update_stats_async())
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

    async with mlb_session() as session:
        ingestor = MLBDataIngestor(session)
        count = await ingestor.ingest_weather()

    return {"games_with_weather": count, "status": "success"}


def run_ingest_weather():
    """Sync wrapper for weather ingestion."""
    log_task("Ingesting weather data...")
    try:
        result = _run_async(ingest_weather_async())
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

    async with mlb_session() as session:
        ingestor = MLBDataIngestor(session)
        count = await ingestor.ingest_odds()

    return {"markets_ingested": count, "status": "success"}


def run_ingest_odds():
    """Sync wrapper for odds ingestion."""
    log_task("Ingesting MLB odds...")
    try:
        result = _run_async(ingest_odds_async())
        log_task("Odds ingestion complete", **result)
        _last_run_times['ingest_odds'] = datetime.now(timezone.utc)
        return result
    except Exception as e:
        log_task(f"Odds ingestion FAILED: {e}")
        _last_run_times['ingest_odds'] = datetime.now(timezone.utc)
        return {"status": "failed", "error": str(e)}


async def run_scoring_async() -> dict:
    """Run scoring pipeline for today's and tomorrow's games (ET)."""
    from src.services.mlb.scorer import run_scoring

    today = _today_et()
    tomorrow = today + timedelta(days=1)
    all_predictions = []

    for game_date in [today, tomorrow]:
        async with mlb_session() as session:
            predictions = await run_scoring(session, game_date=game_date)
            all_predictions.extend(predictions)

    return {
        "games_scored": len(all_predictions),
        "value_bets": sum(1 for p in all_predictions if p.best_bet and p.best_bet.is_value_bet),
        "dates_scored": f"{today},{tomorrow}",
        "status": "success",
    }


def run_scoring():
    """Sync wrapper for scoring."""
    log_task("Running MLB scoring...")
    try:
        result = _run_async(run_scoring_async())
        log_task("Scoring complete", **result)
        _last_run_times['scoring'] = datetime.now(timezone.utc)
        return result
    except Exception as e:
        log_task(f"Scoring FAILED: {e}")
        _last_run_times['scoring'] = datetime.now(timezone.utc)
        return {"status": "failed", "error": str(e)}


async def snapshot_predictions_async(hours_ahead: float = 1.0) -> dict:
    """
    Snapshot predictions for games starting within the next hour.

    Widened from 45min to 60min to ensure we don't miss games.
    """
    from sqlalchemy import select, and_
    from src.models import MLBGame, MLBPrediction, MLBPredictionSnapshot, MLBPitcher, MLBGameContext
    from src.services.mlb.scorer import MLBScorer

    now = datetime.now(timezone.utc)
    cutoff = now + timedelta(hours=hours_ahead)

    snapshots_created = 0

    async with mlb_session() as session:
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
                await session.flush()
                snapshots_created += 1

            except Exception as e:
                logger.warning(f"Failed to snapshot game {game.game_id}: {e}")
                await session.rollback()

        await session.commit()

    return {"snapshots_created": snapshots_created, "games_checked": len(games), "status": "success"}


def run_snapshot():
    """Sync wrapper for prediction snapshot. Runs scoring first to ensure predictions exist."""
    # Score games first so predictions are fresh before snapshotting
    run_scoring()

    log_task("Running prediction snapshot...")
    try:
        result = _run_async(snapshot_predictions_async())
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

    async with mlb_session() as session:
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

            # Grade best runline bet.
            # best_rl_line is from the BET TEAM's perspective: +1.5 for underdogs,
            # -1.5 for favorites. The sign already encodes the direction — add it
            # to the bet team's score and compare to the opponent's score.
            if snapshot.best_rl_team and snapshot.best_rl_line:
                if snapshot.best_rl_team == game.home_team:
                    bet_score = game.home_score
                    opp_score = game.away_score
                else:
                    bet_score = game.away_score
                    opp_score = game.home_score

                adjusted = bet_score + float(snapshot.best_rl_line)
                if adjusted > opp_score:
                    snapshot.best_rl_result = "win"
                    if snapshot.best_rl_odds:
                        snapshot.best_rl_profit = (float(snapshot.best_rl_odds) - 1) * 100
                    else:
                        snapshot.best_rl_profit = 0
                elif adjusted == opp_score:
                    snapshot.best_rl_result = "push"
                    snapshot.best_rl_profit = 0
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


async def update_betting_records_async() -> dict:
    """Compute ATS and O/U records from graded MLB prediction snapshots.

    Counts runline wins/losses for ATS record and total bet overs/unders
    for O/U record, then updates the latest mlb_team_stats row for each team.
    """
    from sqlalchemy import select, func, and_, case
    from src.models import MLBPredictionSnapshot, MLBTeamStats

    updated = 0

    async with mlb_session() as session:
        # Get all teams that have graded snapshots
        teams_result = await session.execute(
            select(MLBPredictionSnapshot.home_team).distinct()
        )
        all_teams = set()
        for row in teams_result.fetchall():
            all_teams.add(row[0])
        teams_result2 = await session.execute(
            select(MLBPredictionSnapshot.away_team).distinct()
        )
        for row in teams_result2.fetchall():
            all_teams.add(row[0])

        for team in all_teams:
            # ATS: count runline results where this team was the bet team
            rl_result = await session.execute(
                select(
                    func.count(case((MLBPredictionSnapshot.best_rl_result == "win", 1))).label("ats_w"),
                    func.count(case((MLBPredictionSnapshot.best_rl_result == "loss", 1))).label("ats_l"),
                    func.count(case((MLBPredictionSnapshot.best_rl_result == "push", 1))).label("ats_p"),
                ).where(
                    and_(
                        MLBPredictionSnapshot.best_rl_team == team,
                        MLBPredictionSnapshot.best_rl_result.isnot(None),
                    )
                )
            )
            rl_row = rl_result.first()

            # O/U: count total results for games this team played in
            ou_result = await session.execute(
                select(
                    func.count(case((MLBPredictionSnapshot.best_total_result == "win", 1))).label("ou_w"),
                    func.count(case((MLBPredictionSnapshot.best_total_result == "loss", 1))).label("ou_l"),
                    func.count(case((MLBPredictionSnapshot.best_total_result == "push", 1))).label("ou_p"),
                ).where(
                    and_(
                        MLBPredictionSnapshot.best_total_result.isnot(None),
                        (MLBPredictionSnapshot.home_team == team) | (MLBPredictionSnapshot.away_team == team),
                    )
                )
            )
            ou_row = ou_result.first()

            if not rl_row and not ou_row:
                continue

            # Update latest team_stats row
            latest = await session.execute(
                select(MLBTeamStats).where(
                    MLBTeamStats.team_abbr == team
                ).order_by(MLBTeamStats.stat_date.desc()).limit(1)
            )
            stats = latest.scalar_one_or_none()
            if not stats:
                continue

            if rl_row:
                stats.ats_wins = rl_row.ats_w or 0
                stats.ats_losses = rl_row.ats_l or 0
                stats.ats_pushes = rl_row.ats_p or 0
            if ou_row:
                stats.ou_overs = ou_row.ou_w or 0
                stats.ou_unders = ou_row.ou_l or 0
                stats.ou_pushes = ou_row.ou_p or 0

            updated += 1

        await session.commit()

    return {"teams_updated": updated, "status": "success"}


def run_grading():
    """Sync wrapper for grading."""
    log_task("Running prediction grading...")
    try:
        result = _run_async(grade_predictions_async())
        log_task("Grading complete", **result)

        # Update ATS/O-U records after grading
        try:
            br_result = _run_async(update_betting_records_async())
            log_task("Betting records updated", **br_result)
        except Exception as e:
            log_task(f"Betting records update FAILED (non-fatal): {e}")

        _last_run_times['grading'] = datetime.now(timezone.utc)
        return result
    except Exception as e:
        log_task(f"Grading FAILED: {e}")
        _last_run_times['grading'] = datetime.now(timezone.utc)
        return {"status": "failed", "error": str(e)}


async def sync_results_async() -> dict:
    """Sync final scores for today and yesterday (ET)."""
    from src.services.mlb.ingest import MLBDataIngestor

    today = _today_et()
    yesterday = today - timedelta(days=1)
    total = 0

    for game_date in [today, yesterday]:
        async with mlb_session() as session:
            ingestor = MLBDataIngestor(session)
            count = await ingestor.sync_results(game_date=game_date)
            total += count

    return {"games_synced": total, "status": "success"}


def run_sync_results():
    """Sync wrapper for results sync."""
    log_task("Syncing game results...")
    try:
        result = _run_async(sync_results_async())
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


def _regrade_runline_picks_once():
    """One-time fix: reset all runline-based grades so they get re-graded.

    Previous grading logic used abs() on the line which broke +1.5 underdog
    picks (graded as losses when they actually covered). Nulling out the
    results lets the next run_grading() cycle re-calculate with the fixed logic.

    Safe to run multiple times — only touches snapshots that have runline results.
    """
    try:
        loop = _mlb_loop
        if loop is None:
            return

        async def _fix():
            from sqlalchemy import text
            async with _mlb_session_factory() as session:
                result = await session.execute(text("""
                    UPDATE mlb_prediction_snapshots
                    SET best_rl_result = NULL,
                        best_rl_profit = NULL,
                        best_bet_result = NULL,
                        best_bet_profit = NULL
                    WHERE best_bet_type = 'runline'
                      AND best_bet_result IS NOT NULL
                    RETURNING id
                """))
                count = len(result.fetchall())
                await session.commit()
                return count

        count = loop.run_until_complete(_fix())
        log_task(f"Runline regrade: reset {count} snapshots for re-grading")
    except Exception as e:
        log_task(f"Runline regrade failed (non-fatal): {e}")


def run_all():
    """Run all MLB tasks once."""
    log_task("=" * 50)
    log_task("Running all MLB scheduled tasks...")

    _regrade_runline_picks_once()
    time.sleep(2)

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


# Persistent event loop for the MLB scheduler thread.
# Reusing one loop avoids creating/destroying connections per task.
_mlb_loop: asyncio.AbstractEventLoop | None = None


def _run_async(coro):
    """Run an async coroutine on the MLB scheduler's persistent event loop."""
    global _mlb_loop
    if _mlb_loop is None:
        _mlb_loop = asyncio.new_event_loop()
    return _mlb_loop.run_until_complete(coro)


def start_scheduler():
    """Start the MLB scheduler loop.

    Uses its own schedule.Scheduler instance to avoid conflicts
    when running alongside the NBA scheduler in the same process.
    Uses a persistent event loop to avoid connection pool exhaustion.
    """
    global _mlb_loop
    _mlb_loop = asyncio.new_event_loop()

    log_task("Starting MLB prediction tracker scheduler...")

    # Delay initial run to let the app finish startup and healthcheck
    log_task("Waiting 120s before initial run to avoid connection pressure during startup...")
    time.sleep(120)

    # Initialize DB engine after startup delay so it doesn't compete with healthcheck
    _init_engine()

    # Run all tasks on startup
    run_all()

    # Use a dedicated scheduler instance (not the global default)
    # to avoid conflicts with the NBA scheduler running in another thread
    mlb_scheduler = schedule.Scheduler()

    # Schedule recurring tasks
    mlb_scheduler.every().day.at("06:00").do(run_sync_teams)  # Daily team sync
    mlb_scheduler.every(2).hours.do(run_ingest_games)
    mlb_scheduler.every(2).hours.do(run_update_stats)
    mlb_scheduler.every(2).hours.do(run_ingest_weather)
    mlb_scheduler.every(30).minutes.do(run_ingest_odds)
    mlb_scheduler.every(30).minutes.do(run_scoring)
    mlb_scheduler.every(15).minutes.do(run_snapshot)
    mlb_scheduler.every(1).hour.do(run_grading)
    mlb_scheduler.every(2).hours.do(run_sync_results)
    mlb_scheduler.every(1).hour.do(run_health_check)

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
        mlb_scheduler.run_pending()
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
