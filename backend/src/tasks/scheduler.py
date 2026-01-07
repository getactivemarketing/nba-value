#!/usr/bin/env python3
"""
Scheduler for automated prediction tracking tasks.

This script runs scheduled tasks:
1. Ingest odds (every 30 min)
2. Run scoring (every 30 min)
3. Snapshot predictions (every 2 hours, for games within 8 hours)
4. Grade completed predictions (every hour)

Can be run as a standalone process or scheduled via cron/Railway.
"""

import asyncio
import hashlib
import sys
import time
from datetime import datetime, timezone, timedelta
from decimal import Decimal

import psycopg2
import structlog
import schedule

# Add parent to path for imports (works on local and Railway)
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.services.data.odds_api import OddsAPIClient
from src.tasks.prediction_tracker import snapshot_predictions, grade_predictions

logger = structlog.get_logger()

# Use environment variable or fallback to Railway URL
DB_URL = os.environ.get('DATABASE_URL', 'postgresql://postgres:wzYHkiAOkykxiPitXKBIqPJxvifFtDPI@maglev.proxy.rlwy.net:46068/railway')

TEAM_ABBREV_MAP = {
    "Los Angeles Lakers": "LAL", "Los Angeles Clippers": "LAC",
    "Boston Celtics": "BOS", "New York Knicks": "NYK", "Brooklyn Nets": "BKN",
    "Philadelphia 76ers": "PHI", "Toronto Raptors": "TOR", "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE", "Detroit Pistons": "DET", "Indiana Pacers": "IND",
    "Milwaukee Bucks": "MIL", "Atlanta Hawks": "ATL", "Charlotte Hornets": "CHA",
    "Miami Heat": "MIA", "Orlando Magic": "ORL", "Washington Wizards": "WAS",
    "Denver Nuggets": "DEN", "Minnesota Timberwolves": "MIN", "Oklahoma City Thunder": "OKC",
    "Portland Trail Blazers": "POR", "Utah Jazz": "UTA", "Golden State Warriors": "GSW",
    "Phoenix Suns": "PHX", "Sacramento Kings": "SAC", "Dallas Mavericks": "DAL",
    "Houston Rockets": "HOU", "Memphis Grizzlies": "MEM", "New Orleans Pelicans": "NOP",
    "San Antonio Spurs": "SAS",
}


def get_team_abbrev(team_name):
    return TEAM_ABBREV_MAP.get(team_name, team_name[:3].upper())


def short_hash(s, length=8):
    return hashlib.md5(s.encode()).hexdigest()[:length]


async def ingest_odds_async():
    """Fetch and store odds from API."""
    client = OddsAPIClient()
    now = datetime.now(timezone.utc)

    try:
        odds_data = await client.get_nba_odds(markets=["h2h", "spreads", "totals"])
        logger.info(f"Fetched {len(odds_data)} games", requests_remaining=client.requests_remaining)
    except Exception as e:
        logger.error(f"Failed to fetch odds: {e}")
        return {"error": str(e)}

    conn = psycopg2.connect(DB_URL)
    conn.autocommit = True
    cur = conn.cursor()

    games_created = 0
    markets_created = 0

    for game_data in odds_data:
        home_team = get_team_abbrev(game_data["home_team"])
        away_team = get_team_abbrev(game_data["away_team"])
        commence_time = datetime.fromisoformat(game_data["commence_time"].replace("Z", "+00:00"))
        game_id = game_data["id"]

        # Upsert game
        cur.execute('''
            INSERT INTO games (game_id, league, season, game_date, tip_time_utc, home_team_id, away_team_id, status, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (game_id) DO UPDATE SET
                tip_time_utc = EXCLUDED.tip_time_utc,
                status = EXCLUDED.status,
                updated_at = EXCLUDED.updated_at
        ''', (game_id, 'NBA', 2025, commence_time.date(), commence_time, home_team, away_team, 'scheduled', now, now))
        games_created += 1

        # Process markets
        for bookmaker in game_data.get("bookmakers", [])[:1]:
            book_name = bookmaker["key"]

            for market in bookmaker.get("markets", []):
                market_key = market["key"]

                for outcome in market.get("outcomes", []):
                    outcome_name = outcome["name"]
                    odds = outcome.get("price", 2.0)
                    line = outcome.get("point")

                    if market_key == "h2h":
                        market_type = "moneyline"
                        outcome_label = "home_win" if outcome_name == game_data["home_team"] else "away_win"
                    elif market_key == "spreads":
                        market_type = "spread"
                        outcome_label = "home_spread" if outcome_name == game_data["home_team"] else "away_spread"
                    elif market_key == "totals":
                        market_type = "total"
                        outcome_label = outcome_name.lower()
                    else:
                        continue

                    market_id = f"{short_hash(game_id)}_{market_type[:3]}_{outcome_label[:4]}"

                    cur.execute('''
                        INSERT INTO markets (market_id, game_id, market_type, outcome_label, line, odds_decimal, book, is_active, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (market_id) DO UPDATE SET
                            line = EXCLUDED.line,
                            odds_decimal = EXCLUDED.odds_decimal,
                            is_active = EXCLUDED.is_active,
                            updated_at = EXCLUDED.updated_at
                    ''', (market_id, game_id, market_type, outcome_label, line, odds, book_name, True, now, now))
                    markets_created += 1

    cur.close()
    conn.close()

    return {"games": games_created, "markets": markets_created}


def run_ingest():
    """Sync wrapper for odds ingestion."""
    logger.info("Running odds ingestion...")
    result = asyncio.run(ingest_odds_async())
    logger.info(f"Ingestion complete: {result}")
    return result


def run_scoring():
    """Run the scoring pipeline."""
    logger.info("Running scoring pipeline...")
    try:
        # Import and run scoring
        from src.tasks.scoring import _run_pre_game_scoring_async
        result = asyncio.run(_run_pre_game_scoring_async())
        logger.info(f"Scoring complete: {result}")
        return result
    except ImportError:
        logger.warning("Could not import scoring module, running minimal scoring")
        return {"status": "skipped"}


def run_snapshot():
    """Snapshot predictions for upcoming games."""
    logger.info("Running prediction snapshot...")
    result = snapshot_predictions(hours_ahead=8)
    logger.info(f"Snapshot complete: {result}")
    return result


def run_grading():
    """Grade completed predictions."""
    logger.info("Running prediction grading...")
    result = grade_predictions()
    logger.info(f"Grading complete: {result}")
    return result


def run_all():
    """Run all tasks once."""
    logger.info("=" * 50)
    logger.info("Running all scheduled tasks...")

    run_ingest()
    time.sleep(2)

    run_snapshot()
    time.sleep(2)

    run_grading()

    logger.info("All tasks complete")
    logger.info("=" * 50)


def start_scheduler():
    """Start the scheduler loop."""
    logger.info("Starting prediction tracker scheduler...")

    # Run all tasks immediately on startup
    run_all()

    # Schedule recurring tasks
    schedule.every(30).minutes.do(run_ingest)
    schedule.every(2).hours.do(run_snapshot)
    schedule.every(1).hour.do(run_grading)

    logger.info("Scheduler configured:")
    logger.info("  - Odds ingestion: every 30 minutes")
    logger.info("  - Prediction snapshot: every 2 hours")
    logger.info("  - Grading: every 1 hour")

    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute


if __name__ == '__main__':
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == 'ingest':
            run_ingest()
        elif command == 'snapshot':
            run_snapshot()
        elif command == 'grade':
            run_grading()
        elif command == 'all':
            run_all()
        elif command == 'daemon':
            start_scheduler()
        else:
            print(f"Unknown command: {command}")
            print("Usage: python scheduler.py [ingest|snapshot|grade|all|daemon]")
    else:
        # Default: run all tasks once
        run_all()
# Scheduler service
