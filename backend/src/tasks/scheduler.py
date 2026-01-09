#!/usr/bin/env python3
"""
Scheduler for automated prediction tracking tasks.

This script runs scheduled tasks:
1. Update team stats (every 2 hours) - rest days, B2B, records
2. Ingest odds (every 30 min)
3. Run scoring (every 30 min)
4. Snapshot predictions (every 15 min, captures games ~30 min before tip)
5. Grade completed predictions (every hour)
6. Sync game results (every 2 hours)

Can be run as a standalone process or scheduled via cron/Railway.
"""

import asyncio
import hashlib
import sys
import time
from datetime import datetime, timezone, timedelta, date
from decimal import Decimal
from collections import defaultdict

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


async def update_team_stats_async() -> dict:
    """
    Update team statistics including rest days, B2B status, and records.
    Uses BallDontLie API for game history.
    """
    from src.services.data.balldontlie import BallDontLieClient

    # Use EST for "today" since NBA games are scheduled in ET
    eastern = timedelta(hours=-5)
    now_est = datetime.now(timezone.utc) + eastern
    today = now_est.date()
    now = datetime.now(timezone.utc)

    season_start = date(2025, 10, 22)  # 2025-26 NBA season

    client = BallDontLieClient()
    logger.info("Fetching games from BallDontLie for team stats...")

    try:
        all_games = await client.get_games(
            start_date=season_start,
            end_date=today,
            seasons=[2025],
        )
    except Exception as e:
        logger.error(f"Failed to fetch games for team stats: {e}")
        return {"error": str(e), "status": "failed"}

    completed = [g for g in all_games if g.status == "Final" and g.home_team_score]
    logger.info(f"Found {len(completed)} completed games for stats calculation")

    if not completed:
        return {"teams_updated": 0, "status": "no_games"}

    # Group games by team
    team_games = defaultdict(list)
    for game in completed:
        team_games[game.home_team.abbreviation].append({
            'date': game.date,
            'is_home': True,
            'points_for': game.home_team_score,
            'points_against': game.away_team_score,
            'won': game.home_team_score > game.away_team_score,
        })
        team_games[game.away_team.abbreviation].append({
            'date': game.date,
            'is_home': False,
            'points_for': game.away_team_score,
            'points_against': game.home_team_score,
            'won': game.away_team_score > game.home_team_score,
        })

    conn = psycopg2.connect(DB_URL)
    conn.autocommit = True
    cur = conn.cursor()

    # Get ATS and O/U records from game_results (last 10 games per team)
    # ATS: 'home_cover' means home team covered, 'away_cover' means away team covered
    # After transformation: 'win' = team covered, 'loss' = team didn't cover
    ats_ou_records = {}
    cur.execute('''
        WITH team_games AS (
            -- Home team games: home_cover = WIN, away_cover = LOSS
            SELECT
                home_team_id AS team,
                CASE
                    WHEN spread_result = 'home_cover' THEN 'win'
                    WHEN spread_result = 'away_cover' THEN 'loss'
                    ELSE 'push'
                END AS ats_result,
                total_result,
                game_date
            FROM game_results
            WHERE spread_result IS NOT NULL
            UNION ALL
            -- Away team games: away_cover = WIN, home_cover = LOSS
            SELECT
                away_team_id AS team,
                CASE
                    WHEN spread_result = 'away_cover' THEN 'win'
                    WHEN spread_result = 'home_cover' THEN 'loss'
                    ELSE 'push'
                END AS ats_result,
                total_result,
                game_date
            FROM game_results
            WHERE spread_result IS NOT NULL
        ),
        numbered AS (
            SELECT
                team,
                ats_result,
                total_result,
                ROW_NUMBER() OVER (PARTITION BY team ORDER BY game_date DESC) as rn
            FROM team_games
        )
        SELECT
            team,
            SUM(CASE WHEN ats_result = 'win' THEN 1 ELSE 0 END) as ats_wins,
            SUM(CASE WHEN ats_result = 'loss' THEN 1 ELSE 0 END) as ats_losses,
            SUM(CASE WHEN ats_result = 'push' THEN 1 ELSE 0 END) as ats_pushes,
            SUM(CASE WHEN total_result = 'over' THEN 1 ELSE 0 END) as ou_overs,
            SUM(CASE WHEN total_result = 'under' THEN 1 ELSE 0 END) as ou_unders,
            SUM(CASE WHEN total_result = 'push' THEN 1 ELSE 0 END) as ou_pushes
        FROM numbered
        WHERE rn <= 10
        GROUP BY team
    ''')

    for row in cur.fetchall():
        team, ats_wins, ats_losses, ats_pushes, ou_overs, ou_unders, ou_pushes = row
        ats_ou_records[team] = {
            'ats_wins': ats_wins or 0,
            'ats_losses': ats_losses or 0,
            'ats_pushes': ats_pushes or 0,
            'ou_overs': ou_overs or 0,
            'ou_unders': ou_unders or 0,
            'ou_pushes': ou_pushes or 0,
        }

    teams_updated = 0

    for team_abbr, games in team_games.items():
        games.sort(key=lambda x: x['date'], reverse=True)

        wins = sum(1 for g in games if g['won'])
        losses = len(games) - wins

        last_10 = games[:10]
        wins_l10 = sum(1 for g in last_10 if g['won'])
        losses_l10 = len(last_10) - wins_l10
        win_pct_10 = round(wins_l10 / len(last_10), 3) if last_10 else None

        ppg_season = round(sum(g['points_for'] for g in games) / len(games), 2)
        opp_ppg_season = round(sum(g['points_against'] for g in games) / len(games), 2)
        ppg_10 = round(sum(g['points_for'] for g in last_10) / len(last_10), 2)
        opp_ppg_10 = round(sum(g['points_against'] for g in last_10) / len(last_10), 2)

        net_rtg_season = ppg_season - opp_ppg_season
        net_rtg_10 = ppg_10 - opp_ppg_10

        home_games = [g for g in games if g['is_home']]
        away_games = [g for g in games if not g['is_home']]
        home_wins = sum(1 for g in home_games if g['won'])
        home_losses = len(home_games) - home_wins
        away_wins = sum(1 for g in away_games if g['won'])
        away_losses = len(away_games) - away_wins

        # Rest calculation
        last_game_date = games[0]['date']
        if isinstance(last_game_date, datetime):
            last_game_date = last_game_date.date()

        days_rest = (today - last_game_date).days
        is_b2b = days_rest <= 1

        # Games in last 7 days
        week_ago = today - timedelta(days=7)
        games_last_7 = sum(1 for g in games if g['date'] >= week_ago)

        # Get ATS/O/U records for this team
        ats_ou = ats_ou_records.get(team_abbr, {})
        ats_wins_l10 = ats_ou.get('ats_wins', 0)
        ats_losses_l10 = ats_ou.get('ats_losses', 0)
        ats_pushes_l10 = ats_ou.get('ats_pushes', 0)
        ou_overs_l10 = ats_ou.get('ou_overs', 0)
        ou_unders_l10 = ats_ou.get('ou_unders', 0)
        ou_pushes_l10 = ats_ou.get('ou_pushes', 0)

        # Upsert team stats
        cur.execute('''
            INSERT INTO team_stats (
                team_id, stat_date, games_played, wins, losses,
                wins_l10, losses_l10, win_pct_10,
                home_wins, home_losses, away_wins, away_losses,
                ppg_10, ppg_season, opp_ppg_10, opp_ppg_season,
                net_rtg_10, net_rtg_season,
                days_rest, is_back_to_back, games_last_7_days,
                ats_wins_l10, ats_losses_l10, ats_pushes_l10,
                ou_overs_l10, ou_unders_l10, ou_pushes_l10,
                created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (team_id, stat_date) DO UPDATE SET
                games_played = EXCLUDED.games_played,
                wins = EXCLUDED.wins,
                losses = EXCLUDED.losses,
                wins_l10 = EXCLUDED.wins_l10,
                losses_l10 = EXCLUDED.losses_l10,
                win_pct_10 = EXCLUDED.win_pct_10,
                home_wins = EXCLUDED.home_wins,
                home_losses = EXCLUDED.home_losses,
                away_wins = EXCLUDED.away_wins,
                away_losses = EXCLUDED.away_losses,
                ppg_10 = EXCLUDED.ppg_10,
                ppg_season = EXCLUDED.ppg_season,
                opp_ppg_10 = EXCLUDED.opp_ppg_10,
                opp_ppg_season = EXCLUDED.opp_ppg_season,
                net_rtg_10 = EXCLUDED.net_rtg_10,
                net_rtg_season = EXCLUDED.net_rtg_season,
                days_rest = EXCLUDED.days_rest,
                is_back_to_back = EXCLUDED.is_back_to_back,
                games_last_7_days = EXCLUDED.games_last_7_days,
                ats_wins_l10 = EXCLUDED.ats_wins_l10,
                ats_losses_l10 = EXCLUDED.ats_losses_l10,
                ats_pushes_l10 = EXCLUDED.ats_pushes_l10,
                ou_overs_l10 = EXCLUDED.ou_overs_l10,
                ou_unders_l10 = EXCLUDED.ou_unders_l10,
                ou_pushes_l10 = EXCLUDED.ou_pushes_l10
        ''', (
            team_abbr, today, len(games), wins, losses,
            wins_l10, losses_l10, win_pct_10,
            home_wins, home_losses, away_wins, away_losses,
            ppg_10, ppg_season, opp_ppg_10, opp_ppg_season,
            net_rtg_10, net_rtg_season,
            days_rest, is_b2b, games_last_7,
            ats_wins_l10, ats_losses_l10, ats_pushes_l10,
            ou_overs_l10, ou_unders_l10, ou_pushes_l10,
            now
        ))

        teams_updated += 1

    cur.close()
    conn.close()

    return {"teams_updated": teams_updated, "stat_date": str(today), "status": "success"}


def run_team_stats():
    """Update team statistics (rest, B2B, records)."""
    logger.info("Running team stats update...")
    result = asyncio.run(update_team_stats_async())
    logger.info(f"Team stats update complete: {result}")
    return result


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
    """Snapshot predictions for games starting soon (within 45 minutes).

    This runs every 15 minutes to capture predictions ~30 mins before tip-off.
    The NOT EXISTS check in snapshot_predictions prevents re-snapshotting.
    """
    logger.info("Running prediction snapshot...")
    # Use 0.75 hours (45 min) window - combined with 15 min schedule,
    # games get snapshotted 15-45 min before tip-off
    result = snapshot_predictions(hours_ahead=0.75)
    logger.info(f"Snapshot complete: {result}")
    return result


def run_grading():
    """Grade completed predictions."""
    logger.info("Running prediction grading...")
    result = grade_predictions()
    logger.info(f"Grading complete: {result}")
    return result


def run_results_sync():
    """Sync game results from completed games, then grade predictions."""
    logger.info("Running results sync...")
    result = sync_game_results()
    logger.info(f"Results sync complete: {result}")

    # Automatically grade predictions after syncing results
    if result.get('results_created', 0) > 0 or result.get('games_synced', 0) > 0:
        logger.info("New results found, running prediction grading...")
        grade_result = grade_predictions()
        logger.info(f"Grading complete: {grade_result}")
        result['grading'] = grade_result

    return result


def sync_game_results() -> dict:
    """
    Sync final scores and populate game_results table.
    Uses BallDontLie (free) for scores.
    """
    from datetime import date

    conn = psycopg2.connect(DB_URL)
    conn.autocommit = True
    cur = conn.cursor()

    now = datetime.now(timezone.utc)
    today = date.today()
    yesterday = today - timedelta(days=1)
    two_days_ago = today - timedelta(days=2)

    games_synced = 0
    results_created = 0

    try:
        # Import BallDontLie client
        from src.services.data.balldontlie import BallDontLieClient
        client = BallDontLieClient()

        # Fetch games from last 2 days
        all_games = asyncio.run(client.get_games(
            start_date=two_days_ago,
            end_date=today,
            seasons=[2025]
        ))

        completed = [g for g in all_games if g.status == 'Final' and g.home_team_score]
        logger.info(f"Found {len(completed)} completed games")

        for game in completed:
            home_abbr = game.home_team.abbreviation
            away_abbr = game.away_team.abbreviation
            home_score = game.home_team_score
            away_score = game.away_team_score
            game_date = game.date
            # BallDontLie uses EST dates, but our DB may have UTC dates (+1 day for evening games)
            game_date_utc = game_date + timedelta(days=1)

            # Update games table with final score (try both EST and UTC dates)
            cur.execute('''
                UPDATE games
                SET home_score = %s, away_score = %s, status = 'final', updated_at = %s
                WHERE game_date IN (%s, %s)
                AND home_team_id = %s
                AND away_team_id = %s
                AND status != 'final'
            ''', (home_score, away_score, now, game_date, game_date_utc, home_abbr, away_abbr))

            if cur.rowcount > 0:
                games_synced += 1

            # Check if game_results already exists (check both dates)
            cur.execute('''
                SELECT 1 FROM game_results
                WHERE game_date IN (%s, %s) AND home_team_id = %s AND away_team_id = %s
            ''', (game_date, game_date_utc, home_abbr, away_abbr))

            if cur.fetchone():
                continue  # Already have results

            # Get game_id from games table (check both dates)
            cur.execute('''
                SELECT game_id, game_date FROM games
                WHERE game_date IN (%s, %s) AND home_team_id = %s AND away_team_id = %s
            ''', (game_date, game_date_utc, home_abbr, away_abbr))

            row = cur.fetchone()
            if not row:
                continue

            game_id = row[0]
            db_game_date = row[1]  # Use the date from our DB

            # Get closing lines from markets table
            cur.execute('''
                SELECT line FROM markets
                WHERE game_id = %s AND market_type = 'spread' AND outcome_label = 'home_spread'
                LIMIT 1
            ''', (game_id,))
            spread_row = cur.fetchone()
            closing_spread = float(spread_row[0]) if spread_row and spread_row[0] else None

            cur.execute('''
                SELECT line FROM markets
                WHERE game_id = %s AND market_type = 'total' AND outcome_label = 'over'
                LIMIT 1
            ''', (game_id,))
            total_row = cur.fetchone()
            closing_total = float(total_row[0]) if total_row and total_row[0] else None

            # Calculate results
            actual_winner = home_abbr if home_score > away_score else away_abbr
            total_score = home_score + away_score

            spread_result = None
            if closing_spread is not None:
                home_adjusted = home_score + closing_spread
                if home_adjusted > away_score:
                    spread_result = 'home_cover'
                elif home_adjusted < away_score:
                    spread_result = 'away_cover'
                else:
                    spread_result = 'push'

            total_result = None
            if closing_total is not None:
                if total_score > closing_total:
                    total_result = 'over'
                elif total_score < closing_total:
                    total_result = 'under'
                else:
                    total_result = 'push'

            # Insert game_results
            cur.execute('''
                INSERT INTO game_results (
                    game_id, game_date, home_team_id, away_team_id,
                    home_score, away_score, total_score,
                    closing_spread, closing_total,
                    actual_winner, spread_result, total_result,
                    created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (game_id) DO NOTHING
            ''', (
                game_id, db_game_date, home_abbr, away_abbr,
                home_score, away_score, total_score,
                closing_spread, closing_total,
                actual_winner, spread_result, total_result,
                now
            ))

            if cur.rowcount > 0:
                results_created += 1

        cur.close()
        conn.close()

        return {
            "games_synced": games_synced,
            "results_created": results_created,
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Results sync failed: {e}")
        cur.close()
        conn.close()
        return {"error": str(e), "status": "failed"}


def backfill_game_results() -> dict:
    """
    Backfill game_results table with full season data.
    This is needed for accurate ATS/O/U records.
    """
    from datetime import date

    conn = psycopg2.connect(DB_URL)
    conn.autocommit = True
    cur = conn.cursor()

    now = datetime.now(timezone.utc)
    today = date.today()
    season_start = date(2025, 10, 22)  # 2025-26 NBA season

    results_created = 0

    try:
        from src.services.data.balldontlie import BallDontLieClient
        client = BallDontLieClient()

        # Fetch ALL games from season start
        logger.info(f"Backfilling game_results from {season_start} to {today}...")
        all_games = asyncio.run(client.get_games(
            start_date=season_start,
            end_date=today,
            seasons=[2025]
        ))

        completed = [g for g in all_games if g.status == 'Final' and g.home_team_score]
        logger.info(f"Found {len(completed)} completed games to backfill")

        for game in completed:
            home_abbr = game.home_team.abbreviation
            away_abbr = game.away_team.abbreviation
            home_score = game.home_team_score
            away_score = game.away_team_score
            game_date = game.date
            # Use UTC date (games played at night ET are next day UTC)
            game_date_utc = game_date + timedelta(days=1)

            # Generate a consistent game_id
            game_id = short_hash(f"{game_date}_{home_abbr}_{away_abbr}")

            # Check if already exists
            cur.execute('''
                SELECT 1 FROM game_results
                WHERE game_date = %s AND home_team_id = %s AND away_team_id = %s
            ''', (game_date_utc, home_abbr, away_abbr))

            if cur.fetchone():
                continue

            # Calculate results (without closing lines for historical)
            actual_winner = home_abbr if home_score > away_score else away_abbr
            total_score = home_score + away_score

            # We don't have closing lines for historical games, so leave spread_result/total_result NULL
            # This means ATS won't count these, but at least we have the data
            cur.execute('''
                INSERT INTO game_results (
                    game_id, game_date, home_team_id, away_team_id,
                    home_score, away_score, total_score,
                    closing_spread, closing_total,
                    actual_winner, spread_result, total_result,
                    created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (game_id) DO NOTHING
            ''', (
                game_id, game_date_utc, home_abbr, away_abbr,
                home_score, away_score, total_score,
                None, None,  # No closing lines for historical
                actual_winner, None, None,  # No spread/total results without lines
                now
            ))

            if cur.rowcount > 0:
                results_created += 1

        cur.close()
        conn.close()

        logger.info(f"Backfill complete: {results_created} new game_results created")
        return {"results_created": results_created, "status": "success"}

    except Exception as e:
        logger.error(f"Backfill failed: {e}")
        cur.close()
        conn.close()
        return {"error": str(e), "status": "failed"}


def run_all():
    """Run all tasks once."""
    logger.info("=" * 50)
    logger.info("Running all scheduled tasks...")

    run_team_stats()  # Update rest/B2B data first
    time.sleep(2)

    run_ingest()
    time.sleep(2)

    run_scoring()
    time.sleep(2)

    run_snapshot()
    time.sleep(2)

    run_grading()
    time.sleep(2)

    run_results_sync()

    logger.info("All tasks complete")
    logger.info("=" * 50)


def start_scheduler():
    """Start the scheduler loop."""
    logger.info("Starting prediction tracker scheduler...")

    # Run all tasks immediately on startup
    run_all()

    # Schedule recurring tasks
    schedule.every(2).hours.do(run_team_stats)  # Update rest/B2B data
    schedule.every(30).minutes.do(run_ingest)
    schedule.every(30).minutes.do(run_scoring)  # Run scoring with odds
    schedule.every(15).minutes.do(run_snapshot)  # Capture predictions ~30 min before tip
    schedule.every(1).hour.do(run_grading)
    schedule.every(2).hours.do(run_results_sync)

    logger.info("Scheduler configured:")
    logger.info("  - Team stats update: every 2 hours")
    logger.info("  - Odds ingestion: every 30 minutes")
    logger.info("  - Scoring: every 30 minutes")
    logger.info("  - Prediction snapshot: every 15 minutes (45 min window)")
    logger.info("  - Grading: every 1 hour")
    logger.info("  - Results sync: every 2 hours")

    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute


if __name__ == '__main__':
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == 'stats':
            run_team_stats()
        elif command == 'ingest':
            run_ingest()
        elif command == 'snapshot':
            run_snapshot()
        elif command == 'grade':
            run_grading()
        elif command == 'results':
            run_results_sync()
        elif command == 'all':
            run_all()
        elif command == 'daemon':
            start_scheduler()
        else:
            print(f"Unknown command: {command}")
            print("Usage: python scheduler.py [stats|ingest|snapshot|grade|results|all|daemon]")
    else:
        # Default: run all tasks once
        run_all()
