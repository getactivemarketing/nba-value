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

# Track last run times for health monitoring
_last_run_times: dict[str, datetime] = {}


def log_task(message: str, **kwargs):
    """Log scheduler task output in a way that's visible in Railway logs."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    extra = " ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
    print(f"[SCHEDULER] {timestamp} | {message} {extra}", flush=True)

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
    log_task("Running team stats update...")
    try:
        result = asyncio.run(update_team_stats_async())
        log_task("Team stats complete", **result)
        _last_run_times['team_stats'] = datetime.now(timezone.utc)
        return result
    except Exception as e:
        log_task(f"Team stats FAILED: {e}")
        _last_run_times['team_stats'] = datetime.now(timezone.utc)  # Track attempt
        return {"status": "failed", "error": str(e)}


async def ingest_odds_async():
    """Fetch and store odds from API, including snapshots for line movement tracking."""
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
    snapshots_created = 0

    for game_data in odds_data:
        home_team = get_team_abbrev(game_data["home_team"])
        away_team = get_team_abbrev(game_data["away_team"])
        commence_time = datetime.fromisoformat(game_data["commence_time"].replace("Z", "+00:00"))
        game_id = game_data["id"]

        # Calculate minutes to tip for snapshots
        minutes_to_tip = int((commence_time - now).total_seconds() / 60)

        # Convert to Eastern Time for game_date (NBA games are scheduled in ET)
        eastern_offset = timedelta(hours=-5)
        commence_time_et = commence_time + eastern_offset
        game_date_et = commence_time_et.date()

        # Upsert game
        cur.execute('''
            INSERT INTO games (game_id, league, season, game_date, tip_time_utc, home_team_id, away_team_id, status, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (game_id) DO UPDATE SET
                game_date = EXCLUDED.game_date,
                tip_time_utc = EXCLUDED.tip_time_utc,
                status = EXCLUDED.status,
                updated_at = EXCLUDED.updated_at
        ''', (game_id, 'NBA', 2025, game_date_et, commence_time, home_team, away_team, 'scheduled', now, now))
        games_created += 1

        # Process markets and collect data for snapshots
        for bookmaker in game_data.get("bookmakers", [])[:1]:
            book_name = bookmaker["key"]

            # Collect odds data for snapshot
            snapshot_data = {
                'home_spread': None, 'home_spread_odds': None,
                'away_spread': None, 'away_spread_odds': None,
                'home_ml_odds': None, 'away_ml_odds': None,
                'total_line': None, 'over_odds': None, 'under_odds': None,
            }

            for market in bookmaker.get("markets", []):
                market_key = market["key"]

                for outcome in market.get("outcomes", []):
                    outcome_name = outcome["name"]
                    odds = outcome.get("price", 2.0)
                    line = outcome.get("point")

                    if market_key == "h2h":
                        market_type = "moneyline"
                        if outcome_name == game_data["home_team"]:
                            outcome_label = "home_win"
                            snapshot_data['home_ml_odds'] = odds
                        else:
                            outcome_label = "away_win"
                            snapshot_data['away_ml_odds'] = odds
                    elif market_key == "spreads":
                        market_type = "spread"
                        if outcome_name == game_data["home_team"]:
                            outcome_label = "home_spread"
                            snapshot_data['home_spread'] = line
                            snapshot_data['home_spread_odds'] = odds
                        else:
                            outcome_label = "away_spread"
                            snapshot_data['away_spread'] = line
                            snapshot_data['away_spread_odds'] = odds
                    elif market_key == "totals":
                        market_type = "total"
                        outcome_label = outcome_name.lower()
                        if outcome_name == "Over":
                            snapshot_data['total_line'] = line
                            snapshot_data['over_odds'] = odds
                        else:
                            snapshot_data['under_odds'] = odds
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

            # Insert odds snapshot for this game/book (only for scheduled games)
            if minutes_to_tip > 0:
                cur.execute('''
                    INSERT INTO odds_snapshots (
                        game_id, market_type, book_key, snapshot_time, minutes_to_tip,
                        home_spread, home_spread_odds, away_spread, away_spread_odds,
                        home_ml_odds, away_ml_odds,
                        total_line, over_odds, under_odds,
                        is_closing_line, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ''', (
                    game_id, 'all', book_name, now, minutes_to_tip,
                    snapshot_data['home_spread'], snapshot_data['home_spread_odds'],
                    snapshot_data['away_spread'], snapshot_data['away_spread_odds'],
                    snapshot_data['home_ml_odds'], snapshot_data['away_ml_odds'],
                    snapshot_data['total_line'], snapshot_data['over_odds'], snapshot_data['under_odds'],
                    False,  # is_closing_line - these are periodic snapshots, not closing lines
                    now
                ))
                snapshots_created += 1

    cur.close()
    conn.close()

    return {"games": games_created, "markets": markets_created, "snapshots": snapshots_created}


def run_ingest():
    """Sync wrapper for odds ingestion."""
    log_task("Running odds ingestion...")
    try:
        result = asyncio.run(ingest_odds_async())
        log_task("Ingestion complete", **result)
        _last_run_times['ingest'] = datetime.now(timezone.utc)
        return result
    except Exception as e:
        log_task(f"Ingestion FAILED: {e}")
        _last_run_times['ingest'] = datetime.now(timezone.utc)
        return {"status": "failed", "error": str(e)}


async def ingest_player_props_async(hours_ahead: int = 6, max_games: int = 3):
    """
    Fetch and store player props for upcoming games.

    Only fetches props for games starting within `hours_ahead` hours to conserve API quota.
    Each game requires a separate API call.

    Args:
        hours_ahead: Only fetch props for games starting within this many hours
        max_games: Maximum number of games to fetch props for (to limit API calls)
    """
    client = OddsAPIClient()
    now = datetime.now(timezone.utc)
    cutoff = now + timedelta(hours=hours_ahead)

    conn = psycopg2.connect(DB_URL)
    conn.autocommit = True
    cur = conn.cursor()

    # Get upcoming games within the window
    cur.execute('''
        SELECT game_id, tip_time_utc, home_team_id, away_team_id
        FROM games
        WHERE status = 'scheduled'
        AND tip_time_utc > %s
        AND tip_time_utc < %s
        ORDER BY tip_time_utc ASC
        LIMIT %s
    ''', (now, cutoff, max_games))

    games = cur.fetchall()
    logger.info(f"Found {len(games)} games for player props ingestion")

    if not games:
        cur.close()
        conn.close()
        return {"games_checked": 0, "props_created": 0}

    props_created = 0
    games_processed = 0

    for game_id, tip_time, home_team, away_team in games:
        try:
            # Fetch player props for this game
            props_data = await client.get_player_props(
                event_id=game_id,
                markets=["player_points", "player_rebounds", "player_assists"],
            )

            # Parse the props
            props = client.parse_player_props(props_data)
            logger.info(f"Fetched {len(props)} props for {away_team} @ {home_team}")

            # Insert props into database
            for prop in props:
                # Try to determine player's team from context
                player_team = None  # Could be enhanced with player roster lookup

                cur.execute('''
                    INSERT INTO player_props (
                        game_id, player_name, player_team, prop_type,
                        line, over_odds, under_odds, book, snapshot_time, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ''', (
                    prop['game_id'],
                    prop['player_name'],
                    player_team,
                    prop['prop_type'],
                    prop['line'],
                    prop.get('over_odds'),
                    prop.get('under_odds'),
                    prop['book'],
                    prop['snapshot_time'],
                    now
                ))
                props_created += 1

            games_processed += 1

        except Exception as e:
            logger.error(f"Failed to fetch props for game {game_id}: {e}")
            continue

    cur.close()
    conn.close()

    return {
        "games_checked": len(games),
        "games_processed": games_processed,
        "props_created": props_created,
        "requests_remaining": client.requests_remaining,
    }


def run_props():
    """Sync wrapper for player props ingestion."""
    log_task("Running player props ingestion...")
    try:
        result = asyncio.run(ingest_player_props_async())
        log_task("Props ingestion complete", **result)
        _last_run_times['props'] = datetime.now(timezone.utc)
        return result
    except Exception as e:
        log_task(f"Props ingestion FAILED: {e}")
        _last_run_times['props'] = datetime.now(timezone.utc)
        return {"status": "failed", "error": str(e)}


def run_prop_snapshot():
    """Score props and save predictions to prop_snapshots table."""
    from src.services.scoring.prop_scorer import snapshot_top_props

    log_task("Running prop snapshot...")
    try:
        saved = asyncio.run(snapshot_top_props(min_score=50))
        log_task(f"Prop snapshot complete: {saved} props saved")
        _last_run_times['prop_snapshot'] = datetime.now(timezone.utc)
        return {"status": "success", "saved": saved}
    except Exception as e:
        log_task(f"Prop snapshot FAILED: {e}")
        _last_run_times['prop_snapshot'] = datetime.now(timezone.utc)
        return {"status": "failed", "error": str(e)}


def run_prop_grade():
    """Grade prop predictions from completed games."""
    from src.services.scoring.prop_scorer import grade_prop_snapshots

    log_task("Running prop grading...")
    try:
        result = asyncio.run(grade_prop_snapshots(days_back=3))
        log_task(f"Prop grading complete", **result)
        _last_run_times['prop_grade'] = datetime.now(timezone.utc)
        return {"status": "success", **result}
    except Exception as e:
        log_task(f"Prop grading FAILED: {e}")
        _last_run_times['prop_grade'] = datetime.now(timezone.utc)
        return {"status": "failed", "error": str(e)}


def run_scoring_sync() -> dict:
    """
    Synchronous scoring using psycopg2 directly.

    This avoids the asyncio event loop conflicts that occur when running
    async SQLAlchemy code from a background thread.
    """
    from src.services.scoring.scorer import ScoringService, ScoringInput, get_scoring_service

    now = datetime.now(timezone.utc)
    cutoff = now + timedelta(hours=24)
    today = now.date()

    conn = psycopg2.connect(DB_URL)
    conn.autocommit = True
    cur = conn.cursor()

    markets_scored = 0
    errors = 0

    # Fetch injury reports
    try:
        injury_reports = asyncio.run(get_all_team_injury_reports())
        logger.info(f"Fetched injury reports for {len(injury_reports)} teams")
    except Exception as e:
        logger.warning(f"Failed to fetch injury reports: {e}")
        injury_reports = {}

    scoring_service = get_scoring_service()

    # Get upcoming games
    cur.execute('''
        SELECT game_id, home_team_id, away_team_id, tip_time_utc
        FROM games
        WHERE tip_time_utc > %s AND tip_time_utc < %s AND status = 'scheduled'
        ORDER BY tip_time_utc
    ''', (now, cutoff))
    games = cur.fetchall()

    logger.info(f"Found {len(games)} games to score")

    for game_id, home_team, away_team, tip_time in games:
        try:
            # Get team stats
            def get_team_stats_dict(team_id):
                cur.execute('''
                    SELECT net_rtg_10, net_rtg_season, ppg_10, ppg_season,
                           opp_ppg_10, opp_ppg_season, days_rest, is_back_to_back, win_pct_10
                    FROM team_stats
                    WHERE team_id = %s AND stat_date <= %s
                    ORDER BY stat_date DESC LIMIT 1
                ''', (team_id, today))
                row = cur.fetchone()
                if row:
                    return {
                        'net_rtg_10': float(row[0]) if row[0] else 0.0,
                        'net_rtg_season': float(row[1]) if row[1] else 0.0,
                        'ppg_10': float(row[2]) if row[2] else 110.0,
                        'ppg_season': float(row[3]) if row[3] else 110.0,
                        'opp_ppg_10': float(row[4]) if row[4] else 110.0,
                        'opp_ppg_season': float(row[5]) if row[5] else 110.0,
                        'days_rest': row[6] if row[6] else 1,
                        'is_b2b': row[7] or False,
                        'win_pct_10': float(row[8]) if row[8] else 0.5,
                    }
                return None

            home_stats = get_team_stats_dict(home_team)
            away_stats = get_team_stats_dict(away_team)

            # Build feature dicts
            def stats_to_features(stats, prefix):
                if not stats:
                    return {f"{prefix}_net_rtg_10": 0.0, f"{prefix}_rest_days": 1, f"{prefix}_b2b": 0, f"{prefix}_win_pct_10": 0.5}
                return {
                    f"{prefix}_net_rtg_10": stats['net_rtg_10'],
                    f"{prefix}_net_rtg_season": stats['net_rtg_season'],
                    f"{prefix}_ppg_10": stats['ppg_10'],
                    f"{prefix}_ppg_season": stats['ppg_season'],
                    f"{prefix}_opp_ppg_10": stats['opp_ppg_10'],
                    f"{prefix}_opp_ppg_season": stats['opp_ppg_season'],
                    f"{prefix}_rest_days": stats['days_rest'],
                    f"{prefix}_b2b": 1 if stats['is_b2b'] else 0,
                    f"{prefix}_win_pct_10": stats['win_pct_10'],
                }

            home_features = stats_to_features(home_stats, "home")
            away_features = stats_to_features(away_stats, "away")

            # Get injury scores
            home_injury = injury_reports.get(home_team)
            away_injury = injury_reports.get(away_team)
            home_injury_score = home_injury.spread_injury_score if home_injury else 0.0
            away_injury_score = away_injury.spread_injury_score if away_injury else 0.0
            home_totals_injury = home_injury.totals_injury_score if home_injury else 0.0
            away_totals_injury = away_injury.totals_injury_score if away_injury else 0.0

            # Get markets for this game
            cur.execute('''
                SELECT market_id, market_type, outcome_label, line, odds_decimal, book
                FROM markets
                WHERE game_id = %s AND is_active = true
            ''', (game_id,))
            markets = cur.fetchall()

            # Delete old predictions/scores for these markets
            market_ids = [m[0] for m in markets]
            if market_ids:
                cur.execute('DELETE FROM value_scores WHERE market_id = ANY(%s)', (market_ids,))
                cur.execute('DELETE FROM model_predictions WHERE market_id = ANY(%s)', (market_ids,))

            # Build odds lookup for de-vigging
            odds_lookup = {}
            for m in markets:
                odds_lookup[(m[1], m[2], m[5])] = float(m[4])

            for market_id, market_type, outcome_label, line, odds_decimal, book in markets:
                try:
                    # Find opposite odds
                    opposite_label = outcome_label.lower()
                    if 'home' in opposite_label:
                        opposite_label = opposite_label.replace('home', 'away')
                    elif 'away' in opposite_label:
                        opposite_label = opposite_label.replace('away', 'home')
                    elif 'over' in opposite_label:
                        opposite_label = opposite_label.replace('over', 'under')
                    elif 'under' in opposite_label:
                        opposite_label = opposite_label.replace('under', 'over')
                    opposite_odds = odds_lookup.get((market_type, opposite_label, book), float(odds_decimal))

                    scoring_input = ScoringInput(
                        game_id=game_id,
                        market_type=market_type,
                        outcome_label=outcome_label,
                        line=float(line) if line else None,
                        odds_decimal=float(odds_decimal),
                        opposite_odds=opposite_odds,
                        home_features=home_features,
                        away_features=away_features,
                        tip_time=tip_time,
                        book=book,
                        home_injury_score=home_injury_score,
                        away_injury_score=away_injury_score,
                        home_totals_injury=home_totals_injury,
                        away_totals_injury=away_totals_injury,
                    )

                    score_result = scoring_service.score_market(scoring_input)

                    # Get edge band
                    edge = score_result.raw_edge
                    if edge < 0:
                        edge_band = "negative"
                    elif edge < 0.02:
                        edge_band = "0-2%"
                    elif edge < 0.05:
                        edge_band = "2-5%"
                    elif edge < 0.10:
                        edge_band = "5-10%"
                    else:
                        edge_band = "10%+"

                    # Insert prediction
                    cur.execute('''
                        INSERT INTO model_predictions (market_id, prediction_time, p_ensemble_mean, p_ensemble_std,
                            p_true, p_market, raw_edge, edge_band, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING prediction_id
                    ''', (market_id, score_result.calc_time, score_result.p_raw, 0.05,
                          score_result.p_calibrated, score_result.p_market, score_result.raw_edge, edge_band, now))
                    prediction_id = cur.fetchone()[0]

                    # Insert value score
                    cur.execute('''
                        INSERT INTO value_scores (prediction_id, market_id, calc_time,
                            algo_a_edge_score, algo_a_confidence, algo_a_market_quality, algo_a_value_score,
                            algo_b_combined_edge, algo_b_confidence, algo_b_market_quality, algo_b_value_score,
                            active_algorithm, time_to_tip_minutes, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ''', (prediction_id, market_id, score_result.calc_time,
                          score_result.algo_a.edge_score, score_result.algo_a_confidence.final_multiplier,
                          score_result.market_quality.final_score, score_result.algo_a.value_score,
                          score_result.algo_b.combined_edge, score_result.algo_b_confidence.final_multiplier,
                          score_result.market_quality.final_score, score_result.algo_b.value_score,
                          'A', score_result.time_to_tip_minutes, now))

                    markets_scored += 1
                except Exception as e:
                    logger.error(f"Failed to score market {market_id}: {e}")
                    errors += 1

        except Exception as e:
            logger.error(f"Failed to process game {game_id}: {e}")
            errors += 1

    cur.close()
    conn.close()

    return {"markets_scored": markets_scored, "errors": errors, "games_processed": len(games), "status": "completed"}


def get_all_team_injury_reports():
    """Wrapper to fetch injury reports."""
    from src.services.injuries import get_all_team_injury_reports as _get_reports
    return _get_reports()


def run_scoring():
    """Run the scoring pipeline."""
    log_task("Running scoring pipeline...")
    try:
        result = run_scoring_sync()
        log_task("Scoring complete", **result)
        _last_run_times['scoring'] = datetime.now(timezone.utc)
        return result
    except Exception as e:
        log_task(f"Scoring FAILED: {e}")
        return {"status": "failed", "error": str(e)}


def run_snapshot():
    """Snapshot predictions for games starting soon (within 45 minutes).

    This runs every 15 minutes to capture predictions ~30 mins before tip-off.
    The NOT EXISTS check in snapshot_predictions prevents re-snapshotting.
    """
    log_task("Running prediction snapshot...")
    try:
        # Use 0.75 hours (45 min) window - combined with 15 min schedule,
        # games get snapshotted 15-45 min before tip-off
        result = snapshot_predictions(hours_ahead=0.75)
        log_task("Snapshot complete", **result)
        _last_run_times['snapshot'] = datetime.now(timezone.utc)
        return result
    except Exception as e:
        log_task(f"Snapshot FAILED: {e}")
        _last_run_times['snapshot'] = datetime.now(timezone.utc)
        return {"status": "failed", "error": str(e)}


def run_grading():
    """Grade completed predictions."""
    log_task("Running prediction grading...")
    try:
        result = grade_predictions()
        log_task("Grading complete", **result)
        _last_run_times['grading'] = datetime.now(timezone.utc)
        return result
    except Exception as e:
        log_task(f"Grading FAILED: {e}")
        _last_run_times['grading'] = datetime.now(timezone.utc)
        return {"status": "failed", "error": str(e)}


def run_results_sync():
    """Sync game results from completed games, then grade predictions."""
    log_task("Running results sync...")
    try:
        result = sync_game_results()
        log_task("Results sync complete", **result)
        _last_run_times['results_sync'] = datetime.now(timezone.utc)

        # Automatically grade predictions after syncing results
        if result.get('results_created', 0) > 0 or result.get('games_synced', 0) > 0:
            log_task("New results found, running prediction grading...")
            grade_result = grade_predictions()
            log_task("Grading complete", **grade_result)
            result['grading'] = grade_result

        return result
    except Exception as e:
        log_task(f"Results sync FAILED: {e}")
        _last_run_times['results_sync'] = datetime.now(timezone.utc)
        return {"status": "failed", "error": str(e)}


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


def backfill_game_results_with_odds() -> dict:
    """
    Backfill game_results table with full season data INCLUDING closing lines.
    Uses paid Odds API historical endpoint to get closing spreads/totals.
    """
    import httpx
    from datetime import date
    from src.config import settings

    conn = psycopg2.connect(DB_URL)
    conn.autocommit = True
    cur = conn.cursor()

    now = datetime.now(timezone.utc)
    today = date.today()
    season_start = date(2025, 10, 22)  # 2025-26 NBA season

    results_created = 0
    api_calls = 0

    try:
        from src.services.data.balldontlie import BallDontLieClient
        bdl_client = BallDontLieClient()

        # Fetch ALL games from season start
        logger.info(f"Backfilling game_results from {season_start} to {today}...")
        all_games = asyncio.run(bdl_client.get_games(
            start_date=season_start,
            end_date=today,
            seasons=[2025]
        ))

        completed = [g for g in all_games if g.status == 'Final' and g.home_team_score]
        logger.info(f"Found {len(completed)} completed games to backfill")

        # Process games in batches by date to minimize API calls
        games_by_date = defaultdict(list)
        for game in completed:
            games_by_date[game.date].append(game)

        for game_date, date_games in sorted(games_by_date.items()):
            # Check which games need backfilling
            games_to_backfill = []
            for game in date_games:
                home_abbr = game.home_team.abbreviation
                away_abbr = game.away_team.abbreviation
                game_date_utc = game_date + timedelta(days=1)

                cur.execute('''
                    SELECT 1 FROM game_results
                    WHERE game_date = %s AND home_team_id = %s AND away_team_id = %s
                ''', (game_date_utc, home_abbr, away_abbr))

                if not cur.fetchone():
                    games_to_backfill.append(game)

            if not games_to_backfill:
                continue

            # Fetch historical odds for this date (get odds from ~1 hour before first game)
            # Games typically start 7pm ET = 00:00 UTC next day, so fetch at 23:00 UTC
            odds_time = datetime.combine(game_date, datetime.min.time().replace(hour=23))
            odds_time = odds_time.replace(tzinfo=timezone.utc)

            try:
                with httpx.Client(timeout=30) as client:
                    resp = client.get(
                        f"https://api.the-odds-api.com/v4/historical/sports/basketball_nba/odds",
                        params={
                            "apiKey": settings.odds_api_key,
                            "regions": "us",
                            "markets": "spreads,totals",
                            "oddsFormat": "decimal",
                            "date": odds_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        }
                    )
                    api_calls += 1

                    if resp.status_code != 200:
                        logger.warning(f"Historical odds API error for {game_date}: {resp.status_code}")
                        continue

                    historical_data = resp.json().get("data", [])
                    logger.info(f"Got {len(historical_data)} games from historical odds for {game_date}")

            except Exception as e:
                logger.error(f"Failed to fetch historical odds for {game_date}: {e}")
                continue

            # Build odds lookup by team matchup
            odds_lookup = {}
            for odds_game in historical_data:
                home = get_team_abbrev(odds_game.get("home_team", ""))
                away = get_team_abbrev(odds_game.get("away_team", ""))
                key = f"{home}_{away}"

                closing_spread = None
                closing_total = None

                for bookmaker in odds_game.get("bookmakers", []):
                    for market in bookmaker.get("markets", []):
                        if market["key"] == "spreads":
                            for outcome in market.get("outcomes", []):
                                if get_team_abbrev(outcome.get("name", "")) == home:
                                    closing_spread = outcome.get("point")
                                    break
                        elif market["key"] == "totals":
                            for outcome in market.get("outcomes", []):
                                if outcome.get("name") == "Over":
                                    closing_total = outcome.get("point")
                                    break
                    if closing_spread is not None and closing_total is not None:
                        break

                odds_lookup[key] = {"spread": closing_spread, "total": closing_total}

            # Insert game results with closing lines
            for game in games_to_backfill:
                home_abbr = game.home_team.abbreviation
                away_abbr = game.away_team.abbreviation
                home_score = game.home_team_score
                away_score = game.away_team_score
                game_date_utc = game_date + timedelta(days=1)

                game_id = short_hash(f"{game_date}_{home_abbr}_{away_abbr}")
                actual_winner = home_abbr if home_score > away_score else away_abbr
                total_score = home_score + away_score

                # Look up closing lines
                key = f"{home_abbr}_{away_abbr}"
                odds = odds_lookup.get(key, {})
                closing_spread = odds.get("spread")
                closing_total = odds.get("total")

                # Calculate spread result
                spread_result = None
                if closing_spread is not None:
                    home_adjusted = home_score + closing_spread
                    if home_adjusted > away_score:
                        spread_result = 'home_cover'
                    elif home_adjusted < away_score:
                        spread_result = 'away_cover'
                    else:
                        spread_result = 'push'

                # Calculate total result
                total_result = None
                if closing_total is not None:
                    if total_score > closing_total:
                        total_result = 'over'
                    elif total_score < closing_total:
                        total_result = 'under'
                    else:
                        total_result = 'push'

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
                    closing_spread, closing_total,
                    actual_winner, spread_result, total_result,
                    now
                ))

                if cur.rowcount > 0:
                    results_created += 1
                    if results_created % 50 == 0:
                        logger.info(f"Progress: {results_created} results created, {api_calls} API calls")

        cur.close()
        conn.close()

        logger.info(f"Backfill complete: {results_created} new game_results, {api_calls} API calls")
        return {"results_created": results_created, "api_calls": api_calls, "status": "success"}

    except Exception as e:
        logger.error(f"Backfill failed: {e}")
        cur.close()
        conn.close()
        return {"error": str(e), "status": "failed"}


def run_all():
    """Run all tasks once."""
    log_task("=" * 50)
    log_task("Running all scheduled tasks...")

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

    log_task("All tasks complete")
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
        "team_stats": 120,  # 2 hours
        "ingest": 30,
        "scoring": 30,
        "snapshot": 15,
        "grading": 60,
        "results_sync": 120,
    }

    for task, last_run in _last_run_times.items():
        minutes_ago = (now - last_run).total_seconds() / 60
        status["last_run_times"][task] = {
            "last_run": last_run.isoformat(),
            "minutes_ago": round(minutes_ago, 1),
        }

        # Check if task is overdue (2x expected interval)
        expected = expected_intervals.get(task, 60)
        is_healthy = minutes_ago < expected * 2
        status["task_health"][task] = "healthy" if is_healthy else "overdue"

    # Overall health
    overdue_tasks = [t for t, h in status["task_health"].items() if h == "overdue"]
    status["healthy"] = len(overdue_tasks) == 0
    status["overdue_tasks"] = overdue_tasks

    return status


def run_health_check():
    """Check scheduler health and log warnings for overdue tasks."""
    status = get_scheduler_status()

    if status["overdue_tasks"]:
        log_task(f"WARNING: Overdue tasks detected: {status['overdue_tasks']}")
        for task in status["overdue_tasks"]:
            task_info = status["last_run_times"].get(task, {})
            minutes_ago = task_info.get("minutes_ago", "unknown")
            log_task(f"  - {task}: last ran {minutes_ago} minutes ago")
    else:
        # Log a heartbeat every hour to confirm scheduler is alive
        log_task("Health check: All tasks running normally", tasks=len(_last_run_times))


def start_scheduler():
    """Start the scheduler loop."""
    log_task("Starting prediction tracker scheduler...")

    # Run all tasks immediately on startup
    run_all()

    # Schedule recurring tasks
    schedule.every(2).hours.do(run_team_stats)  # Update rest/B2B data
    schedule.every(30).minutes.do(run_ingest)
    schedule.every(30).minutes.do(run_scoring)  # Run scoring with odds
    schedule.every(15).minutes.do(run_snapshot)  # Capture predictions ~30 min before tip
    schedule.every(1).hour.do(run_grading)
    schedule.every(2).hours.do(run_results_sync)
    schedule.every(1).hour.do(run_health_check)  # Monitor task health

    log_task("Scheduler configured:")
    log_task("  - Team stats update: every 2 hours")
    log_task("  - Odds ingestion: every 30 minutes")
    log_task("  - Scoring: every 30 minutes")
    log_task("  - Prediction snapshot: every 15 minutes (45 min window)")
    log_task("  - Grading: every 1 hour")
    log_task("  - Results sync: every 2 hours")
    log_task("  - Health monitoring: every 1 hour")

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
        elif command == 'props':
            run_props()
        elif command == 'prop_snapshot':
            run_prop_snapshot()
        elif command == 'prop_grade':
            run_prop_grade()
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
            print("Usage: python scheduler.py [stats|ingest|props|prop_snapshot|prop_grade|snapshot|grade|results|all|daemon]")
    else:
        # Default: run all tasks once
        run_all()
