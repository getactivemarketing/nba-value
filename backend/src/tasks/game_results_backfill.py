"""Backfill game results with closing lines, scores, and predictions."""

import asyncio
from datetime import datetime, timezone, timedelta, date
from decimal import Decimal
from collections import defaultdict

import psycopg2
import structlog

from src.services.data.odds_api import OddsAPIClient
from src.services.data.balldontlie import BallDontLieClient

logger = structlog.get_logger()

ABBR_TO_FULL = {
    'ATL': 'Atlanta Hawks', 'BOS': 'Boston Celtics', 'BKN': 'Brooklyn Nets',
    'CHA': 'Charlotte Hornets', 'CHI': 'Chicago Bulls', 'CLE': 'Cleveland Cavaliers',
    'DAL': 'Dallas Mavericks', 'DEN': 'Denver Nuggets', 'DET': 'Detroit Pistons',
    'GSW': 'Golden State Warriors', 'HOU': 'Houston Rockets', 'IND': 'Indiana Pacers',
    'LAC': 'Los Angeles Clippers', 'LAL': 'Los Angeles Lakers', 'MEM': 'Memphis Grizzlies',
    'MIA': 'Miami Heat', 'MIL': 'Milwaukee Bucks', 'MIN': 'Minnesota Timberwolves',
    'NOP': 'New Orleans Pelicans', 'NYK': 'New York Knicks', 'OKC': 'Oklahoma City Thunder',
    'ORL': 'Orlando Magic', 'PHI': 'Philadelphia 76ers', 'PHX': 'Phoenix Suns',
    'POR': 'Portland Trail Blazers', 'SAC': 'Sacramento Kings', 'SAS': 'San Antonio Spurs',
    'TOR': 'Toronto Raptors', 'UTA': 'Utah Jazz', 'WAS': 'Washington Wizards',
}

FULL_TO_ABBR = {v: k for k, v in ABBR_TO_FULL.items()}


def calculate_spread_result(home_score: int, away_score: int, home_spread: float) -> str:
    """Calculate spread result."""
    home_adjusted = home_score + home_spread
    if home_adjusted > away_score:
        return 'home_cover'
    elif home_adjusted < away_score:
        return 'away_cover'
    else:
        return 'push'


def calculate_total_result(home_score: int, away_score: int, total_line: float) -> str:
    """Calculate over/under result."""
    actual_total = home_score + away_score
    if actual_total > total_line:
        return 'over'
    elif actual_total < total_line:
        return 'under'
    else:
        return 'push'


async def backfill_game_results(days_back: int = 30, db_url: str = None) -> dict:
    """
    Backfill game results with closing lines and outcomes.

    Args:
        days_back: Number of days to look back
        db_url: Database connection string

    Returns:
        Summary of results
    """
    today = date.today()
    start_date = today - timedelta(days=days_back)

    games_processed = 0
    games_with_lines = 0

    # ATS and O/U tracking by team
    team_ats = defaultdict(list)
    team_ou = defaultdict(list)

    # Fetch completed games
    print(f'Fetching games from {start_date} to {today}...')
    bdl_client = BallDontLieClient()
    odds_client = OddsAPIClient()

    all_games = await bdl_client.get_games(start_date=start_date, end_date=today, seasons=[2025])
    completed_games = [g for g in all_games if g.status == 'Final' and g.home_team_score and g.away_team_score]
    completed_games.sort(key=lambda x: x.date)

    print(f'Found {len(completed_games)} completed games')

    # Cache odds by date
    date_odds_cache = {}

    # Connect to database
    conn = psycopg2.connect(db_url or 'postgresql://postgres:wzYHkiAOkykxiPitXKBIqPJxvifFtDPI@maglev.proxy.rlwy.net:46068/railway')
    conn.autocommit = True
    cur = conn.cursor()

    for game in completed_games:
        home_abbr = game.home_team.abbreviation
        away_abbr = game.away_team.abbreviation
        home_full = ABBR_TO_FULL.get(home_abbr)
        away_full = ABBR_TO_FULL.get(away_abbr)

        if not home_full or not away_full:
            continue

        game_date_key = game.date.isoformat()

        # Fetch odds for this date if not cached (get both spreads and totals)
        if game_date_key not in date_odds_cache:
            game_dt = datetime.combine(game.date, datetime.min.time().replace(hour=23)).replace(tzinfo=timezone.utc)
            try:
                games_data = await odds_client.get_historical_odds(
                    date=game_dt,
                    markets=['spreads', 'totals'],
                )
                date_odds_cache[game_date_key] = games_data
                await asyncio.sleep(0.3)
            except Exception as e:
                print(f'  Error fetching {game_date_key}: {e}')
                continue

        games_data = date_odds_cache.get(game_date_key, [])

        # Find spread and total for this game
        closing_spread = None
        closing_total = None

        for g in games_data:
            if g.get('home_team') == home_full and g.get('away_team') == away_full:
                for bm in g.get('bookmakers', [])[:1]:  # First bookmaker
                    for mkt in bm.get('markets', []):
                        if mkt.get('key') == 'spreads':
                            for out in mkt.get('outcomes', []):
                                if out.get('name') == home_full:
                                    closing_spread = out.get('point')
                        elif mkt.get('key') == 'totals':
                            for out in mkt.get('outcomes', []):
                                if out.get('name') == 'Over':
                                    closing_total = out.get('point')
                break

        if closing_spread is None and closing_total is None:
            continue

        games_with_lines += 1

        # Calculate results
        home_score = game.home_team_score
        away_score = game.away_team_score
        total_score = home_score + away_score
        actual_winner = home_abbr if home_score > away_score else away_abbr

        spread_result = None
        total_result = None

        if closing_spread is not None:
            spread_result = calculate_spread_result(home_score, away_score, closing_spread)
            # Track ATS for each team
            if spread_result == 'home_cover':
                team_ats[home_abbr].append('win')
                team_ats[away_abbr].append('loss')
            elif spread_result == 'away_cover':
                team_ats[home_abbr].append('loss')
                team_ats[away_abbr].append('win')
            else:
                team_ats[home_abbr].append('push')
                team_ats[away_abbr].append('push')

        if closing_total is not None:
            total_result = calculate_total_result(home_score, away_score, closing_total)
            # Track O/U for each team
            if total_result == 'over':
                team_ou[home_abbr].append('over')
                team_ou[away_abbr].append('over')
            elif total_result == 'under':
                team_ou[home_abbr].append('under')
                team_ou[away_abbr].append('under')
            else:
                team_ou[home_abbr].append('push')
                team_ou[away_abbr].append('push')

        # Generate a game_id
        game_id = f"{game.date.isoformat()}_{away_abbr}_{home_abbr}"

        # Insert or update game_results
        cur.execute('''
            INSERT INTO game_results (
                game_id, game_date, home_team_id, away_team_id,
                home_score, away_score, total_score,
                closing_spread, closing_total,
                actual_winner, spread_result, total_result
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (game_id) DO UPDATE SET
                home_score = EXCLUDED.home_score,
                away_score = EXCLUDED.away_score,
                total_score = EXCLUDED.total_score,
                closing_spread = EXCLUDED.closing_spread,
                closing_total = EXCLUDED.closing_total,
                actual_winner = EXCLUDED.actual_winner,
                spread_result = EXCLUDED.spread_result,
                total_result = EXCLUDED.total_result
        ''', (
            game_id, game.date, home_abbr, away_abbr,
            home_score, away_score, total_score,
            closing_spread, closing_total,
            actual_winner, spread_result, total_result
        ))

        games_processed += 1

        print(f'  {away_abbr} @ {home_abbr}: {away_score}-{home_score} | '
              f'spread {closing_spread} -> {spread_result} | '
              f'total {closing_total} -> {total_result} ({total_score})')

    # Calculate and update L10 records
    print('\n--- L10 Records ---')

    for abbr in sorted(set(team_ats.keys()) | set(team_ou.keys())):
        # ATS L10
        ats_results = team_ats.get(abbr, [])
        ats_last_10 = ats_results[-10:] if len(ats_results) >= 10 else ats_results
        ats_wins = sum(1 for r in ats_last_10 if r == 'win')
        ats_losses = sum(1 for r in ats_last_10 if r == 'loss')
        ats_pushes = sum(1 for r in ats_last_10 if r == 'push')
        ats_pct = ats_wins / (ats_wins + ats_losses) if (ats_wins + ats_losses) > 0 else 0.5

        # O/U L10
        ou_results = team_ou.get(abbr, [])
        ou_last_10 = ou_results[-10:] if len(ou_results) >= 10 else ou_results
        ou_overs = sum(1 for r in ou_last_10 if r == 'over')
        ou_unders = sum(1 for r in ou_last_10 if r == 'under')
        ou_pushes = sum(1 for r in ou_last_10 if r == 'push')

        print(f'{abbr}: ATS {ats_wins}-{ats_losses}-{ats_pushes} | O/U {ou_overs}-{ou_unders}-{ou_pushes}')

        # Update team_stats
        cur.execute('''
            UPDATE team_stats SET
                ats_wins_l10 = %s, ats_losses_l10 = %s, ats_pushes_l10 = %s, ats_pct_l10 = %s,
                ou_overs_l10 = %s, ou_unders_l10 = %s, ou_pushes_l10 = %s
            WHERE team_id = %s AND stat_date = %s
        ''', (
            ats_wins, ats_losses, ats_pushes, round(ats_pct, 3),
            ou_overs, ou_unders, ou_pushes,
            abbr, today
        ))

    cur.close()
    conn.close()

    return {
        'games_processed': games_processed,
        'games_with_lines': games_with_lines,
        'teams_updated': len(set(team_ats.keys()) | set(team_ou.keys())),
        'api_requests_remaining': odds_client.requests_remaining,
        'status': 'completed',
    }


if __name__ == '__main__':
    import sys
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    result = asyncio.run(backfill_game_results(days))
    print(f'\nResults: {result}')
