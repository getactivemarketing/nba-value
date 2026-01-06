"""Team statistics calculation task."""

import asyncio
from datetime import datetime, timezone, timedelta, date
from decimal import Decimal
from collections import defaultdict

import structlog
from sqlalchemy import select, and_
from sqlalchemy.dialects.postgresql import insert

from src.celery_app import celery_app
from src.database import async_session_maker
from src.models import Game, Team, TeamStats

logger = structlog.get_logger()


async def _calculate_team_stats_async() -> dict:
    """
    Calculate rolling team statistics from completed games.

    For each team, calculates:
    - Overall record (W-L)
    - Last 10 win percentage
    - Points per game (season and L10)
    - Opponent PPG (season and L10)
    - Net rating approximation (PPG - Opp PPG)
    - Days rest and back-to-back detection
    - Home/Away win percentages
    """
    today = datetime.now(timezone.utc).date()
    season_start = date(2025, 10, 22)  # 2025-26 NBA season start

    teams_updated = 0
    errors = 0

    async with async_session_maker() as session:
        # Get all teams
        teams_query = select(Team)
        teams_result = await session.execute(teams_query)
        teams = list(teams_result.scalars().all())

        # Get all completed games this season
        games_query = (
            select(Game)
            .where(Game.status == "final")
            .where(Game.game_date >= season_start)
            .order_by(Game.game_date.desc())
        )
        games_result = await session.execute(games_query)
        all_games = list(games_result.scalars().all())

        logger.info(f"Processing {len(teams)} teams with {len(all_games)} completed games")

        # Group games by team
        team_games = defaultdict(list)
        for game in all_games:
            if game.home_score is not None and game.away_score is not None:
                # Add game from home team perspective
                team_games[game.home_team_id].append({
                    'date': game.game_date,
                    'is_home': True,
                    'points_for': game.home_score,
                    'points_against': game.away_score,
                    'won': game.home_score > game.away_score,
                })
                # Add game from away team perspective
                team_games[game.away_team_id].append({
                    'date': game.game_date,
                    'is_home': False,
                    'points_for': game.away_score,
                    'points_against': game.home_score,
                    'won': game.away_score > game.home_score,
                })

        for team in teams:
            try:
                games = team_games.get(team.team_id, [])

                if not games:
                    # No games yet, skip
                    continue

                # Sort by date (most recent first)
                games.sort(key=lambda x: x['date'], reverse=True)

                # Calculate overall record
                wins = sum(1 for g in games if g['won'])
                losses = len(games) - wins

                # Last 10 games
                last_10 = games[:10]
                wins_l10 = sum(1 for g in last_10 if g['won'])
                win_pct_10 = Decimal(str(wins_l10 / len(last_10))) if last_10 else None

                # Points per game (season)
                ppg_season = Decimal(str(sum(g['points_for'] for g in games) / len(games)))
                opp_ppg_season = Decimal(str(sum(g['points_against'] for g in games) / len(games)))

                # Points per game (last 10)
                ppg_10 = Decimal(str(sum(g['points_for'] for g in last_10) / len(last_10))) if last_10 else None
                opp_ppg_10 = Decimal(str(sum(g['points_against'] for g in last_10) / len(last_10))) if last_10 else None

                # Net rating approximation (using PPG differential as proxy)
                # Real net rating would need possession data
                net_rtg_season = ppg_season - opp_ppg_season if ppg_season and opp_ppg_season else None
                net_rtg_10 = ppg_10 - opp_ppg_10 if ppg_10 and opp_ppg_10 else None

                # Home/Away splits
                home_games = [g for g in games if g['is_home']]
                away_games = [g for g in games if not g['is_home']]

                home_wins = sum(1 for g in home_games if g['won'])
                away_wins = sum(1 for g in away_games if g['won'])

                home_win_pct = Decimal(str(home_wins / len(home_games))) if home_games else None
                away_win_pct = Decimal(str(away_wins / len(away_games))) if away_games else None

                # Rest days calculation
                days_rest = None
                is_b2b = False
                games_last_7 = 0

                if games:
                    last_game_date = games[0]['date']
                    if isinstance(last_game_date, datetime):
                        last_game_date = last_game_date.date()

                    days_rest = (today - last_game_date).days
                    is_b2b = days_rest <= 1

                    # Count games in last 7 days
                    week_ago = today - timedelta(days=7)
                    games_last_7 = sum(1 for g in games if g['date'] >= week_ago)

                # Upsert team stats for today
                stmt = insert(TeamStats).values(
                    team_id=team.team_id,
                    stat_date=today,
                    games_played=len(games),
                    wins=wins,
                    losses=losses,
                    win_pct_10=win_pct_10,
                    ppg_10=ppg_10,
                    ppg_season=ppg_season,
                    opp_ppg_10=opp_ppg_10,
                    opp_ppg_season=opp_ppg_season,
                    net_rtg_10=net_rtg_10,
                    net_rtg_season=net_rtg_season,
                    home_win_pct=home_win_pct,
                    away_win_pct=away_win_pct,
                    days_rest=days_rest,
                    is_back_to_back=is_b2b,
                    games_last_7_days=games_last_7,
                ).on_conflict_do_update(
                    index_elements=['team_id', 'stat_date'],
                    set_={
                        'games_played': len(games),
                        'wins': wins,
                        'losses': losses,
                        'win_pct_10': win_pct_10,
                        'ppg_10': ppg_10,
                        'ppg_season': ppg_season,
                        'opp_ppg_10': opp_ppg_10,
                        'opp_ppg_season': opp_ppg_season,
                        'net_rtg_10': net_rtg_10,
                        'net_rtg_season': net_rtg_season,
                        'home_win_pct': home_win_pct,
                        'away_win_pct': away_win_pct,
                        'days_rest': days_rest,
                        'is_back_to_back': is_b2b,
                        'games_last_7_days': games_last_7,
                    }
                )
                await session.execute(stmt)
                teams_updated += 1

                logger.debug(
                    "Updated team stats",
                    team=team.team_id,
                    record=f"{wins}-{losses}",
                    win_pct_10=float(win_pct_10) if win_pct_10 else None,
                )

            except Exception as e:
                logger.error("Failed to calculate stats for team", team=team.team_id, error=str(e))
                errors += 1

        await session.commit()

    return {
        "teams_updated": teams_updated,
        "errors": errors,
        "status": "completed",
    }


@celery_app.task(name="src.tasks.stats_calculation.calculate_team_stats")
def calculate_team_stats() -> dict:
    """
    Calculate and store rolling team statistics.

    Should run daily before odds ingestion to ensure fresh stats
    for the scoring pipeline.
    """
    logger.info("Starting team stats calculation")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(_calculate_team_stats_async())
    finally:
        loop.close()

    logger.info("Completed team stats calculation", **result)
    return result
