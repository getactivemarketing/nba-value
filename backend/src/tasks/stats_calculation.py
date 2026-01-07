"""Team statistics calculation task."""

import asyncio
from datetime import datetime, timezone, timedelta, date
from decimal import Decimal
from collections import defaultdict

import structlog
from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert

from src.celery_app import celery_app
from src.database import async_session_maker
from src.models import Team, TeamStats
from src.services.data.balldontlie import BallDontLieClient
from src.services.data.nba_stats import NBAStatsClient

logger = structlog.get_logger()

# Mapping from NBA.com team IDs to our abbreviations
NBA_TEAM_ID_TO_ABBR = {
    1610612737: "ATL", 1610612738: "BOS", 1610612739: "CLE", 1610612740: "NOP",
    1610612741: "CHI", 1610612742: "DAL", 1610612743: "DEN", 1610612744: "GSW",
    1610612745: "HOU", 1610612746: "LAC", 1610612747: "LAL", 1610612748: "MIA",
    1610612749: "MIL", 1610612750: "MIN", 1610612751: "BKN", 1610612752: "NYK",
    1610612753: "ORL", 1610612754: "IND", 1610612755: "PHI", 1610612756: "PHX",
    1610612757: "POR", 1610612758: "SAC", 1610612759: "SAS", 1610612760: "OKC",
    1610612761: "TOR", 1610612762: "UTA", 1610612763: "MEM", 1610612764: "WAS",
    1610612765: "DET", 1610612766: "CHA",
}


async def _calculate_team_stats_async() -> dict:
    """
    Calculate rolling team statistics from BallDontLie API.

    Fetches historical game data and calculates:
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

    # Fetch historical games from BallDontLie API
    client = BallDontLieClient()

    logger.info("Fetching historical games from BallDontLie API...")
    all_games = await client.get_games(
        start_date=season_start,
        end_date=today,
        seasons=[2025],  # 2025-26 season
    )

    # Filter to only completed games with scores
    completed_games = [
        g for g in all_games
        if g.status == "Final" and g.home_team_score and g.away_team_score
    ]

    logger.info(f"Found {len(completed_games)} completed games this season")

    if not completed_games:
        logger.warning("No completed games found - team stats cannot be calculated")
        return {
            "teams_updated": 0,
            "errors": 0,
            "status": "no_games",
            "message": "No completed games found this season",
        }

    async with async_session_maker() as session:
        # Get all teams from our database
        teams_query = select(Team)
        teams_result = await session.execute(teams_query)
        teams = list(teams_result.scalars().all())

        # Group games by team (using abbreviation as key)
        team_games = defaultdict(list)
        for game in completed_games:
            # Add game from home team perspective
            team_games[game.home_team.abbreviation].append({
                'date': game.date,
                'is_home': True,
                'points_for': game.home_team_score,
                'points_against': game.away_team_score,
                'won': game.home_team_score > game.away_team_score,
            })
            # Add game from away team perspective
            team_games[game.away_team.abbreviation].append({
                'date': game.date,
                'is_home': False,
                'points_for': game.away_team_score,
                'points_against': game.home_team_score,
                'won': game.away_team_score > game.home_team_score,
            })

        for team in teams:
            try:
                games = team_games.get(team.abbreviation, [])

                if not games:
                    # No games yet for this team, skip
                    continue

                # Sort by date (most recent first)
                games.sort(key=lambda x: x['date'], reverse=True)

                # Calculate overall record
                wins = sum(1 for g in games if g['won'])
                losses = len(games) - wins

                # Last 10 games
                last_10 = games[:10]
                wins_l10 = sum(1 for g in last_10 if g['won'])
                win_pct_10 = Decimal(str(round(wins_l10 / len(last_10), 3))) if last_10 else None

                # Points per game (season)
                ppg_season = Decimal(str(round(sum(g['points_for'] for g in games) / len(games), 2)))
                opp_ppg_season = Decimal(str(round(sum(g['points_against'] for g in games) / len(games), 2)))

                # Points per game (last 10)
                ppg_10 = Decimal(str(round(sum(g['points_for'] for g in last_10) / len(last_10), 2))) if last_10 else None
                opp_ppg_10 = Decimal(str(round(sum(g['points_against'] for g in last_10) / len(last_10), 2))) if last_10 else None

                # Net rating approximation (using PPG differential as proxy)
                net_rtg_season = ppg_season - opp_ppg_season if ppg_season and opp_ppg_season else None
                net_rtg_10 = ppg_10 - opp_ppg_10 if ppg_10 and opp_ppg_10 else None

                # Last 10 record (as wins-losses, not just percentage)
                losses_l10 = len(last_10) - wins_l10 if last_10 else None

                # Home/Away splits
                home_games = [g for g in games if g['is_home']]
                away_games = [g for g in games if not g['is_home']]

                home_wins = sum(1 for g in home_games if g['won'])
                home_losses = len(home_games) - home_wins
                away_wins = sum(1 for g in away_games if g['won'])
                away_losses = len(away_games) - away_wins

                home_win_pct = Decimal(str(round(home_wins / len(home_games), 3))) if home_games else None
                away_win_pct = Decimal(str(round(away_wins / len(away_games), 3))) if away_games else None

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
                    wins_l10=wins_l10,
                    losses_l10=losses_l10,
                    win_pct_10=win_pct_10,
                    home_wins=home_wins,
                    home_losses=home_losses,
                    away_wins=away_wins,
                    away_losses=away_losses,
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
                        'wins_l10': wins_l10,
                        'losses_l10': losses_l10,
                        'win_pct_10': win_pct_10,
                        'home_wins': home_wins,
                        'home_losses': home_losses,
                        'away_wins': away_wins,
                        'away_losses': away_losses,
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
                    team=team.abbreviation,
                    record=f"{wins}-{losses}",
                    win_pct_10=float(win_pct_10) if win_pct_10 else None,
                    net_rtg_10=float(net_rtg_10) if net_rtg_10 else None,
                )

            except Exception as e:
                logger.error("Failed to calculate stats for team", team=team.team_id, error=str(e))
                errors += 1

        await session.commit()

    return {
        "teams_updated": teams_updated,
        "games_processed": len(completed_games),
        "errors": errors,
        "status": "completed",
    }


async def _fetch_advanced_stats_async() -> dict:
    """
    Fetch advanced stats (ORtg, DRtg, Pace) from NBA.com and update team_stats.

    This supplements the basic stats from BallDontLie with advanced metrics.
    """
    today = datetime.now(timezone.utc).date()
    teams_updated = 0
    errors = 0

    try:
        # Fetch advanced stats from NBA.com
        nba_client = NBAStatsClient(request_delay=0.8)
        advanced_stats = nba_client.get_team_advanced_stats(season="2025-26")

        logger.info(f"Fetched advanced stats for {len(advanced_stats)} teams from NBA.com")

        # Build lookup by abbreviation
        stats_by_abbr = {}
        for stat in advanced_stats:
            abbr = NBA_TEAM_ID_TO_ABBR.get(stat.team_id)
            if abbr:
                stats_by_abbr[abbr] = stat

        async with async_session_maker() as session:
            # Get all teams
            teams_query = select(Team)
            teams_result = await session.execute(teams_query)
            teams = list(teams_result.scalars().all())

            for team in teams:
                try:
                    nba_stat = stats_by_abbr.get(team.abbreviation)
                    if not nba_stat:
                        logger.warning(f"No NBA stats found for {team.abbreviation}")
                        continue

                    # Update the team_stats record for today with advanced stats
                    stmt = (
                        update(TeamStats)
                        .where(TeamStats.team_id == team.team_id)
                        .where(TeamStats.stat_date == today)
                        .values(
                            ortg_10=Decimal(str(round(nba_stat.off_rating, 2))),
                            ortg_season=Decimal(str(round(nba_stat.off_rating, 2))),
                            drtg_10=Decimal(str(round(nba_stat.def_rating, 2))),
                            drtg_season=Decimal(str(round(nba_stat.def_rating, 2))),
                            pace_10=Decimal(str(round(nba_stat.pace, 2))),
                            pace_season=Decimal(str(round(nba_stat.pace, 2))),
                        )
                    )
                    result = await session.execute(stmt)

                    if result.rowcount > 0:
                        teams_updated += 1
                        logger.debug(
                            "Updated advanced stats",
                            team=team.abbreviation,
                            ortg=nba_stat.off_rating,
                            drtg=nba_stat.def_rating,
                            pace=nba_stat.pace,
                        )
                    else:
                        logger.warning(f"No team_stats row to update for {team.abbreviation}")

                except Exception as e:
                    logger.error(f"Failed to update advanced stats for {team.abbreviation}: {e}")
                    errors += 1

            await session.commit()

    except Exception as e:
        logger.error(f"Failed to fetch advanced stats from NBA.com: {e}")
        errors += 1

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
        # First calculate basic stats from BallDontLie
        result = loop.run_until_complete(_calculate_team_stats_async())

        # Then fetch advanced stats from NBA.com
        if result.get("teams_updated", 0) > 0:
            logger.info("Fetching advanced stats from NBA.com...")
            advanced_result = loop.run_until_complete(_fetch_advanced_stats_async())
            result["advanced_stats_updated"] = advanced_result.get("teams_updated", 0)
            result["advanced_errors"] = advanced_result.get("errors", 0)
    finally:
        loop.close()

    logger.info("Completed team stats calculation", **result)
    return result


@celery_app.task(name="src.tasks.stats_calculation.refresh_advanced_stats")
def refresh_advanced_stats() -> dict:
    """
    Refresh only the advanced stats from NBA.com.

    Useful for updating ORtg, DRtg, Pace without re-fetching basic stats.
    """
    logger.info("Refreshing advanced stats from NBA.com")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(_fetch_advanced_stats_async())
    finally:
        loop.close()

    logger.info("Completed advanced stats refresh", **result)
    return result
