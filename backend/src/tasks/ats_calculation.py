"""ATS (Against the Spread) calculation and historical backfill."""

import asyncio
from datetime import datetime, timezone, timedelta, date
from decimal import Decimal
from collections import defaultdict

import structlog
from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert

from src.celery_app import celery_app
from src.database import async_session_maker
from src.models import Team, TeamStats, Game
from src.services.data.odds_api import OddsAPIClient
from src.services.data.balldontlie import BallDontLieClient

logger = structlog.get_logger()

# Team name mapping from Odds API to our abbreviations
ODDS_API_TEAM_TO_ABBR = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}


def calculate_ats_result(
    home_score: int,
    away_score: int,
    home_spread: float,
) -> tuple[str, str]:
    """
    Calculate ATS result for both teams.

    Args:
        home_score: Home team final score
        away_score: Away team final score
        home_spread: Home team spread (negative = favorite)

    Returns:
        Tuple of (home_result, away_result) - each is 'win', 'loss', or 'push'
    """
    # Home team covers if: home_score + spread > away_score
    home_adjusted = home_score + home_spread

    if home_adjusted > away_score:
        return ("win", "loss")
    elif home_adjusted < away_score:
        return ("loss", "win")
    else:
        return ("push", "push")


async def fetch_historical_ats_for_game(
    client: OddsAPIClient,
    game_date: datetime,
    home_team: str,
    away_team: str,
) -> float | None:
    """
    Fetch closing spread for a specific game.

    Args:
        client: OddsAPIClient instance
        game_date: Game tip time
        home_team: Home team full name
        away_team: Away team full name

    Returns:
        Home team closing spread or None if not found
    """
    try:
        # Query 30 minutes before game time for closing line
        query_time = game_date - timedelta(minutes=30)

        games_data = await client.get_historical_odds(
            date=query_time,
            markets=["spreads"],
        )

        # Find matching game
        for game in games_data:
            if game.get("home_team") == home_team and game.get("away_team") == away_team:
                # Get consensus spread from first bookmaker
                for bookmaker in game.get("bookmakers", []):
                    for market in bookmaker.get("markets", []):
                        if market.get("key") == "spreads":
                            for outcome in market.get("outcomes", []):
                                if outcome.get("name") == home_team:
                                    return outcome.get("point")

        return None

    except Exception as e:
        logger.error(f"Failed to fetch historical odds: {e}")
        return None


async def backfill_ats_data(days_back: int = 30) -> dict:
    """
    Backfill ATS data for completed games.

    Args:
        days_back: Number of days to look back

    Returns:
        Summary of results
    """
    today = date.today()
    start_date = today - timedelta(days=days_back)

    games_processed = 0
    ats_results_found = 0
    errors = 0

    # Team ATS tracking
    team_ats: dict[str, list[str]] = defaultdict(list)  # team_abbr -> list of results

    # Fetch completed games from BallDontLie
    bdl_client = BallDontLieClient()
    odds_client = OddsAPIClient()

    logger.info(f"Fetching games from {start_date} to {today}...")

    all_games = await bdl_client.get_games(
        start_date=start_date,
        end_date=today,
        seasons=[2025],
    )

    completed_games = [
        g for g in all_games
        if g.status == "Final" and g.home_team_score and g.away_team_score
    ]

    logger.info(f"Found {len(completed_games)} completed games to process")

    for game in completed_games:
        try:
            # Convert team names to full names for Odds API lookup
            home_abbr = game.home_team.abbreviation
            away_abbr = game.away_team.abbreviation
            home_full = game.home_team.full_name
            away_full = game.away_team.full_name

            # Create datetime from game date (assume 7 PM ET tip for historical)
            game_dt = datetime.combine(
                game.date,
                datetime.min.time().replace(hour=19),
            ).replace(tzinfo=timezone.utc)

            # Fetch historical spread
            home_spread = await fetch_historical_ats_for_game(
                odds_client,
                game_dt,
                home_full,
                away_full,
            )

            if home_spread is not None:
                # Calculate ATS result
                home_result, away_result = calculate_ats_result(
                    game.home_team_score,
                    game.away_team_score,
                    home_spread,
                )

                team_ats[home_abbr].append(home_result)
                team_ats[away_abbr].append(away_result)

                ats_results_found += 1

                logger.debug(
                    f"{away_abbr} @ {home_abbr}: {game.away_team_score}-{game.home_team_score}, "
                    f"spread {home_spread}, home {home_result}"
                )

            games_processed += 1

            # Rate limit - wait between API calls
            await asyncio.sleep(0.5)

        except Exception as e:
            logger.error(f"Error processing game: {e}")
            errors += 1

    # Calculate L10 ATS records for each team
    team_ats_l10 = {}
    for abbr, results in team_ats.items():
        # Take last 10 results
        last_10 = results[-10:] if len(results) >= 10 else results
        wins = sum(1 for r in last_10 if r == "win")
        losses = sum(1 for r in last_10 if r == "loss")
        pushes = sum(1 for r in last_10 if r == "push")

        team_ats_l10[abbr] = {
            "wins": wins,
            "losses": losses,
            "pushes": pushes,
            "pct": wins / (wins + losses) if (wins + losses) > 0 else 0.5,
        }

        logger.info(f"{abbr} ATS L10: {wins}-{losses}-{pushes}")

    # Update team_stats with ATS data
    async with async_session_maker() as session:
        teams_query = select(Team)
        teams_result = await session.execute(teams_query)
        teams = list(teams_result.scalars().all())

        for team in teams:
            ats = team_ats_l10.get(team.abbreviation)
            if ats:
                stmt = (
                    update(TeamStats)
                    .where(TeamStats.team_id == team.team_id)
                    .where(TeamStats.stat_date == today)
                    .values(
                        ats_wins_l10=ats["wins"],
                        ats_losses_l10=ats["losses"],
                        ats_pushes_l10=ats["pushes"],
                        ats_pct_l10=Decimal(str(round(ats["pct"], 3))),
                    )
                )
                await session.execute(stmt)

        await session.commit()

    return {
        "games_processed": games_processed,
        "ats_results_found": ats_results_found,
        "teams_with_ats": len(team_ats_l10),
        "errors": errors,
        "api_requests_remaining": odds_client.requests_remaining,
        "status": "completed",
    }


@celery_app.task(name="src.tasks.ats_calculation.backfill_ats")
def backfill_ats(days_back: int = 30) -> dict:
    """
    Celery task to backfill ATS data.
    """
    logger.info(f"Starting ATS backfill for last {days_back} days")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(backfill_ats_data(days_back))
    finally:
        loop.close()

    logger.info("Completed ATS backfill", **result)
    return result


if __name__ == "__main__":
    # Run standalone
    import sys
    sys.path.insert(0, ".")

    days = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    result = asyncio.run(backfill_ats_data(days))
    print(f"\nResults: {result}")
