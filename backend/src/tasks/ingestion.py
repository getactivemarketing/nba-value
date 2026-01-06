"""Data ingestion tasks."""

import asyncio
from datetime import datetime, timezone, timedelta
from decimal import Decimal

import structlog
from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert

from src.celery_app import celery_app
from src.database import async_session_maker
from src.services.data.odds_api import OddsAPIClient
from src.services.data.balldontlie import BallDontLieClient
from src.models import Game as GameModel, Team as TeamModel, TeamStats, OddsSnapshot, Market

logger = structlog.get_logger()


def run_async(coro):
    """Helper to run async code in sync Celery task."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# Team abbreviation mapping (The Odds API -> our standard)
TEAM_ABBREV_MAP = {
    "Los Angeles Lakers": "LAL",
    "Los Angeles Clippers": "LAC",
    "Boston Celtics": "BOS",
    "New York Knicks": "NYK",
    "Brooklyn Nets": "BKN",
    "Philadelphia 76ers": "PHI",
    "Toronto Raptors": "TOR",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Detroit Pistons": "DET",
    "Indiana Pacers": "IND",
    "Milwaukee Bucks": "MIL",
    "Atlanta Hawks": "ATL",
    "Charlotte Hornets": "CHA",
    "Miami Heat": "MIA",
    "Orlando Magic": "ORL",
    "Washington Wizards": "WAS",
    "Denver Nuggets": "DEN",
    "Minnesota Timberwolves": "MIN",
    "Oklahoma City Thunder": "OKC",
    "Portland Trail Blazers": "POR",
    "Utah Jazz": "UTA",
    "Golden State Warriors": "GSW",
    "Phoenix Suns": "PHX",
    "Sacramento Kings": "SAC",
    "Dallas Mavericks": "DAL",
    "Houston Rockets": "HOU",
    "Memphis Grizzlies": "MEM",
    "New Orleans Pelicans": "NOP",
    "San Antonio Spurs": "SAS",
}


def get_team_abbrev(team_name: str) -> str:
    """Convert full team name to abbreviation."""
    return TEAM_ABBREV_MAP.get(team_name, team_name[:3].upper())


def calculate_implied_prob(odds1: float, odds2: float) -> tuple[float, float]:
    """Calculate de-vigged implied probabilities from two-way odds."""
    raw1 = 1 / odds1
    raw2 = 1 / odds2
    total = raw1 + raw2
    return raw1 / total, raw2 / total


@celery_app.task(name="src.tasks.ingestion.ingest_odds")
def ingest_odds() -> dict:
    """
    Ingest current odds from The Odds API.

    Runs every 15 minutes and:
    1. Fetches NBA odds for all active games
    2. Stores snapshots in odds_snapshots table
    3. Creates/updates games
    """
    logger.info("Starting odds ingestion")
    return run_async(_ingest_odds_async())


async def _ingest_odds_async() -> dict:
    """Async implementation of odds ingestion."""
    client = OddsAPIClient()

    try:
        # Fetch odds from API
        odds_data = await client.get_nba_odds(
            markets=["h2h", "spreads", "totals"]
        )

        if not odds_data:
            logger.info("No games with odds found")
            return {
                "games_fetched": 0,
                "snapshots_stored": 0,
                "api_requests_remaining": client.requests_remaining,
                "status": "no_games",
            }

        games_processed = 0
        snapshots_stored = 0
        snapshot_time = datetime.now(timezone.utc)

        async with async_session_maker() as session:
            for game_data in odds_data:
                game_id = game_data["id"]
                home_team = game_data["home_team"]
                away_team = game_data["away_team"]
                commence_time = datetime.fromisoformat(
                    game_data["commence_time"].replace("Z", "+00:00")
                )

                # Calculate minutes to tip
                minutes_to_tip = int((commence_time - snapshot_time).total_seconds() / 60)

                # Upsert game
                home_abbrev = get_team_abbrev(home_team)
                away_abbrev = get_team_abbrev(away_team)

                game_stmt = insert(GameModel).values(
                    game_id=game_id,
                    league="NBA",
                    season=2025,  # Current season
                    game_date=commence_time.date(),
                    tip_time_utc=commence_time,
                    home_team_id=home_abbrev,
                    away_team_id=away_abbrev,
                    status="scheduled" if minutes_to_tip > 0 else "in_progress",
                ).on_conflict_do_update(
                    index_elements=["game_id"],
                    set_={
                        "tip_time_utc": commence_time,
                        "updated_at": datetime.utcnow(),
                    }
                )
                await session.execute(game_stmt)
                games_processed += 1

                markets_created = 0

                # Process each bookmaker
                for bookmaker in game_data.get("bookmakers", []):
                    book_key = bookmaker["key"]

                    # Initialize snapshot data
                    snapshot_data = {
                        "game_id": game_id,
                        "book_key": book_key,
                        "snapshot_time": snapshot_time,
                        "minutes_to_tip": minutes_to_tip,
                        "market_type": "all",  # We'll store all markets in one row
                    }

                    for market in bookmaker.get("markets", []):
                        market_key = market["key"]
                        outcomes = {o["name"]: o for o in market.get("outcomes", [])}

                        if market_key == "h2h":
                            # Moneyline
                            home_ml = outcomes.get(home_team, {})
                            away_ml = outcomes.get(away_team, {})
                            if home_ml and away_ml:
                                snapshot_data["home_ml_odds"] = Decimal(str(home_ml.get("price", 0)))
                                snapshot_data["away_ml_odds"] = Decimal(str(away_ml.get("price", 0)))
                                # Calculate implied prob
                                if home_ml.get("price") and away_ml.get("price"):
                                    home_prob, _ = calculate_implied_prob(
                                        home_ml["price"], away_ml["price"]
                                    )
                                    snapshot_data["home_ml_prob"] = Decimal(str(round(home_prob, 4)))

                                    # Create Market records for home and away ML
                                    for side, ml_data in [("home", home_ml), ("away", away_ml)]:
                                        market_id = f"{game_id}_{book_key}_ml_{side}"
                                        market_stmt = insert(Market).values(
                                            market_id=market_id,
                                            game_id=game_id,
                                            market_type="moneyline",
                                            outcome_label=f"{side}_ml",
                                            line=None,
                                            odds_decimal=Decimal(str(ml_data.get("price", 1.91))),
                                            book=book_key,
                                            is_active=True,
                                        ).on_conflict_do_update(
                                            index_elements=["market_id"],
                                            set_={
                                                "odds_decimal": Decimal(str(ml_data.get("price", 1.91))),
                                                "updated_at": datetime.utcnow(),
                                            }
                                        )
                                        await session.execute(market_stmt)
                                        markets_created += 1

                        elif market_key == "spreads":
                            # Spread
                            home_spread = outcomes.get(home_team, {})
                            away_spread = outcomes.get(away_team, {})
                            if home_spread and away_spread:
                                snapshot_data["home_spread"] = Decimal(str(home_spread.get("point", 0)))
                                snapshot_data["home_spread_odds"] = Decimal(str(home_spread.get("price", 0)))
                                snapshot_data["away_spread"] = Decimal(str(away_spread.get("point", 0)))
                                snapshot_data["away_spread_odds"] = Decimal(str(away_spread.get("price", 0)))
                                # Calculate implied prob
                                if home_spread.get("price") and away_spread.get("price"):
                                    home_prob, _ = calculate_implied_prob(
                                        home_spread["price"], away_spread["price"]
                                    )
                                    snapshot_data["home_spread_prob"] = Decimal(str(round(home_prob, 4)))

                                    # Create Market records for home and away spread
                                    for side, spread_data in [("home", home_spread), ("away", away_spread)]:
                                        market_id = f"{game_id}_{book_key}_spread_{side}"
                                        market_stmt = insert(Market).values(
                                            market_id=market_id,
                                            game_id=game_id,
                                            market_type="spread",
                                            outcome_label=f"{side}_spread",
                                            line=Decimal(str(spread_data.get("point", 0))),
                                            odds_decimal=Decimal(str(spread_data.get("price", 1.91))),
                                            book=book_key,
                                            is_active=True,
                                        ).on_conflict_do_update(
                                            index_elements=["market_id"],
                                            set_={
                                                "line": Decimal(str(spread_data.get("point", 0))),
                                                "odds_decimal": Decimal(str(spread_data.get("price", 1.91))),
                                                "updated_at": datetime.utcnow(),
                                            }
                                        )
                                        await session.execute(market_stmt)
                                        markets_created += 1

                        elif market_key == "totals":
                            # Totals
                            over = outcomes.get("Over", {})
                            under = outcomes.get("Under", {})
                            if over and under:
                                snapshot_data["total_line"] = Decimal(str(over.get("point", 0)))
                                snapshot_data["over_odds"] = Decimal(str(over.get("price", 0)))
                                snapshot_data["under_odds"] = Decimal(str(under.get("price", 0)))
                                # Calculate implied prob
                                if over.get("price") and under.get("price"):
                                    over_prob, _ = calculate_implied_prob(
                                        over["price"], under["price"]
                                    )
                                    snapshot_data["over_prob"] = Decimal(str(round(over_prob, 4)))

                                    # Create Market records for over and under
                                    for side, total_data in [("over", over), ("under", under)]:
                                        market_id = f"{game_id}_{book_key}_total_{side}"
                                        market_stmt = insert(Market).values(
                                            market_id=market_id,
                                            game_id=game_id,
                                            market_type="total",
                                            outcome_label=side,
                                            line=Decimal(str(total_data.get("point", 0))),
                                            odds_decimal=Decimal(str(total_data.get("price", 1.91))),
                                            book=book_key,
                                            is_active=True,
                                        ).on_conflict_do_update(
                                            index_elements=["market_id"],
                                            set_={
                                                "line": Decimal(str(total_data.get("point", 0))),
                                                "odds_decimal": Decimal(str(total_data.get("price", 1.91))),
                                                "updated_at": datetime.utcnow(),
                                            }
                                        )
                                        await session.execute(market_stmt)
                                        markets_created += 1

                    # Store snapshot
                    snapshot_stmt = insert(OddsSnapshot).values(**snapshot_data)
                    await session.execute(snapshot_stmt)
                    snapshots_stored += 1

            await session.commit()

        result = {
            "games_fetched": games_processed,
            "snapshots_stored": snapshots_stored,
            "api_requests_remaining": client.requests_remaining,
            "status": "success",
        }

        logger.info("Completed odds ingestion", **result)
        return result

    except Exception as e:
        logger.error("Odds ingestion failed", error=str(e))
        return {
            "games_fetched": 0,
            "snapshots_stored": 0,
            "api_requests_remaining": client.requests_remaining,
            "status": "error",
            "error": str(e),
        }


@celery_app.task(name="src.tasks.ingestion.update_nba_stats")
def update_nba_stats() -> dict:
    """
    Update NBA team and player statistics.

    Runs daily and:
    1. Fetches latest team info and games from BALLDONTLIE
    2. Updates teams table
    3. Fetches game results for stats calculation
    """
    logger.info("Starting NBA stats update")
    return run_async(_update_nba_stats_async())


async def _update_nba_stats_async() -> dict:
    """Async implementation of stats update."""
    client = BallDontLieClient()

    try:
        teams_updated = 0
        games_fetched = 0

        async with async_session_maker() as session:
            # Fetch and store teams (returns list of Team dataclasses)
            teams_data = await client.get_teams()

            for team in teams_data:
                # team is a dataclass, access via attributes
                team_stmt = insert(TeamModel).values(
                    team_id=team.abbreviation,
                    external_id=team.id,
                    full_name=team.full_name,
                    abbreviation=team.abbreviation,
                    city=team.city,
                    name=team.name,
                    conference=team.conference,
                    division=team.division,
                ).on_conflict_do_update(
                    index_elements=["team_id"],
                    set_={
                        "external_id": team.id,
                        "full_name": team.full_name,
                        "updated_at": datetime.utcnow(),
                    }
                )
                await session.execute(team_stmt)
                teams_updated += 1

            # Fetch recent games for stats calculation
            today = datetime.now(timezone.utc).date()
            start_date = today - timedelta(days=30)  # Last 30 days

            games_data = await client.get_games(
                start_date=start_date,
                end_date=today,
            )
            games_fetched = len(games_data)

            # Update game results in our database
            for game in games_data:
                # game is a Game dataclass
                if game.status == "Final" and game.home_team_score:
                    # Update matching game in our database
                    stmt = (
                        update(GameModel)
                        .where(
                            GameModel.game_date == game.date,
                            GameModel.home_team_id == game.home_team.abbreviation,
                            GameModel.away_team_id == game.away_team.abbreviation,
                        )
                        .values(
                            home_score=game.home_team_score,
                            away_score=game.away_team_score,
                            status="final",
                            updated_at=datetime.utcnow(),
                        )
                    )
                    await session.execute(stmt)

            await session.commit()

        result = {
            "teams_updated": teams_updated,
            "games_fetched": games_fetched,
            "status": "success",
        }

        logger.info("Completed NBA stats update", **result)
        return result

    except Exception as e:
        logger.error("NBA stats update failed", error=str(e))
        return {
            "teams_updated": 0,
            "games_fetched": 0,
            "status": "error",
            "error": str(e),
        }


@celery_app.task(name="src.tasks.ingestion.check_injuries")
def check_injuries() -> dict:
    """
    Check for injury updates.

    Runs every 30 minutes and:
    1. Fetches latest injury report
    2. Updates injury status for affected players

    Note: Full implementation would require a paid injury data source.
    For now, this is a placeholder that could be extended.
    """
    logger.info("Starting injury check")

    # TODO: Implement with a real injury data source
    # Options: RotoWire API, FantasyLabs, or scraping ESPN

    result = {
        "injuries_found": 0,
        "changes_detected": 0,
        "status": "placeholder",
        "note": "Requires injury data source integration",
    }

    logger.info("Completed injury check", **result)
    return result


@celery_app.task(name="src.tasks.ingestion.backfill_historical_odds")
def backfill_historical_odds(start_date: str, end_date: str) -> dict:
    """
    Backfill historical odds data for backtesting.

    Manual task for loading historical data.
    Note: The Odds API historical data requires higher tier subscription.
    """
    logger.info("Starting historical odds backfill", start_date=start_date, end_date=end_date)

    # TODO: Implement historical backfill
    # This would require:
    # 1. Historical odds API endpoint
    # 2. Pagination through date range
    # 3. Bulk insert into odds_snapshots

    return {
        "status": "not_implemented",
        "start_date": start_date,
        "end_date": end_date,
        "note": "Requires historical data API access",
    }


@celery_app.task(name="src.tasks.ingestion.sync_game_results")
def sync_game_results() -> dict:
    """
    Sync final game results for completed games.

    Updates games table with final scores for any games
    that have completed since last sync.
    """
    logger.info("Starting game results sync")
    return run_async(_sync_game_results_async())


async def _sync_game_results_async() -> dict:
    """Async implementation of game results sync."""
    client = BallDontLieClient()

    try:
        async with async_session_maker() as session:
            # Find games that are scheduled or in_progress
            stmt = select(GameModel).where(GameModel.status.in_(["scheduled", "in_progress"]))
            result = await session.execute(stmt)
            pending_games = result.scalars().all()

            if not pending_games:
                return {"games_updated": 0, "status": "no_pending_games"}

            # Get today's games from API
            today = datetime.now(timezone.utc).date()
            yesterday = today - timedelta(days=1)

            games_data = await client.get_games(
                start_date=yesterday,
                end_date=today,
            )

            games_updated = 0
            for api_game in games_data:
                # api_game is a Game dataclass
                if api_game.status == "Final" and api_game.home_team_score:
                    # Update matching game
                    stmt = (
                        update(GameModel)
                        .where(
                            GameModel.game_date == api_game.date,
                            GameModel.home_team_id == api_game.home_team.abbreviation,
                            GameModel.away_team_id == api_game.away_team.abbreviation,
                            GameModel.status != "final",
                        )
                        .values(
                            home_score=api_game.home_team_score,
                            away_score=api_game.away_team_score,
                            status="final",
                            updated_at=datetime.utcnow(),
                        )
                    )
                    result = await session.execute(stmt)
                    if result.rowcount > 0:
                        games_updated += 1

            await session.commit()

            return {
                "games_updated": games_updated,
                "status": "success",
            }

    except Exception as e:
        logger.error("Game results sync failed", error=str(e))
        return {
            "games_updated": 0,
            "status": "error",
            "error": str(e),
        }
