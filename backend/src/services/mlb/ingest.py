"""MLB data ingestion orchestration."""

import structlog
from datetime import datetime, date, timezone, timedelta
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert

from src.models import (
    MLBTeam, MLBPitcher, MLBPitcherStats, MLBTeamStats,
    MLBGame, MLBGameContext, MLBMarket,
)
from src.services.mlb.mlb_api import MLBStatsAPIClient, MLBGameData, PARK_FACTORS, DOME_VENUES
from src.services.mlb.weather_api import WeatherAPIClient
from src.services.data.odds_api import OddsAPIClient

logger = structlog.get_logger()


# MLB team name mapping for odds API
MLB_TEAM_NAME_TO_ABBR = {
    "Arizona Diamondbacks": "ARI",
    "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC",
    "Chicago White Sox": "CWS",
    "Cincinnati Reds": "CIN",
    "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL",
    "Detroit Tigers": "DET",
    "Houston Astros": "HOU",
    "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA",
    "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN",
    "New York Mets": "NYM",
    "New York Yankees": "NYY",
    "Oakland Athletics": "OAK",
    "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SD",
    "San Francisco Giants": "SF",
    "Seattle Mariners": "SEA",
    "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TB",
    "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSH",
}


class MLBOddsClient(OddsAPIClient):
    """Odds API client configured for MLB."""

    SPORT = "baseball_mlb"

    async def get_mlb_odds(
        self,
        markets: list[str] = ["h2h", "spreads", "totals"],
        bookmakers: list[str] | None = None,
    ) -> list[dict]:
        """Fetch current MLB odds."""
        params = {
            "apiKey": self.api_key,
            "regions": "us",
            "markets": ",".join(markets),
            "oddsFormat": "decimal",
        }

        if bookmakers:
            params["bookmakers"] = ",".join(bookmakers)

        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.BASE_URL}/sports/{self.SPORT}/odds",
                params=params,
                timeout=30.0,
            )
            response.raise_for_status()

            self.requests_remaining = int(
                response.headers.get("x-requests-remaining", 0)
            )
            self.requests_used = int(response.headers.get("x-requests-used", 0))

            logger.info(
                "Fetched MLB odds",
                requests_remaining=self.requests_remaining,
            )

            return response.json()


class MLBDataIngestor:
    """Orchestrates MLB data ingestion from multiple sources."""

    def __init__(
        self,
        session: AsyncSession,
        mlb_client: MLBStatsAPIClient | None = None,
        weather_client: WeatherAPIClient | None = None,
        odds_client: MLBOddsClient | None = None,
    ):
        self.session = session
        self.mlb_client = mlb_client or MLBStatsAPIClient()
        self.weather_client = weather_client or WeatherAPIClient()
        self.odds_client = odds_client or MLBOddsClient()

    async def sync_teams(self) -> int:
        """Sync all MLB teams to database."""
        teams = await self.mlb_client.get_all_teams()

        count = 0
        for team in teams:
            stmt = insert(MLBTeam).values(
                team_abbr=team.abbr,
                team_name=team.name,
                league=team.league,
                division=team.division,
                external_id=team.team_id,
            ).on_conflict_do_update(
                index_elements=["team_abbr"],
                set_={
                    "team_name": team.name,
                    "league": team.league,
                    "division": team.division,
                    "external_id": team.team_id,
                },
            )
            await self.session.execute(stmt)
            count += 1

        await self.session.commit()
        logger.info("Synced MLB teams", count=count)
        return count

    async def ingest_games(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> int:
        """
        Ingest games from MLB Stats API.

        Args:
            start_date: Start date (defaults to today)
            end_date: End date (defaults to 7 days from start)

        Returns:
            Number of games ingested
        """
        if start_date is None:
            start_date = date.today()
        if end_date is None:
            end_date = start_date + timedelta(days=7)

        games = await self.mlb_client.get_schedule(
            start_date=start_date,
            end_date=end_date,
        )

        count = 0
        for game in games:
            await self._upsert_game(game)
            count += 1

        await self.session.commit()
        logger.info("Ingested MLB games", count=count, start=start_date.isoformat())
        return count

    async def _upsert_game(self, game: MLBGameData) -> None:
        """Insert or update a game record."""
        # First ensure pitchers exist
        home_starter_id = None
        away_starter_id = None

        if game.home_starter_id:
            home_starter_id = await self._ensure_pitcher(
                game.home_starter_id,
                game.home_starter_name,
                game.home_team,
            )

        if game.away_starter_id:
            away_starter_id = await self._ensure_pitcher(
                game.away_starter_id,
                game.away_starter_name,
                game.away_team,
            )

        stmt = insert(MLBGame).values(
            game_id=game.game_id,
            game_date=game.game_date,
            game_time=game.game_time,
            home_team=game.home_team,
            away_team=game.away_team,
            home_starter_id=home_starter_id,
            away_starter_id=away_starter_id,
            status=game.status,
            home_score=game.home_score,
            away_score=game.away_score,
            inning=game.inning,
            inning_state=game.inning_state,
            external_id=game.game_id,
            season=game.game_date.year,
            game_type=game.game_type,
        ).on_conflict_do_update(
            index_elements=["game_id"],
            set_={
                "home_starter_id": home_starter_id,
                "away_starter_id": away_starter_id,
                "status": game.status,
                "home_score": game.home_score,
                "away_score": game.away_score,
                "inning": game.inning,
                "inning_state": game.inning_state,
                "updated_at": datetime.now(timezone.utc),
            },
        )
        await self.session.execute(stmt)

        # Also update game context if venue available
        if game.venue_name:
            await self._upsert_game_context(game)

    async def _ensure_pitcher(
        self,
        external_id: int,
        name: str | None,
        team_abbr: str | None,
    ) -> int:
        """Ensure pitcher exists and return their ID."""
        # Check if pitcher exists
        stmt = select(MLBPitcher).where(MLBPitcher.external_id == str(external_id))
        result = await self.session.execute(stmt)
        pitcher = result.scalar_one_or_none()

        if pitcher:
            return pitcher.pitcher_id

        # Create new pitcher
        new_pitcher = MLBPitcher(
            player_name=name or "Unknown",
            team_abbr=team_abbr,
            external_id=str(external_id),
        )
        self.session.add(new_pitcher)
        await self.session.flush()  # Get the ID
        return new_pitcher.pitcher_id

    async def _upsert_game_context(self, game: MLBGameData) -> None:
        """Insert or update game context (venue, park factor)."""
        venue = game.venue_name or ""
        park_factor = PARK_FACTORS.get(venue, 1.0)
        is_dome = venue in DOME_VENUES
        roof_type = DOME_VENUES.get(venue)

        stmt = insert(MLBGameContext).values(
            game_id=game.game_id,
            venue_name=venue,
            venue_id=game.venue_id,
            park_factor=park_factor,
            is_dome=is_dome,
            is_retractable=roof_type == "retractable",
        ).on_conflict_do_update(
            index_elements=["game_id"],
            set_={
                "venue_name": venue,
                "park_factor": park_factor,
                "is_dome": is_dome,
                "updated_at": datetime.now(timezone.utc),
            },
        )
        await self.session.execute(stmt)

    async def ingest_weather(self, game_date: date | None = None) -> int:
        """
        Fetch and store weather for today's games.

        Args:
            game_date: Date to fetch weather for (defaults to today)

        Returns:
            Number of games with weather data
        """
        if game_date is None:
            game_date = date.today()

        # Get games for the date
        stmt = select(MLBGame, MLBGameContext).join(
            MLBGameContext, MLBGame.game_id == MLBGameContext.game_id
        ).where(
            and_(
                MLBGame.game_date == game_date,
                MLBGame.status == "scheduled",
            )
        )
        result = await self.session.execute(stmt)
        rows = result.all()

        count = 0
        for game, context in rows:
            if context.is_dome:
                # Skip domes - no weather impact
                continue

            if not game.game_time or not context.venue_name:
                continue

            weather = await self.weather_client.get_weather_for_game(
                context.venue_name,
                game.game_time,
            )

            if weather:
                weather_factor = self.weather_client.calculate_weather_factor(
                    weather, context.venue_name
                )

                # Update context
                context.temperature = weather.temperature
                context.wind_speed = weather.wind_speed
                context.wind_direction = str(weather.wind_direction)
                context.humidity = weather.humidity
                context.precipitation_pct = weather.precipitation_probability
                context.sky_condition = "clear" if weather.is_clear else "cloudy"
                context.weather_factor = weather_factor
                context.updated_at = datetime.now(timezone.utc)

                count += 1

        await self.session.commit()
        logger.info("Ingested weather data", count=count, date=game_date.isoformat())
        return count

    async def ingest_odds(self) -> int:
        """
        Fetch and store current MLB odds.

        Returns:
            Number of markets ingested
        """
        odds_data = await self.odds_client.get_mlb_odds()

        count = 0
        for game in odds_data:
            # Match to our game
            home_team = MLB_TEAM_NAME_TO_ABBR.get(game["home_team"])
            away_team = MLB_TEAM_NAME_TO_ABBR.get(game["away_team"])

            if not home_team or not away_team:
                logger.warning(
                    "Unknown team in odds",
                    home=game["home_team"],
                    away=game["away_team"],
                )
                continue

            # Find matching game
            game_time = datetime.fromisoformat(
                game["commence_time"].replace("Z", "+00:00")
            )
            game_date = game_time.date()

            stmt = select(MLBGame).where(
                and_(
                    MLBGame.home_team == home_team,
                    MLBGame.away_team == away_team,
                    MLBGame.game_date == game_date,
                )
            )
            result = await self.session.execute(stmt)
            mlb_game = result.scalar_one_or_none()

            if not mlb_game:
                logger.debug(
                    "No matching game for odds",
                    home=home_team,
                    away=away_team,
                    date=game_date.isoformat(),
                )
                continue

            # Process each bookmaker
            for bookmaker in game.get("bookmakers", []):
                book = bookmaker["key"]

                for market in bookmaker.get("markets", []):
                    market_key = market["key"]

                    if market_key == "h2h":
                        # Moneyline
                        await self._upsert_market_moneyline(
                            mlb_game.game_id, market, book
                        )
                        count += 1

                    elif market_key == "spreads":
                        # Runline
                        await self._upsert_market_runline(
                            mlb_game.game_id, market, book, home_team
                        )
                        count += 1

                    elif market_key == "totals":
                        # Over/Under
                        await self._upsert_market_total(
                            mlb_game.game_id, market, book
                        )
                        count += 1

        await self.session.commit()
        logger.info("Ingested MLB odds", count=count)
        return count

    async def _upsert_market_moneyline(
        self,
        game_id: str,
        market: dict,
        book: str,
    ) -> None:
        """Insert/update moneyline market."""
        home_odds = None
        away_odds = None

        for outcome in market.get("outcomes", []):
            if outcome.get("name") == "Home":
                home_odds = outcome["price"]
            else:
                away_odds = outcome["price"]

        # Find or create market
        stmt = select(MLBMarket).where(
            and_(
                MLBMarket.game_id == game_id,
                MLBMarket.market_type == "moneyline",
                MLBMarket.book == book,
            )
        )
        result = await self.session.execute(stmt)
        existing = result.scalar_one_or_none()

        if existing:
            existing.home_odds = home_odds
            existing.away_odds = away_odds
            existing.updated_at = datetime.now(timezone.utc)
        else:
            new_market = MLBMarket(
                game_id=game_id,
                market_type="moneyline",
                home_odds=home_odds,
                away_odds=away_odds,
                book=book,
            )
            self.session.add(new_market)

    async def _upsert_market_runline(
        self,
        game_id: str,
        market: dict,
        book: str,
        home_team: str,
    ) -> None:
        """Insert/update runline market."""
        line = None
        home_odds = None
        away_odds = None

        for outcome in market.get("outcomes", []):
            team_abbr = MLB_TEAM_NAME_TO_ABBR.get(outcome.get("name"), "")
            if team_abbr == home_team:
                line = outcome.get("point")
                home_odds = outcome["price"]
            else:
                away_odds = outcome["price"]

        stmt = select(MLBMarket).where(
            and_(
                MLBMarket.game_id == game_id,
                MLBMarket.market_type == "runline",
                MLBMarket.book == book,
            )
        )
        result = await self.session.execute(stmt)
        existing = result.scalar_one_or_none()

        if existing:
            existing.line = line
            existing.home_odds = home_odds
            existing.away_odds = away_odds
            existing.updated_at = datetime.now(timezone.utc)
        else:
            new_market = MLBMarket(
                game_id=game_id,
                market_type="runline",
                line=line,
                home_odds=home_odds,
                away_odds=away_odds,
                book=book,
            )
            self.session.add(new_market)

    async def _upsert_market_total(
        self,
        game_id: str,
        market: dict,
        book: str,
    ) -> None:
        """Insert/update total market."""
        line = None
        over_odds = None
        under_odds = None

        for outcome in market.get("outcomes", []):
            if outcome.get("name") == "Over":
                line = outcome.get("point")
                over_odds = outcome["price"]
            else:
                under_odds = outcome["price"]

        stmt = select(MLBMarket).where(
            and_(
                MLBMarket.game_id == game_id,
                MLBMarket.market_type == "total",
                MLBMarket.book == book,
            )
        )
        result = await self.session.execute(stmt)
        existing = result.scalar_one_or_none()

        if existing:
            existing.line = line
            existing.over_odds = over_odds
            existing.under_odds = under_odds
            existing.updated_at = datetime.now(timezone.utc)
        else:
            new_market = MLBMarket(
                game_id=game_id,
                market_type="total",
                line=line,
                over_odds=over_odds,
                under_odds=under_odds,
                book=book,
            )
            self.session.add(new_market)

    async def ingest_pitcher_stats(self, date_: date | None = None) -> int:
        """
        Fetch and store pitcher stats for all known pitchers.

        Args:
            date_: Stats date (defaults to today)

        Returns:
            Number of pitchers updated
        """
        if date_ is None:
            date_ = date.today()

        # Get all pitchers with external IDs
        stmt = select(MLBPitcher).where(MLBPitcher.external_id.isnot(None))
        result = await self.session.execute(stmt)
        pitchers = result.scalars().all()

        count = 0
        for pitcher in pitchers:
            try:
                stats = await self.mlb_client.get_pitcher_stats(
                    int(pitcher.external_id)
                )
                if stats:
                    await self._upsert_pitcher_stats(pitcher.pitcher_id, stats, date_)
                    count += 1
            except Exception as e:
                logger.warning(
                    "Failed to fetch pitcher stats",
                    pitcher_id=pitcher.pitcher_id,
                    error=str(e),
                )

        await self.session.commit()
        logger.info("Ingested pitcher stats", count=count)
        return count

    async def _upsert_pitcher_stats(
        self,
        pitcher_id: int,
        stats,
        stat_date: date,
    ) -> None:
        """Insert or update pitcher stats."""
        from src.services.mlb.pitcher_quality import PitcherQualityScorer

        quality_score = PitcherQualityScorer.calculate_quality_score(
            era=stats.era,
            whip=stats.whip,
            k_per_9=stats.k_per_9,
            bb_per_9=stats.bb_per_9,
        )

        stmt = insert(MLBPitcherStats).values(
            pitcher_id=pitcher_id,
            stat_date=stat_date,
            era=stats.era,
            whip=stats.whip,
            k_per_9=stats.k_per_9,
            bb_per_9=stats.bb_per_9,
            innings_pitched=stats.innings_pitched,
            games_started=stats.games_started,
            quality_score=quality_score,
        ).on_conflict_do_update(
            constraint="idx_mlb_pitcher_stats_unique",
            set_={
                "era": stats.era,
                "whip": stats.whip,
                "k_per_9": stats.k_per_9,
                "bb_per_9": stats.bb_per_9,
                "innings_pitched": stats.innings_pitched,
                "quality_score": quality_score,
            },
        )
        await self.session.execute(stmt)

    async def update_team_stats(self) -> int:
        """
        Update team rolling statistics from standings.

        Returns:
            Number of teams updated
        """
        standings = await self.mlb_client.get_team_standings()
        today = date.today()

        count = 0
        for record in standings:
            team_abbr = record.get("team_abbr")
            if not team_abbr:
                continue

            wins = record.get("wins", 0)
            losses = record.get("losses", 0)
            games = wins + losses

            if games == 0:
                continue

            runs_scored = record.get("runs_scored", 0)
            runs_allowed = record.get("runs_allowed", 0)

            stmt = insert(MLBTeamStats).values(
                team_abbr=team_abbr,
                stat_date=today,
                wins=wins,
                losses=losses,
                win_pct=record.get("win_pct"),
                runs_per_game=round(runs_scored / games, 2) if games > 0 else None,
                runs_allowed_per_game=round(runs_allowed / games, 2) if games > 0 else None,
                run_diff_per_game=round((runs_scored - runs_allowed) / games, 2) if games > 0 else None,
                home_wins=record.get("home_wins"),
                home_losses=record.get("home_losses"),
                away_wins=record.get("away_wins"),
                away_losses=record.get("away_losses"),
                last_10_wins=record.get("last_10_wins"),
                last_10_losses=record.get("last_10_losses"),
                last_10_record=f"{record.get('last_10_wins', 0)}-{record.get('last_10_losses', 0)}",
            ).on_conflict_do_update(
                constraint="idx_mlb_team_stats_unique",
                set_={
                    "wins": wins,
                    "losses": losses,
                    "win_pct": record.get("win_pct"),
                    "runs_per_game": round(runs_scored / games, 2) if games > 0 else None,
                    "runs_allowed_per_game": round(runs_allowed / games, 2) if games > 0 else None,
                    "run_diff_per_game": round((runs_scored - runs_allowed) / games, 2) if games > 0 else None,
                    "last_10_wins": record.get("last_10_wins"),
                    "last_10_losses": record.get("last_10_losses"),
                },
            )
            await self.session.execute(stmt)
            count += 1

        await self.session.commit()
        logger.info("Updated team stats", count=count)
        return count

    async def sync_results(self, game_date: date | None = None) -> int:
        """
        Sync final scores for completed games.

        Args:
            game_date: Date to sync (defaults to yesterday)

        Returns:
            Number of games updated
        """
        if game_date is None:
            game_date = date.today() - timedelta(days=1)

        games = await self.mlb_client.get_schedule(
            start_date=game_date,
            end_date=game_date,
        )

        count = 0
        for game in games:
            if game.status != "final":
                continue

            stmt = select(MLBGame).where(MLBGame.game_id == game.game_id)
            result = await self.session.execute(stmt)
            db_game = result.scalar_one_or_none()

            if db_game:
                db_game.status = "final"
                db_game.home_score = game.home_score
                db_game.away_score = game.away_score
                db_game.updated_at = datetime.now(timezone.utc)
                count += 1

        await self.session.commit()
        logger.info("Synced game results", count=count, date=game_date.isoformat())
        return count

    async def run_full_ingest(self, days_ahead: int = 7) -> dict:
        """
        Run complete data ingestion pipeline.

        Args:
            days_ahead: How many days ahead to fetch games

        Returns:
            Dict with counts of each type ingested
        """
        logger.info("Starting full MLB data ingest")

        results = {}

        # 1. Sync teams
        results["teams"] = await self.sync_teams()

        # 2. Ingest games
        results["games"] = await self.ingest_games(
            start_date=date.today(),
            end_date=date.today() + timedelta(days=days_ahead),
        )

        # 3. Update team stats
        results["team_stats"] = await self.update_team_stats()

        # 4. Ingest weather for today
        results["weather"] = await self.ingest_weather()

        # 5. Ingest odds
        results["odds"] = await self.ingest_odds()

        # 6. Sync yesterday's results
        results["results"] = await self.sync_results()

        logger.info("Completed full MLB data ingest", results=results)
        return results
