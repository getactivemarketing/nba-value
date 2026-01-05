"""The Odds API client for fetching betting odds."""

import httpx
import structlog
from datetime import datetime, timezone
from typing import Literal
from dataclasses import dataclass

from src.config import settings

logger = structlog.get_logger()

# Market type mapping
MARKET_MAPPING = {
    "h2h": "moneyline",
    "spreads": "spread",
    "totals": "total",
}


@dataclass
class OddsSnapshot:
    """Parsed odds data from API."""

    game_id: str
    home_team: str
    away_team: str
    commence_time: datetime
    market_type: str
    outcome_label: str
    line: float | None
    odds_decimal: float
    book: str
    snapshot_time: datetime


class OddsAPIClient:
    """Client for The Odds API."""

    BASE_URL = "https://api.the-odds-api.com/v4"
    SPORT = "basketball_nba"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or settings.odds_api_key
        self.requests_remaining: int | None = None
        self.requests_used: int | None = None

    async def get_nba_odds(
        self,
        markets: list[str] = ["h2h", "spreads", "totals"],
        bookmakers: list[str] | None = None,
    ) -> list[dict]:
        """
        Fetch current NBA odds from US sportsbooks.

        Args:
            markets: List of markets - h2h (moneyline), spreads, totals
            bookmakers: Optional list of specific bookmakers

        Returns:
            List of games with odds from each bookmaker
        """
        params = {
            "apiKey": self.api_key,
            "regions": "us",
            "markets": ",".join(markets),
            "oddsFormat": "decimal",
        }

        if bookmakers:
            params["bookmakers"] = ",".join(bookmakers)

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.BASE_URL}/sports/{self.SPORT}/odds",
                params=params,
                timeout=30.0,
            )
            response.raise_for_status()

            # Track usage from headers
            self.requests_remaining = int(
                response.headers.get("x-requests-remaining", 0)
            )
            self.requests_used = int(response.headers.get("x-requests-used", 0))

            logger.info(
                "Fetched odds from API",
                requests_remaining=self.requests_remaining,
                requests_used=self.requests_used,
            )

            return response.json()

    async def get_upcoming_games(self) -> list[dict]:
        """Get list of upcoming NBA games."""
        params = {
            "apiKey": self.api_key,
        }

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.BASE_URL}/sports/{self.SPORT}/events",
                params=params,
                timeout=30.0,
            )
            response.raise_for_status()
            return response.json()

    def parse_odds_to_snapshots(
        self, games_data: list[dict]
    ) -> list[OddsSnapshot]:
        """
        Parse raw API response into structured OddsSnapshot objects.

        Args:
            games_data: Raw response from get_nba_odds()

        Returns:
            List of OddsSnapshot objects
        """
        snapshots = []
        snapshot_time = datetime.now(timezone.utc)

        for game in games_data:
            game_id = game["id"]
            home_team = game["home_team"]
            away_team = game["away_team"]
            commence_time = datetime.fromisoformat(
                game["commence_time"].replace("Z", "+00:00")
            )

            for bookmaker in game.get("bookmakers", []):
                book = bookmaker["key"]

                for market in bookmaker.get("markets", []):
                    market_key = market["key"]
                    market_type = MARKET_MAPPING.get(market_key, market_key)

                    for outcome in market.get("outcomes", []):
                        # Determine outcome label
                        outcome_name = outcome["name"]
                        if market_type == "moneyline":
                            outcome_label = f"{outcome_name}_ml"
                        elif market_type == "spread":
                            outcome_label = f"{outcome_name}_spread"
                        elif market_type == "total":
                            outcome_label = f"{outcome_name.lower()}"  # Over/Under
                        else:
                            outcome_label = outcome_name

                        snapshots.append(
                            OddsSnapshot(
                                game_id=game_id,
                                home_team=home_team,
                                away_team=away_team,
                                commence_time=commence_time,
                                market_type=market_type,
                                outcome_label=outcome_label,
                                line=outcome.get("point"),
                                odds_decimal=outcome["price"],
                                book=book,
                                snapshot_time=snapshot_time,
                            )
                        )

        return snapshots

    def find_best_odds(
        self, snapshots: list[OddsSnapshot]
    ) -> dict[str, OddsSnapshot]:
        """
        Find best odds across all books for each market.

        Args:
            snapshots: List of OddsSnapshot objects

        Returns:
            Dict mapping market_key to best OddsSnapshot
        """
        best_odds: dict[str, OddsSnapshot] = {}

        for snapshot in snapshots:
            key = f"{snapshot.game_id}_{snapshot.market_type}_{snapshot.outcome_label}"

            if key not in best_odds or snapshot.odds_decimal > best_odds[key].odds_decimal:
                best_odds[key] = snapshot

        return best_odds

    def calculate_market_consensus(
        self, snapshots: list[OddsSnapshot]
    ) -> dict[str, dict]:
        """
        Calculate consensus odds and variance across books.

        Args:
            snapshots: List of OddsSnapshot objects

        Returns:
            Dict with mean, std, and count per market
        """
        from collections import defaultdict
        import statistics

        market_odds: dict[str, list[float]] = defaultdict(list)

        for snapshot in snapshots:
            key = f"{snapshot.game_id}_{snapshot.market_type}_{snapshot.outcome_label}"
            market_odds[key].append(snapshot.odds_decimal)

        consensus = {}
        for key, odds_list in market_odds.items():
            consensus[key] = {
                "mean": statistics.mean(odds_list),
                "std": statistics.stdev(odds_list) if len(odds_list) > 1 else 0,
                "count": len(odds_list),
                "min": min(odds_list),
                "max": max(odds_list),
            }

        return consensus


async def test_odds_api():
    """Test function to verify API connection."""
    client = OddsAPIClient()

    print("Fetching upcoming NBA games...")
    games = await client.get_upcoming_games()
    print(f"Found {len(games)} upcoming games")

    if games:
        print("\nFetching odds...")
        odds_data = await client.get_nba_odds(
            markets=["h2h", "spreads", "totals"],
        )
        print(f"Fetched odds for {len(odds_data)} games")

        snapshots = client.parse_odds_to_snapshots(odds_data)
        print(f"Parsed {len(snapshots)} odds snapshots")

        if snapshots:
            print("\nSample snapshot:")
            s = snapshots[0]
            print(f"  {s.away_team} @ {s.home_team}")
            print(f"  Market: {s.market_type}, Outcome: {s.outcome_label}")
            print(f"  Line: {s.line}, Odds: {s.odds_decimal}, Book: {s.book}")

        print(f"\nAPI requests remaining: {client.requests_remaining}")

    return games, odds_data if games else None


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_odds_api())
