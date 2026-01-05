"""BALLDONTLIE API client for NBA statistics and injuries."""

import httpx
import structlog
from datetime import date, datetime
from dataclasses import dataclass

from src.config import settings

logger = structlog.get_logger()


@dataclass
class Team:
    """NBA team data."""

    id: int
    name: str
    full_name: str
    abbreviation: str
    city: str
    conference: str
    division: str


@dataclass
class Player:
    """NBA player data."""

    id: int
    first_name: str
    last_name: str
    position: str
    team_id: int
    jersey_number: str | None = None


@dataclass
class Game:
    """NBA game data."""

    id: int
    date: date
    season: int
    status: str
    home_team: Team
    away_team: Team
    home_team_score: int | None
    away_team_score: int | None


@dataclass
class PlayerStats:
    """Player box score stats."""

    player_id: int
    game_id: int
    minutes: str
    points: int
    rebounds: int
    assists: int
    steals: int
    blocks: int
    turnovers: int
    fg_made: int
    fg_attempted: int
    fg3_made: int
    fg3_attempted: int
    ft_made: int
    ft_attempted: int


@dataclass
class Injury:
    """Player injury report."""

    player_id: int
    player_name: str
    team_id: int
    team_name: str
    status: str  # Out, Doubtful, Questionable, Probable
    comment: str | None


class BallDontLieClient:
    """Client for BALLDONTLIE API."""

    BASE_URL = "https://api.balldontlie.io/v1"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or settings.balldontlie_api_key
        self.headers = {"Authorization": self.api_key}

    async def _get(self, endpoint: str, params: dict | None = None) -> dict:
        """Make authenticated GET request."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.BASE_URL}/{endpoint}",
                headers=self.headers,
                params=params or {},
                timeout=30.0,
            )
            response.raise_for_status()
            return response.json()

    async def get_teams(self) -> list[Team]:
        """Get all NBA teams."""
        data = await self._get("teams")
        return [
            Team(
                id=t["id"],
                name=t["name"],
                full_name=t["full_name"],
                abbreviation=t["abbreviation"],
                city=t["city"],
                conference=t["conference"],
                division=t["division"],
            )
            for t in data["data"]
        ]

    async def get_games(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
        team_ids: list[int] | None = None,
        seasons: list[int] | None = None,
    ) -> list[Game]:
        """
        Get games within date range.

        Args:
            start_date: Start date filter
            end_date: End date filter
            team_ids: Filter by team IDs
            seasons: Filter by seasons (e.g., [2024, 2025])
        """
        params = {}
        if start_date:
            params["start_date"] = start_date.isoformat()
        if end_date:
            params["end_date"] = end_date.isoformat()
        if team_ids:
            params["team_ids[]"] = team_ids
        if seasons:
            params["seasons[]"] = seasons

        data = await self._get("games", params)

        games = []
        for g in data["data"]:
            games.append(
                Game(
                    id=g["id"],
                    date=date.fromisoformat(g["date"][:10]),
                    season=g["season"],
                    status=g["status"],
                    home_team=Team(
                        id=g["home_team"]["id"],
                        name=g["home_team"]["name"],
                        full_name=g["home_team"]["full_name"],
                        abbreviation=g["home_team"]["abbreviation"],
                        city=g["home_team"]["city"],
                        conference=g["home_team"]["conference"],
                        division=g["home_team"]["division"],
                    ),
                    away_team=Team(
                        id=g["visitor_team"]["id"],
                        name=g["visitor_team"]["name"],
                        full_name=g["visitor_team"]["full_name"],
                        abbreviation=g["visitor_team"]["abbreviation"],
                        city=g["visitor_team"]["city"],
                        conference=g["visitor_team"]["conference"],
                        division=g["visitor_team"]["division"],
                    ),
                    home_team_score=g.get("home_team_score"),
                    away_team_score=g.get("visitor_team_score"),
                )
            )

        return games

    async def get_todays_games(self) -> list[Game]:
        """Get games scheduled for today."""
        today = date.today()
        return await self.get_games(start_date=today, end_date=today)

    async def get_player_stats(
        self,
        game_ids: list[int] | None = None,
        player_ids: list[int] | None = None,
        seasons: list[int] | None = None,
        per_page: int = 100,
    ) -> list[PlayerStats]:
        """Get player statistics."""
        params = {"per_page": per_page}
        if game_ids:
            params["game_ids[]"] = game_ids
        if player_ids:
            params["player_ids[]"] = player_ids
        if seasons:
            params["seasons[]"] = seasons

        data = await self._get("stats", params)

        stats = []
        for s in data["data"]:
            stats.append(
                PlayerStats(
                    player_id=s["player"]["id"],
                    game_id=s["game"]["id"],
                    minutes=s.get("min", "0"),
                    points=s.get("pts", 0),
                    rebounds=s.get("reb", 0),
                    assists=s.get("ast", 0),
                    steals=s.get("stl", 0),
                    blocks=s.get("blk", 0),
                    turnovers=s.get("turnover", 0),
                    fg_made=s.get("fgm", 0),
                    fg_attempted=s.get("fga", 0),
                    fg3_made=s.get("fg3m", 0),
                    fg3_attempted=s.get("fg3a", 0),
                    ft_made=s.get("ftm", 0),
                    ft_attempted=s.get("fta", 0),
                )
            )

        return stats

    async def get_injuries(self, team_id: int | None = None) -> list[Injury]:
        """
        Get current injury report.

        Note: Requires ALL-STAR tier or higher.
        """
        params = {}
        if team_id:
            params["team_ids[]"] = [team_id]

        try:
            data = await self._get("injuries", params)

            injuries = []
            for i in data["data"]:
                injuries.append(
                    Injury(
                        player_id=i["player"]["id"],
                        player_name=f"{i['player']['first_name']} {i['player']['last_name']}",
                        team_id=i["team"]["id"],
                        team_name=i["team"]["full_name"],
                        status=i.get("status", "Unknown"),
                        comment=i.get("comment"),
                    )
                )

            return injuries
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 403:
                logger.warning("Injuries endpoint requires paid tier")
                return []
            raise

    async def get_season_averages(
        self, player_ids: list[int], season: int = 2025
    ) -> list[dict]:
        """
        Get season averages for players.

        Note: Requires GOAT tier.
        """
        params = {
            "season": season,
            "player_ids[]": player_ids,
        }

        try:
            data = await self._get("season_averages", params)
            return data["data"]
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 403:
                logger.warning("Season averages endpoint requires GOAT tier")
                return []
            raise


async def test_balldontlie():
    """Test function to verify API connection."""
    client = BallDontLieClient()

    print("Fetching NBA teams...")
    teams = await client.get_teams()
    print(f"Found {len(teams)} teams")

    if teams:
        print(f"\nSample team: {teams[0].full_name} ({teams[0].abbreviation})")

    print("\nFetching today's games...")
    games = await client.get_todays_games()
    print(f"Found {len(games)} games today")

    for game in games[:3]:
        print(f"  {game.away_team.abbreviation} @ {game.home_team.abbreviation}")

    print("\nFetching injuries...")
    injuries = await client.get_injuries()
    print(f"Found {len(injuries)} injuries")

    for inj in injuries[:5]:
        print(f"  {inj.player_name} ({inj.team_name}): {inj.status}")

    return teams, games


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_balldontlie())
