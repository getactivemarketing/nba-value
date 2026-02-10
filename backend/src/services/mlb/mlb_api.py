"""MLB Stats API client for fetching game, team, and pitcher data."""

import httpx
import structlog
from datetime import datetime, date, timezone
from dataclasses import dataclass
from typing import Any

logger = structlog.get_logger()


@dataclass
class MLBGameData:
    """Parsed game data from MLB Stats API."""
    game_id: str
    game_date: date
    game_time: datetime | None
    home_team: str
    away_team: str
    home_team_id: int
    away_team_id: int
    home_starter_id: int | None
    away_starter_id: int | None
    home_starter_name: str | None
    away_starter_name: str | None
    status: str
    home_score: int | None
    away_score: int | None
    venue_name: str | None
    venue_id: int | None
    inning: int | None
    inning_state: str | None
    game_type: str  # R=regular, P=playoff, S=spring


@dataclass
class MLBPitcherData:
    """Pitcher data from MLB Stats API."""
    pitcher_id: int
    name: str
    team_abbr: str | None
    throws: str | None
    era: float | None
    whip: float | None
    k_per_9: float | None
    bb_per_9: float | None
    fip: float | None
    innings_pitched: float | None
    games_started: int | None


@dataclass
class MLBTeamData:
    """Team data from MLB Stats API."""
    team_id: int
    abbr: str
    name: str
    league: str  # AL or NL
    division: str  # East, Central, West


# Team abbreviation mapping from MLB API team IDs
MLB_TEAM_ABBR = {
    108: "LAA", 109: "ARI", 110: "BAL", 111: "BOS", 112: "CHC",
    113: "CIN", 114: "CLE", 115: "COL", 116: "DET", 117: "HOU",
    118: "KC", 119: "LAD", 120: "WSH", 121: "NYM", 133: "OAK",
    134: "PIT", 135: "SD", 136: "SEA", 137: "SF", 138: "STL",
    139: "TB", 140: "TEX", 141: "TOR", 142: "MIN", 143: "PHI",
    144: "ATL", 145: "CWS", 146: "MIA", 147: "NYY", 158: "MIL",
}

# Park factors (run scoring environment - 1.0 = neutral)
PARK_FACTORS = {
    "Coors Field": 1.15,  # Colorado - extreme hitter's park
    "Chase Field": 1.08,  # Arizona
    "Fenway Park": 1.06,  # Boston
    "Great American Ball Park": 1.06,  # Cincinnati
    "Globe Life Field": 1.04,  # Texas
    "Citizens Bank Park": 1.03,  # Philadelphia
    "Target Field": 1.02,  # Minnesota
    "Yankee Stadium": 1.02,  # New York Yankees
    "Truist Park": 1.00,  # Atlanta
    "Busch Stadium": 0.99,  # St. Louis
    "Wrigley Field": 0.99,  # Chicago Cubs
    "Dodger Stadium": 0.99,  # Los Angeles Dodgers
    "Angel Stadium": 0.98,  # Los Angeles Angels
    "T-Mobile Park": 0.97,  # Seattle
    "PNC Park": 0.96,  # Pittsburgh
    "Kauffman Stadium": 0.96,  # Kansas City
    "loanDepot park": 0.95,  # Miami
    "Petco Park": 0.95,  # San Diego
    "Oracle Park": 0.94,  # San Francisco
    "Tropicana Field": 0.93,  # Tampa Bay
    "Oakland Coliseum": 0.93,  # Oakland
    "Comerica Park": 0.92,  # Detroit
    "Guaranteed Rate Field": 1.01,  # Chicago White Sox
    "Progressive Field": 1.00,  # Cleveland
    "Oriole Park at Camden Yards": 1.01,  # Baltimore
    "Rogers Centre": 1.00,  # Toronto
    "Citi Field": 0.98,  # New York Mets
    "Nationals Park": 1.00,  # Washington
    "Minute Maid Park": 1.02,  # Houston
    "American Family Field": 1.02,  # Milwaukee
}

# Dome/retractable roof venues
DOME_VENUES = {
    "Tropicana Field": "dome",
    "loanDepot park": "retractable",
    "Chase Field": "retractable",
    "Minute Maid Park": "retractable",
    "Rogers Centre": "retractable",
    "T-Mobile Park": "retractable",
    "American Family Field": "retractable",
    "Globe Life Field": "retractable",
}


class MLBStatsAPIClient:
    """Client for MLB Stats API (free, no key required)."""

    BASE_URL = "https://statsapi.mlb.com/api/v1"

    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout

    async def get_schedule(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
        game_types: list[str] | None = None,
    ) -> list[MLBGameData]:
        """
        Fetch MLB game schedule.

        Args:
            start_date: Start date (defaults to today)
            end_date: End date (defaults to start_date)
            game_types: Game types - R (regular), P (playoff), S (spring), etc.

        Returns:
            List of MLBGameData objects
        """
        if start_date is None:
            start_date = date.today()
        if end_date is None:
            end_date = start_date

        params = {
            "sportId": 1,  # MLB
            "startDate": start_date.isoformat(),
            "endDate": end_date.isoformat(),
            "hydrate": "probablePitcher,linescore,venue",
        }

        if game_types:
            params["gameTypes"] = ",".join(game_types)

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.BASE_URL}/schedule",
                params=params,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()

        games = []
        for game_date in data.get("dates", []):
            for game in game_date.get("games", []):
                parsed = self._parse_game(game)
                if parsed:
                    games.append(parsed)

        logger.info(
            "Fetched MLB schedule",
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            games_count=len(games),
        )

        return games

    def _parse_game(self, game: dict) -> MLBGameData | None:
        """Parse a single game from the schedule response."""
        try:
            game_pk = str(game["gamePk"])
            game_date_str = game["officialDate"]
            game_date = datetime.strptime(game_date_str, "%Y-%m-%d").date()

            # Parse game time
            game_time = None
            if game.get("gameDate"):
                game_time = datetime.fromisoformat(
                    game["gameDate"].replace("Z", "+00:00")
                )

            # Teams
            home_team_data = game["teams"]["home"]["team"]
            away_team_data = game["teams"]["away"]["team"]

            home_team_id = home_team_data["id"]
            away_team_id = away_team_data["id"]

            home_team = MLB_TEAM_ABBR.get(home_team_id, home_team_data.get("abbreviation", "UNK"))
            away_team = MLB_TEAM_ABBR.get(away_team_id, away_team_data.get("abbreviation", "UNK"))

            # Probable pitchers
            home_starter = game["teams"]["home"].get("probablePitcher", {})
            away_starter = game["teams"]["away"].get("probablePitcher", {})

            # Status
            status_code = game["status"]["statusCode"]
            if status_code in ("F", "FT"):
                status = "final"
            elif status_code in ("I", "IP"):
                status = "in_progress"
            else:
                status = "scheduled"

            # Scores
            home_score = game["teams"]["home"].get("score")
            away_score = game["teams"]["away"].get("score")

            # Linescore (inning info)
            linescore = game.get("linescore", {})
            inning = linescore.get("currentInning")
            inning_state = linescore.get("inningState")

            # Venue
            venue = game.get("venue", {})

            return MLBGameData(
                game_id=game_pk,
                game_date=game_date,
                game_time=game_time,
                home_team=home_team,
                away_team=away_team,
                home_team_id=home_team_id,
                away_team_id=away_team_id,
                home_starter_id=home_starter.get("id"),
                away_starter_id=away_starter.get("id"),
                home_starter_name=home_starter.get("fullName"),
                away_starter_name=away_starter.get("fullName"),
                status=status,
                home_score=home_score,
                away_score=away_score,
                venue_name=venue.get("name"),
                venue_id=venue.get("id"),
                inning=inning,
                inning_state=inning_state,
                game_type=game.get("gameType", "R"),
            )
        except Exception as e:
            logger.warning("Failed to parse game", error=str(e), game_pk=game.get("gamePk"))
            return None

    async def get_pitcher_stats(
        self,
        player_id: int,
        season: int | None = None,
    ) -> MLBPitcherData | None:
        """
        Fetch pitcher stats for a specific player.

        Args:
            player_id: MLB player ID
            season: Season year (defaults to current)

        Returns:
            MLBPitcherData object or None
        """
        if season is None:
            season = date.today().year

        params = {
            "stats": "season",
            "season": season,
            "group": "pitching",
        }

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.BASE_URL}/people/{player_id}/stats",
                params=params,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()

        # Also get player info
        player_response = await client.get(
            f"{self.BASE_URL}/people/{player_id}",
            timeout=self.timeout,
        )
        player_data = player_response.json()
        player_info = player_data.get("people", [{}])[0]

        stats_list = data.get("stats", [])
        if not stats_list:
            return None

        splits = stats_list[0].get("splits", [])
        if not splits:
            return None

        stat = splits[0].get("stat", {})
        team = splits[0].get("team", {})

        return MLBPitcherData(
            pitcher_id=player_id,
            name=player_info.get("fullName", "Unknown"),
            team_abbr=MLB_TEAM_ABBR.get(team.get("id")),
            throws=player_info.get("pitchHand", {}).get("code"),
            era=self._safe_float(stat.get("era")),
            whip=self._safe_float(stat.get("whip")),
            k_per_9=self._safe_float(stat.get("strikeoutsPer9Inn")),
            bb_per_9=self._safe_float(stat.get("walksPer9Inn")),
            fip=None,  # Not directly available, calculate separately
            innings_pitched=self._safe_float(stat.get("inningsPitched")),
            games_started=stat.get("gamesStarted"),
        )

    async def get_team_stats(
        self,
        team_id: int,
        season: int | None = None,
    ) -> dict[str, Any]:
        """
        Fetch team stats.

        Args:
            team_id: MLB team ID
            season: Season year

        Returns:
            Dict with team statistics
        """
        if season is None:
            season = date.today().year

        params = {
            "stats": "season",
            "season": season,
            "group": "hitting,pitching",
        }

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.BASE_URL}/teams/{team_id}/stats",
                params=params,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()

    async def get_team_standings(
        self,
        season: int | None = None,
    ) -> list[dict]:
        """
        Fetch current standings for all teams.

        Args:
            season: Season year

        Returns:
            List of standings records
        """
        if season is None:
            season = date.today().year

        params = {
            "leagueId": "103,104",  # AL and NL
            "season": season,
        }

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.BASE_URL}/standings",
                params=params,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()

        records = []
        for division in data.get("records", []):
            for team_record in division.get("teamRecords", []):
                team = team_record.get("team", {})
                records.append({
                    "team_id": team.get("id"),
                    "team_abbr": MLB_TEAM_ABBR.get(team.get("id")),
                    "wins": team_record.get("wins", 0),
                    "losses": team_record.get("losses", 0),
                    "win_pct": self._safe_float(team_record.get("winningPercentage")),
                    "runs_scored": team_record.get("runsScored", 0),
                    "runs_allowed": team_record.get("runsAllowed", 0),
                    "run_differential": team_record.get("runDifferential", 0),
                    "home_wins": team_record.get("records", {}).get("splitRecords", [{}])[0].get("wins", 0),
                    "home_losses": team_record.get("records", {}).get("splitRecords", [{}])[0].get("losses", 0),
                    "away_wins": team_record.get("records", {}).get("splitRecords", [{}])[1].get("wins", 0) if len(team_record.get("records", {}).get("splitRecords", [])) > 1 else 0,
                    "away_losses": team_record.get("records", {}).get("splitRecords", [{}])[1].get("losses", 0) if len(team_record.get("records", {}).get("splitRecords", [])) > 1 else 0,
                    "last_10_wins": team_record.get("records", {}).get("splitRecords", [{}])[2].get("wins", 0) if len(team_record.get("records", {}).get("splitRecords", [])) > 2 else 0,
                    "last_10_losses": team_record.get("records", {}).get("splitRecords", [{}])[2].get("losses", 0) if len(team_record.get("records", {}).get("splitRecords", [])) > 2 else 0,
                    "streak": team_record.get("streak", {}).get("streakCode", ""),
                })

        return records

    async def get_all_teams(self) -> list[MLBTeamData]:
        """Fetch all MLB teams."""
        params = {
            "sportId": 1,  # MLB
        }

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.BASE_URL}/teams",
                params=params,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()

        teams = []
        for team in data.get("teams", []):
            if team.get("sport", {}).get("id") != 1:
                continue

            league_data = team.get("league", {})
            division_data = team.get("division", {})

            # Determine league (AL/NL)
            league_name = league_data.get("name", "")
            if "American" in league_name:
                league = "AL"
            elif "National" in league_name:
                league = "NL"
            else:
                continue

            # Determine division
            division_name = division_data.get("name", "")
            if "East" in division_name:
                division = "East"
            elif "Central" in division_name:
                division = "Central"
            elif "West" in division_name:
                division = "West"
            else:
                division = "Unknown"

            teams.append(MLBTeamData(
                team_id=team["id"],
                abbr=MLB_TEAM_ABBR.get(team["id"], team.get("abbreviation", "UNK")),
                name=team.get("name", "Unknown"),
                league=league,
                division=division,
            ))

        return teams

    async def get_linescore(self, game_id: str) -> dict:
        """
        Fetch live linescore for a game.

        Args:
            game_id: Game PK

        Returns:
            Linescore data dict
        """
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.BASE_URL}/game/{game_id}/linescore",
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()

    def get_park_factor(self, venue_name: str) -> float:
        """Get park factor for a venue."""
        return PARK_FACTORS.get(venue_name, 1.0)

    def is_dome(self, venue_name: str) -> bool:
        """Check if venue is a dome or has retractable roof."""
        return venue_name in DOME_VENUES

    def get_roof_type(self, venue_name: str) -> str | None:
        """Get roof type for venue."""
        return DOME_VENUES.get(venue_name)

    def _safe_float(self, value: Any) -> float | None:
        """Safely convert to float."""
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None


async def test_mlb_api():
    """Test function to verify MLB API connection."""
    client = MLBStatsAPIClient()

    print("Fetching today's MLB schedule...")
    games = await client.get_schedule()
    print(f"Found {len(games)} games")

    if games:
        game = games[0]
        print(f"\nSample game: {game.away_team} @ {game.home_team}")
        print(f"  Date: {game.game_date}, Time: {game.game_time}")
        print(f"  Venue: {game.venue_name}")
        print(f"  Starters: {game.away_starter_name} vs {game.home_starter_name}")
        print(f"  Park Factor: {client.get_park_factor(game.venue_name or '')}")

        # Fetch pitcher stats if available
        if game.home_starter_id:
            print(f"\nFetching stats for {game.home_starter_name}...")
            pitcher = await client.get_pitcher_stats(game.home_starter_id)
            if pitcher:
                print(f"  ERA: {pitcher.era}, WHIP: {pitcher.whip}, K/9: {pitcher.k_per_9}")

    print("\nFetching standings...")
    standings = await client.get_team_standings()
    print(f"Found {len(standings)} team records")

    if standings:
        top = standings[0]
        print(f"  Sample: {top['team_abbr']} - {top['wins']}-{top['losses']}")

    return games


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_mlb_api())
