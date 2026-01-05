"""NBA Stats API client using nba_api library."""

import time
import structlog
from dataclasses import dataclass
from functools import lru_cache

import pandas as pd

logger = structlog.get_logger()

# Delay between requests to avoid rate limiting
REQUEST_DELAY = 0.6


@dataclass
class TeamAdvancedStats:
    """Team advanced statistics."""

    team_id: int
    team_name: str
    games_played: int
    wins: int
    losses: int
    off_rating: float  # Offensive rating
    def_rating: float  # Defensive rating
    net_rating: float  # Net rating
    pace: float  # Pace factor
    pie: float  # Player Impact Estimate


class NBAStatsClient:
    """Client for NBA.com statistics via nba_api."""

    def __init__(self, request_delay: float = REQUEST_DELAY):
        self.request_delay = request_delay

    def _delay(self):
        """Add delay between requests to avoid rate limiting."""
        time.sleep(self.request_delay)

    def get_team_advanced_stats(
        self,
        season: str = "2024-25",
        per_mode: str = "PerGame",
    ) -> list[TeamAdvancedStats]:
        """
        Get advanced team statistics for the season.

        Args:
            season: Season string (e.g., "2024-25")
            per_mode: Per game or totals

        Returns:
            List of TeamAdvancedStats
        """
        from nba_api.stats.endpoints import leaguedashteamstats

        self._delay()

        try:
            stats = leaguedashteamstats.LeagueDashTeamStats(
                season=season,
                measure_type_detailed_defense="Advanced",
                per_mode_detailed=per_mode,
            )

            df = stats.get_data_frames()[0]

            results = []
            for _, row in df.iterrows():
                results.append(
                    TeamAdvancedStats(
                        team_id=row["TEAM_ID"],
                        team_name=row["TEAM_NAME"],
                        games_played=row["GP"],
                        wins=row["W"],
                        losses=row["L"],
                        off_rating=row["OFF_RATING"],
                        def_rating=row["DEF_RATING"],
                        net_rating=row["NET_RATING"],
                        pace=row["PACE"],
                        pie=row["PIE"],
                    )
                )

            logger.info(f"Fetched advanced stats for {len(results)} teams")
            return results

        except Exception as e:
            logger.error(f"Failed to fetch team stats: {e}")
            raise

    def get_team_ratings_df(self, season: str = "2024-25") -> pd.DataFrame:
        """
        Get team offensive and defensive ratings as DataFrame.

        Args:
            season: Season string

        Returns:
            DataFrame with team ratings
        """
        from nba_api.stats.endpoints import leaguedashteamstats

        self._delay()

        stats = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            measure_type_detailed_defense="Advanced",
        )

        df = stats.get_data_frames()[0]

        return df[
            [
                "TEAM_NAME",
                "TEAM_ID",
                "OFF_RATING",
                "DEF_RATING",
                "NET_RATING",
                "PACE",
                "PIE",
            ]
        ].sort_values("NET_RATING", ascending=False)

    def get_player_game_logs(
        self,
        season: str = "2024-25",
        last_n_games: int = 10,
    ) -> pd.DataFrame:
        """Get recent player game logs."""
        from nba_api.stats.endpoints import playergamelogs

        self._delay()

        logs = playergamelogs.PlayerGameLogs(
            season_nullable=season,
            last_n_games_nullable=last_n_games,
        )

        return logs.get_data_frames()[0]

    def get_team_game_logs(
        self,
        team_id: int,
        season: str = "2024-25",
    ) -> pd.DataFrame:
        """Get game logs for a specific team."""
        from nba_api.stats.endpoints import teamgamelog

        self._delay()

        logs = teamgamelog.TeamGameLog(
            team_id=team_id,
            season=season,
        )

        return logs.get_data_frames()[0]

    def calculate_rolling_stats(
        self,
        team_id: int,
        windows: list[int] = [5, 10, 20],
        season: str = "2024-25",
    ) -> dict:
        """
        Calculate rolling statistics for a team.

        Args:
            team_id: NBA team ID
            windows: Rolling window sizes
            season: Season string

        Returns:
            Dict with rolling stats for each window
        """
        df = self.get_team_game_logs(team_id, season)

        if df.empty:
            return {}

        # Calculate basic metrics per game
        df["MARGIN"] = df["PTS"] - df["PTS"].shift(-1)  # Approximate

        rolling_stats = {}
        for window in windows:
            key = f"last_{window}"
            rolling_stats[key] = {
                "avg_pts": df["PTS"].head(window).mean(),
                "avg_margin": df["PLUS_MINUS"].head(window).mean(),
                "win_pct": df["WL"].head(window).apply(lambda x: 1 if x == "W" else 0).mean(),
            }

        return rolling_stats

    @staticmethod
    @lru_cache(maxsize=32)
    def get_team_id_by_abbreviation(abbreviation: str) -> int | None:
        """Get team ID from abbreviation."""
        from nba_api.stats.static import teams

        all_teams = teams.get_teams()
        for team in all_teams:
            if team["abbreviation"] == abbreviation:
                return team["id"]
        return None

    @staticmethod
    def get_all_teams() -> list[dict]:
        """Get all NBA teams."""
        from nba_api.stats.static import teams

        return teams.get_teams()


def test_nba_stats():
    """Test function to verify nba_api connection."""
    client = NBAStatsClient()

    print("Fetching all teams...")
    teams = client.get_all_teams()
    print(f"Found {len(teams)} teams")

    print("\nFetching team advanced stats...")
    try:
        stats = client.get_team_advanced_stats()
        print(f"Got stats for {len(stats)} teams")

        if stats:
            top = stats[0]
            print(f"\nTop team by query order: {top.team_name}")
            print(f"  ORtg: {top.off_rating:.1f}, DRtg: {top.def_rating:.1f}")
            print(f"  Net: {top.net_rating:.1f}, Pace: {top.pace:.1f}")
    except Exception as e:
        print(f"Error fetching stats: {e}")

    return teams


if __name__ == "__main__":
    test_nba_stats()
