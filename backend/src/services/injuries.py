"""
Injury impact calculation service.

Fetches injury data from BallDontLie and calculates team-level injury impact
scores that can be used to adjust betting value calculations.
"""

import asyncio
from dataclasses import dataclass
from datetime import date

import structlog

from src.services.data.balldontlie import BallDontLieClient, Injury

logger = structlog.get_logger()

# Team ID to abbreviation mapping
TEAM_ID_TO_ABBREV = {
    1: 'ATL', 2: 'BOS', 3: 'BKN', 4: 'CHA', 5: 'CHI', 6: 'CLE', 7: 'DAL', 8: 'DEN',
    9: 'DET', 10: 'GSW', 11: 'HOU', 12: 'IND', 13: 'LAC', 14: 'LAL', 15: 'MEM',
    16: 'MIA', 17: 'MIL', 18: 'MIN', 19: 'NOP', 20: 'NYK', 21: 'OKC', 22: 'ORL',
    23: 'PHI', 24: 'PHX', 25: 'POR', 26: 'SAC', 27: 'SAS', 28: 'TOR', 29: 'UTA', 30: 'WAS'
}

ABBREV_TO_TEAM_ID = {v: k for k, v in TEAM_ID_TO_ABBREV.items()}

# Known star players with their approximate impact (points above replacement per game)
# These are approximate values based on typical star impact
STAR_PLAYERS = {
    # Tier 1 - MVP caliber (~10+ win shares, huge impact)
    "Nikola Jokic": 8.0,
    "Giannis Antetokounmpo": 7.5,
    "Luka Doncic": 7.5,
    "Jayson Tatum": 7.0,
    "Shai Gilgeous-Alexander": 7.0,
    "Anthony Edwards": 6.5,
    "Joel Embiid": 7.0,
    "Kevin Durant": 6.5,
    "LeBron James": 6.0,
    "Stephen Curry": 6.5,

    # Tier 2 - All-Star caliber (~5-8 impact)
    "Donovan Mitchell": 5.5,
    "Trae Young": 5.5,
    "De'Aaron Fox": 5.5,
    "Tyrese Haliburton": 5.5,
    "Cade Cunningham": 5.0,
    "Ja Morant": 6.0,
    "Damian Lillard": 5.5,
    "Kyrie Irving": 5.0,
    "Jimmy Butler": 5.0,
    "Jaylen Brown": 5.5,
    "Paolo Banchero": 5.0,
    "Franz Wagner": 5.0,
    "Anthony Davis": 6.0,
    "Karl-Anthony Towns": 5.0,
    "Jalen Brunson": 5.5,
    "Domantas Sabonis": 5.0,
    "Bam Adebayo": 5.0,
    "Devin Booker": 5.5,
    "Bradley Beal": 4.5,
    "Zion Williamson": 5.0,
    "LaMelo Ball": 5.0,
    "Victor Wembanyama": 5.5,
    "Alperen Sengun": 4.5,
    "Fred VanVleet": 4.0,
    "Scottie Barnes": 5.0,

    # Tier 3 - Quality starters (~3-5 impact)
    "Khris Middleton": 4.0,
    "CJ McCollum": 4.0,
    "Tobias Harris": 3.5,
    "Jalen Duren": 3.5,
    "Austin Reaves": 3.5,
    "Jerami Grant": 3.5,
    "Jalen Suggs": 3.5,
    "Moritz Wagner": 3.0,
    "Josh Hart": 3.0,
    "Max Strus": 3.0,
    "Corey Kispert": 3.0,
    "Brandon Miller": 4.0,
    "Isaiah Hartenstein": 3.5,
    "Zach Edey": 3.5,
    "Keegan Murray": 4.0,
    "Bennedict Mathurin": 3.5,
    "Jonas Valanciunas": 3.5,
    "Walker Kessler": 3.0,
    "Jalen Green": 4.0,
    "Alex Caruso": 3.0,
    "Chris Paul": 3.0,
    "Dereck Lively II": 3.5,
    "Obi Toppin": 3.0,
    "Terry Rozier": 3.5,
    "Jakob Poeltl": 3.0,
    "Josh Giddey": 3.5,
    "Dejounte Murray": 4.5,
    "Herbert Jones": 3.0,
    "Trey Murphy III": 3.5,
    "Devin Vassell": 4.0,
    "Onyeka Okongwu": 3.0,
    "Rui Hachimura": 3.0,
    "Scoot Henderson": 3.5,
    "Grant Williams": 2.5,
    "Coby White": 3.5,
}

# Default impact for unknown players (role players)
DEFAULT_PLAYER_IMPACT = 1.5


@dataclass
class TeamInjuryReport:
    """Injury report for a single team."""

    team_id: int
    team_abbrev: str
    players_out: list[str]
    players_questionable: list[str]
    total_impact: float  # Sum of impact points for OUT players
    questionable_impact: float  # Sum for questionable (weighted at 50%)
    injury_score: float  # 0-1 scale, higher = more injuries


@dataclass
class InjuryContext:
    """Full injury context for a game."""

    home_team: TeamInjuryReport
    away_team: TeamInjuryReport
    home_injury_edge: float  # Positive = home has advantage (opponent more injured)
    game_uncertainty: float  # Higher if key players are questionable


def get_player_impact(player_name: str) -> float:
    """Get the impact value for a player."""
    return STAR_PLAYERS.get(player_name, DEFAULT_PLAYER_IMPACT)


async def fetch_all_injuries() -> dict[str, list[Injury]]:
    """
    Fetch all current injuries grouped by team abbreviation.

    Returns:
        Dict mapping team abbreviation to list of injuries
    """
    client = BallDontLieClient()
    injuries = await client.get_injuries()

    by_team: dict[str, list[Injury]] = {}
    for inj in injuries:
        abbrev = TEAM_ID_TO_ABBREV.get(inj.team_id, "UNK")
        if abbrev not in by_team:
            by_team[abbrev] = []
        by_team[abbrev].append(inj)

    logger.info(f"Fetched {len(injuries)} injuries across {len(by_team)} teams")
    return by_team


def calculate_team_injury_report(
    team_abbrev: str,
    injuries: list[Injury]
) -> TeamInjuryReport:
    """
    Calculate injury impact for a single team.

    Args:
        team_abbrev: Team abbreviation (e.g., "DET")
        injuries: List of injuries for this team

    Returns:
        TeamInjuryReport with impact calculations
    """
    players_out = []
    players_questionable = []
    total_impact = 0.0
    questionable_impact = 0.0

    for inj in injuries:
        impact = get_player_impact(inj.player_name)

        if inj.status == "Out":
            players_out.append(inj.player_name)
            total_impact += impact
        elif inj.status in ("Day-To-Day", "Questionable", "Doubtful"):
            players_questionable.append(inj.player_name)
            questionable_impact += impact

    # Calculate injury score (0-1 scale)
    # Typical team might have 15 impact points of starters
    # So losing 10+ points is severe (~0.7+ score)
    effective_impact = total_impact + (questionable_impact * 0.3)
    injury_score = min(1.0, effective_impact / 15.0)

    return TeamInjuryReport(
        team_id=ABBREV_TO_TEAM_ID.get(team_abbrev, 0),
        team_abbrev=team_abbrev,
        players_out=players_out,
        players_questionable=players_questionable,
        total_impact=total_impact,
        questionable_impact=questionable_impact,
        injury_score=injury_score,
    )


async def get_game_injury_context(
    home_team: str,
    away_team: str,
    injuries_by_team: dict[str, list[Injury]] | None = None
) -> InjuryContext:
    """
    Get full injury context for a game.

    Args:
        home_team: Home team abbreviation
        away_team: Away team abbreviation
        injuries_by_team: Pre-fetched injuries (optional, will fetch if not provided)

    Returns:
        InjuryContext with both team reports and edge calculation
    """
    if injuries_by_team is None:
        injuries_by_team = await fetch_all_injuries()

    home_injuries = injuries_by_team.get(home_team, [])
    away_injuries = injuries_by_team.get(away_team, [])

    home_report = calculate_team_injury_report(home_team, home_injuries)
    away_report = calculate_team_injury_report(away_team, away_injuries)

    # Calculate edge: positive = home team has advantage
    # (opponent has more injury impact)
    home_injury_edge = away_report.injury_score - home_report.injury_score

    # Game uncertainty based on questionable players
    game_uncertainty = (
        home_report.questionable_impact + away_report.questionable_impact
    ) / 20.0  # Normalize to 0-1ish

    return InjuryContext(
        home_team=home_report,
        away_team=away_report,
        home_injury_edge=home_injury_edge,
        game_uncertainty=min(1.0, game_uncertainty),
    )


async def get_all_team_injury_reports() -> dict[str, TeamInjuryReport]:
    """
    Get injury reports for all teams.

    Returns:
        Dict mapping team abbreviation to TeamInjuryReport
    """
    injuries_by_team = await fetch_all_injuries()

    reports = {}
    for abbrev in TEAM_ID_TO_ABBREV.values():
        team_injuries = injuries_by_team.get(abbrev, [])
        reports[abbrev] = calculate_team_injury_report(abbrev, team_injuries)

    return reports


# CLI test
if __name__ == "__main__":
    async def main():
        print("Fetching injury reports...\n")
        reports = await get_all_team_injury_reports()

        # Sort by injury score
        sorted_teams = sorted(
            reports.values(),
            key=lambda r: r.injury_score,
            reverse=True
        )

        print("Teams by Injury Severity:")
        print("=" * 60)
        for report in sorted_teams:
            if report.injury_score > 0:
                print(f"\n{report.team_abbrev}: {report.injury_score:.2f} injury score")
                print(f"  Impact: {report.total_impact:.1f} pts (OUT)")
                if report.players_out:
                    print(f"  OUT: {', '.join(report.players_out[:5])}")
                if report.players_questionable:
                    print(f"  GTD: {', '.join(report.players_questionable[:3])}")

        # Test specific game context
        print("\n" + "=" * 60)
        print("\nExample: DET @ WAS game context:")
        context = await get_game_injury_context("WAS", "DET")
        print(f"  WAS injury score: {context.home_team.injury_score:.2f}")
        print(f"  DET injury score: {context.away_team.injury_score:.2f}")
        print(f"  Home (WAS) edge from injuries: {context.home_injury_edge:+.2f}")

    asyncio.run(main())
