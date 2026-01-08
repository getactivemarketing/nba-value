"""
Injury impact calculation service.

Fetches injury data from BallDontLie and calculates team-level injury impact
scores that can be used to adjust betting value calculations.

Position-aware weighting:
- Star players have higher impact than role players
- Position scarcity matters (2 centers out > 2 guards out if only 3 centers on roster)
- Centers impact totals more (rebounding = 2nd chance points, pace)
"""

import asyncio
from dataclasses import dataclass, field
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


# Position categories for grouping
def normalize_position(pos: str) -> str:
    """Normalize position string to G, F, or C."""
    pos = pos.upper().strip()
    if not pos:
        return "F"  # Default to forward if unknown
    if "C" in pos:
        return "C"
    if "G" in pos:
        return "G"
    return "F"


# Typical roster depth by position (used for scarcity calculation)
TYPICAL_ROSTER_DEPTH = {
    "G": 6,  # ~6 guards on typical roster
    "F": 5,  # ~5 forwards
    "C": 3,  # ~3 centers (scarcest position)
}

# Position impact multipliers for totals markets
# Centers affect rebounding/pace more, guards affect pace/3pt shooting
POSITION_TOTAL_IMPACT = {
    "C": 1.4,  # Centers have 40% more impact on totals (rebounding, rim protection)
    "F": 1.0,  # Forwards baseline
    "G": 0.9,  # Guards slightly less impact on totals (but more on spread/ML)
}


# Known star players with (impact, position)
# Impact = approximate points above replacement per game
STAR_PLAYERS: dict[str, tuple[float, str]] = {
    # Tier 1 - MVP caliber (~10+ win shares, huge impact)
    "Nikola Jokic": (8.0, "C"),
    "Giannis Antetokounmpo": (7.5, "F"),
    "Luka Doncic": (7.5, "G"),
    "Jayson Tatum": (7.0, "F"),
    "Shai Gilgeous-Alexander": (7.0, "G"),
    "Anthony Edwards": (6.5, "G"),
    "Joel Embiid": (7.0, "C"),
    "Kevin Durant": (6.5, "F"),
    "LeBron James": (6.0, "F"),
    "Stephen Curry": (6.5, "G"),

    # Tier 2 - All-Star caliber (~5-8 impact)
    "Donovan Mitchell": (5.5, "G"),
    "Trae Young": (5.5, "G"),
    "De'Aaron Fox": (5.5, "G"),
    "Tyrese Haliburton": (5.5, "G"),
    "Cade Cunningham": (5.0, "G"),
    "Ja Morant": (6.0, "G"),
    "Damian Lillard": (5.5, "G"),
    "Kyrie Irving": (5.0, "G"),
    "Jimmy Butler": (5.0, "F"),
    "Jaylen Brown": (5.5, "G"),
    "Paolo Banchero": (5.0, "F"),
    "Franz Wagner": (5.0, "F"),
    "Anthony Davis": (6.0, "C"),
    "Karl-Anthony Towns": (5.0, "C"),
    "Jalen Brunson": (5.5, "G"),
    "Domantas Sabonis": (5.0, "C"),
    "Bam Adebayo": (5.0, "C"),
    "Devin Booker": (5.5, "G"),
    "Bradley Beal": (4.5, "G"),
    "Zion Williamson": (5.0, "F"),
    "LaMelo Ball": (5.0, "G"),
    "Victor Wembanyama": (5.5, "C"),
    "Alperen Sengun": (4.5, "C"),
    "Fred VanVleet": (4.0, "G"),
    "Scottie Barnes": (5.0, "F"),

    # Tier 3 - Quality starters (~3-5 impact)
    "Khris Middleton": (4.0, "F"),
    "CJ McCollum": (4.0, "G"),
    "Tobias Harris": (3.5, "F"),
    "Jalen Duren": (3.5, "C"),
    "Austin Reaves": (3.5, "G"),
    "Jerami Grant": (3.5, "F"),
    "Jalen Suggs": (3.5, "G"),
    "Moritz Wagner": (3.0, "C"),
    "Josh Hart": (3.0, "G"),
    "Max Strus": (3.0, "G"),
    "Corey Kispert": (3.0, "F"),
    "Brandon Miller": (4.0, "F"),
    "Isaiah Hartenstein": (3.5, "C"),
    "Zach Edey": (3.5, "C"),
    "Keegan Murray": (4.0, "F"),
    "Bennedict Mathurin": (3.5, "G"),
    "Jonas Valanciunas": (3.5, "C"),
    "Walker Kessler": (3.0, "C"),
    "Jalen Green": (4.0, "G"),
    "Alex Caruso": (3.0, "G"),
    "Chris Paul": (3.0, "G"),
    "Dereck Lively II": (3.5, "C"),
    "Obi Toppin": (3.0, "F"),
    "Terry Rozier": (3.5, "G"),
    "Jakob Poeltl": (3.0, "C"),
    "Josh Giddey": (3.5, "G"),
    "Dejounte Murray": (4.5, "G"),
    "Herbert Jones": (3.0, "F"),
    "Trey Murphy III": (3.5, "F"),
    "Devin Vassell": (4.0, "G"),
    "Onyeka Okongwu": (3.0, "C"),
    "Rui Hachimura": (3.0, "F"),
    "Scoot Henderson": (3.5, "G"),
    "Grant Williams": (2.5, "F"),
    "Coby White": (3.5, "G"),
    "Nikola Vucevic": (4.0, "C"),
    "Rudy Gobert": (4.5, "C"),
    "Brook Lopez": (3.5, "C"),
    "Evan Mobley": (4.5, "C"),
    "Jaren Jackson Jr.": (4.5, "C"),
    "Myles Turner": (3.5, "C"),
    "Clint Capela": (3.0, "C"),
    "Robert Williams III": (3.5, "C"),
    "Mitchell Robinson": (3.0, "C"),
    "Daniel Gafford": (3.0, "C"),
    "Mark Williams": (3.0, "C"),
    "Nick Richards": (2.5, "C"),
}

# Default impact for unknown players (role players)
DEFAULT_PLAYER_IMPACT = 1.5


@dataclass
class PositionInjuryData:
    """Injury data broken down by position."""

    guards_out: int = 0
    forwards_out: int = 0
    centers_out: int = 0
    guard_impact: float = 0.0
    forward_impact: float = 0.0
    center_impact: float = 0.0

    @property
    def center_scarcity(self) -> float:
        """How scarce centers are (0-1, higher = more scarce)."""
        # If 2+ centers out of typical 3, that's severe
        return min(1.0, self.centers_out / TYPICAL_ROSTER_DEPTH["C"])

    @property
    def guard_scarcity(self) -> float:
        """How scarce guards are (0-1, higher = more scarce)."""
        return min(1.0, self.guards_out / TYPICAL_ROSTER_DEPTH["G"])


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

    # Position-specific data
    position_data: PositionInjuryData = field(default_factory=PositionInjuryData)

    # Market-specific scores (0-1 scale)
    spread_injury_score: float = 0.0  # For spread/ML bets
    totals_injury_score: float = 0.0  # For over/under bets (weighted toward centers)


@dataclass
class InjuryContext:
    """Full injury context for a game."""

    home_team: TeamInjuryReport
    away_team: TeamInjuryReport
    home_injury_edge: float  # Positive = home has advantage (opponent more injured)
    game_uncertainty: float  # Higher if key players are questionable

    # Market-specific edges
    home_spread_edge: float = 0.0  # Edge for spread/ML markets
    home_totals_edge: float = 0.0  # Edge for totals (positive = home healthier for scoring)


def get_player_impact(player_name: str, position: str = "") -> tuple[float, str]:
    """
    Get the impact value and position for a player.

    Returns:
        Tuple of (impact_value, position)
    """
    if player_name in STAR_PLAYERS:
        return STAR_PLAYERS[player_name]

    # For unknown players, use position from API if available
    norm_pos = normalize_position(position) if position else "F"
    return (DEFAULT_PLAYER_IMPACT, norm_pos)


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
    Calculate injury impact for a single team with position-aware weighting.

    Args:
        team_abbrev: Team abbreviation (e.g., "DET")
        injuries: List of injuries for this team

    Returns:
        TeamInjuryReport with impact calculations including position breakdown
    """
    players_out = []
    players_questionable = []
    total_impact = 0.0
    questionable_impact = 0.0

    # Position-specific tracking
    pos_data = PositionInjuryData()

    for inj in injuries:
        # Get impact and position (from our database or API)
        impact, position = get_player_impact(inj.player_name, inj.position)

        if inj.status == "Out":
            players_out.append(inj.player_name)
            total_impact += impact

            # Track by position
            if position == "G":
                pos_data.guards_out += 1
                pos_data.guard_impact += impact
            elif position == "C":
                pos_data.centers_out += 1
                pos_data.center_impact += impact
            else:  # F
                pos_data.forwards_out += 1
                pos_data.forward_impact += impact

        elif inj.status in ("Day-To-Day", "Questionable", "Doubtful"):
            players_questionable.append(inj.player_name)
            questionable_impact += impact

    # Calculate base injury score (0-1 scale)
    # Typical team might have 15 impact points of starters
    effective_impact = total_impact + (questionable_impact * 0.3)
    injury_score = min(1.0, effective_impact / 15.0)

    # Calculate spread/ML injury score (star impact matters most)
    # This is the standard calculation
    spread_injury_score = injury_score

    # Calculate totals injury score (center injuries matter MORE)
    # Centers affect rebounding = 2nd chance points, rim protection, pace
    # Apply position multipliers and scarcity bonus
    totals_weighted_impact = (
        pos_data.guard_impact * POSITION_TOTAL_IMPACT["G"] +
        pos_data.forward_impact * POSITION_TOTAL_IMPACT["F"] +
        pos_data.center_impact * POSITION_TOTAL_IMPACT["C"]
    )

    # Add scarcity bonus: if 2+ centers out, major impact on totals
    # This represents the "no backup center" catastrophic scenario
    center_scarcity_bonus = pos_data.center_scarcity * 3.0  # Up to 3 extra impact points

    totals_effective_impact = totals_weighted_impact + center_scarcity_bonus + (questionable_impact * 0.3)
    totals_injury_score = min(1.0, totals_effective_impact / 15.0)

    return TeamInjuryReport(
        team_id=ABBREV_TO_TEAM_ID.get(team_abbrev, 0),
        team_abbrev=team_abbrev,
        players_out=players_out,
        players_questionable=players_questionable,
        total_impact=total_impact,
        questionable_impact=questionable_impact,
        injury_score=injury_score,
        position_data=pos_data,
        spread_injury_score=spread_injury_score,
        totals_injury_score=totals_injury_score,
    )


async def get_game_injury_context(
    home_team: str,
    away_team: str,
    injuries_by_team: dict[str, list[Injury]] | None = None
) -> InjuryContext:
    """
    Get full injury context for a game with market-specific edges.

    Args:
        home_team: Home team abbreviation
        away_team: Away team abbreviation
        injuries_by_team: Pre-fetched injuries (optional, will fetch if not provided)

    Returns:
        InjuryContext with both team reports and market-specific edge calculations
    """
    if injuries_by_team is None:
        injuries_by_team = await fetch_all_injuries()

    home_injuries = injuries_by_team.get(home_team, [])
    away_injuries = injuries_by_team.get(away_team, [])

    home_report = calculate_team_injury_report(home_team, home_injuries)
    away_report = calculate_team_injury_report(away_team, away_injuries)

    # Calculate general edge: positive = home team has advantage
    # (opponent has more injury impact)
    home_injury_edge = away_report.injury_score - home_report.injury_score

    # Calculate spread/ML edge (based on overall star power lost)
    home_spread_edge = away_report.spread_injury_score - home_report.spread_injury_score

    # Calculate totals edge (weighted toward centers/rebounding)
    # Positive = home team healthier for scoring potential
    # For totals: we care about COMBINED injuries affecting total points
    home_totals_edge = away_report.totals_injury_score - home_report.totals_injury_score

    # Game uncertainty based on questionable players
    game_uncertainty = (
        home_report.questionable_impact + away_report.questionable_impact
    ) / 20.0  # Normalize to 0-1ish

    return InjuryContext(
        home_team=home_report,
        away_team=away_report,
        home_injury_edge=home_injury_edge,
        game_uncertainty=min(1.0, game_uncertainty),
        home_spread_edge=home_spread_edge,
        home_totals_edge=home_totals_edge,
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
        print("Fetching injury reports with position data...\n")
        reports = await get_all_team_injury_reports()

        # Sort by injury score
        sorted_teams = sorted(
            reports.values(),
            key=lambda r: r.injury_score,
            reverse=True
        )

        print("Teams by Injury Severity (Position-Aware):")
        print("=" * 70)
        for report in sorted_teams:
            if report.injury_score > 0:
                print(f"\n{report.team_abbrev}: spread={report.spread_injury_score:.2f}, totals={report.totals_injury_score:.2f}")
                pd = report.position_data
                print(f"  Position breakdown: G={pd.guards_out}({pd.guard_impact:.1f}), "
                      f"F={pd.forwards_out}({pd.forward_impact:.1f}), "
                      f"C={pd.centers_out}({pd.center_impact:.1f})")
                if pd.centers_out >= 2:
                    print(f"  ⚠️  CENTER SCARCITY: {pd.center_scarcity:.0%} of typical depth out!")
                if report.players_out:
                    print(f"  OUT: {', '.join(report.players_out[:5])}")
                if report.players_questionable:
                    print(f"  GTD: {', '.join(report.players_questionable[:3])}")

        # Test specific game context
        print("\n" + "=" * 70)
        print("\nExample: DET @ WAS game context:")
        context = await get_game_injury_context("WAS", "DET")
        print(f"  WAS: spread={context.home_team.spread_injury_score:.2f}, totals={context.home_team.totals_injury_score:.2f}")
        print(f"  DET: spread={context.away_team.spread_injury_score:.2f}, totals={context.away_team.totals_injury_score:.2f}")
        print(f"  Home spread edge: {context.home_spread_edge:+.2f}")
        print(f"  Home totals edge: {context.home_totals_edge:+.2f}")

    asyncio.run(main())
