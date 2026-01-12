"""
Stats-based injury impact calculation service.

Calculates team-level injury impact by:
1. Fetching injured players and their actual season averages
2. Calculating % of team production lost in each category
3. Applying recency decay (recent injury = more impact, team hasn't adjusted)

Key metrics:
- scoring_impact: PPG lost as % of team average (~110 PPG)
- rebounding_impact: RPG lost as % of team average (~44 RPG)
- playmaking_impact: APG lost as % of team average (~25 APG)
- defense_impact: (SPG + BPG) lost as % of team average (~12 combined)
"""

import asyncio
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

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

# Typical NBA team totals per game (for calculating % impact)
TEAM_AVERAGES = {
    'ppg': 113.0,   # Points per game
    'rpg': 43.0,    # Rebounds per game
    'apg': 26.0,    # Assists per game
    'spg': 7.5,     # Steals per game
    'bpg': 5.0,     # Blocks per game
    'mpg': 240.0,   # Total minutes (5 players * 48 min)
}


def calculate_star_multiplier(ppg: float, apg: float, position: str) -> float:
    """
    Calculate star player multiplier based on production and role.

    Star players have outsized impact beyond their raw stats because:
    1. Usage in crunch time - stars get the ball in big moments
    2. Defensive attention - teams scheme around them, opening teammates
    3. Gravity - their presence creates spacing and open shots
    4. Leadership - experience and confidence in pressure situations

    Args:
        ppg: Points per game
        apg: Assists per game
        position: Player position (G, F, C)

    Returns:
        Multiplier from 1.0 (role player) to 1.6 (superstar)
    """
    # Base multiplier starts at 1.0
    multiplier = 1.0

    # PPG-based star impact
    # - 25+ PPG = superstar (1.4x)
    # - 20-25 PPG = star (1.25x)
    # - 15-20 PPG = solid starter (1.1x)
    # - <15 PPG = role player (1.0x)
    if ppg >= 25:
        multiplier += 0.40  # Superstar impact
    elif ppg >= 20:
        multiplier += 0.25  # Star impact
    elif ppg >= 15:
        multiplier += 0.10  # Quality starter impact

    # Playmakers get extra weight - point guards orchestrate offense
    # - 8+ APG = elite playmaker (+0.15)
    # - 5-8 APG = good playmaker (+0.08)
    if apg >= 8:
        multiplier += 0.15
    elif apg >= 5:
        multiplier += 0.08

    # Position scarcity bonus
    # - Guards (especially PGs) harder to replace at high level
    # - Their ball-handling/decision-making is scarce
    if position == 'G' and apg >= 5:
        multiplier += 0.05

    return min(1.60, multiplier)  # Cap at 1.6x


@dataclass
class PlayerInjuryImpact:
    """Individual player's injury impact based on their stats."""

    player_id: int
    player_name: str
    position: str
    status: str  # Out, Day-To-Day, Questionable

    # Season averages
    ppg: float = 0.0
    rpg: float = 0.0
    apg: float = 0.0
    spg: float = 0.0
    bpg: float = 0.0
    mpg: float = 0.0
    games_played: int = 0

    # Recency info
    days_out: int = 0  # How long they've been out
    recency_weight: float = 1.0  # 1.0 = full impact, decays over time

    @property
    def star_multiplier(self) -> float:
        """Get star player multiplier based on production level."""
        return calculate_star_multiplier(self.ppg, self.apg, self.position)

    @property
    def is_star(self) -> bool:
        """Is this player considered a star (20+ PPG or elite playmaker)?"""
        return self.ppg >= 20 or (self.apg >= 8 and self.ppg >= 15)

    @property
    def scoring_impact(self) -> float:
        """PPG lost, weighted by recency and star factor."""
        return self.ppg * self.recency_weight * self.status_weight * self.star_multiplier

    @property
    def rebounding_impact(self) -> float:
        """RPG lost, weighted by recency and star factor."""
        return self.rpg * self.recency_weight * self.status_weight * self.star_multiplier

    @property
    def playmaking_impact(self) -> float:
        """APG lost, weighted by recency and star factor."""
        return self.apg * self.recency_weight * self.status_weight * self.star_multiplier

    @property
    def defense_impact(self) -> float:
        """(SPG + BPG) lost, weighted by recency and star factor."""
        return (self.spg + self.bpg) * self.recency_weight * self.status_weight * self.star_multiplier

    @property
    def minutes_impact(self) -> float:
        """Minutes per game lost."""
        return self.mpg * self.recency_weight * self.status_weight

    @property
    def status_weight(self) -> float:
        """Weight based on injury status."""
        if self.status == "Out":
            return 1.0
        elif self.status == "Doubtful":
            return 0.80
        elif self.status in ("Questionable", "Day-To-Day"):
            return 0.50  # Increased from 0.4 - Q players miss more often
        elif self.status == "Probable":
            return 0.15
        return 0.0


@dataclass
class TeamInjuryReport:
    """Comprehensive injury report for a team based on actual stats."""

    team_id: int
    team_abbrev: str

    # Injured players with their impacts
    injured_players: list[PlayerInjuryImpact] = field(default_factory=list)

    # Aggregate stats lost
    total_ppg_lost: float = 0.0
    total_rpg_lost: float = 0.0
    total_apg_lost: float = 0.0
    total_defense_lost: float = 0.0
    total_minutes_lost: float = 0.0

    # Impact scores (0-1 scale, % of team production lost)
    scoring_impact: float = 0.0      # For spread/ML
    rebounding_impact: float = 0.0   # For totals (2nd chance pts)
    playmaking_impact: float = 0.0   # For spread/ML
    defense_impact: float = 0.0      # For totals (opponent scoring)

    # Overall scores for scoring system
    spread_injury_score: float = 0.0   # Weighted toward scoring/playmaking
    totals_injury_score: float = 0.0   # Weighted toward rebounding/defense
    injury_score: float = 0.0          # General score (backwards compatible)

    @property
    def players_out(self) -> list[str]:
        """List of player names who are OUT."""
        return [p.player_name for p in self.injured_players if p.status == "Out"]

    @property
    def players_questionable(self) -> list[str]:
        """List of player names who are questionable/GTD."""
        return [p.player_name for p in self.injured_players
                if p.status in ("Questionable", "Day-To-Day", "Doubtful")]


@dataclass
class InjuryContext:
    """Full injury context for a game."""

    home_team: TeamInjuryReport
    away_team: TeamInjuryReport

    # Edges (positive = home advantage)
    home_injury_edge: float = 0.0
    home_spread_edge: float = 0.0
    home_totals_edge: float = 0.0
    game_uncertainty: float = 0.0


def calculate_recency_weight(days_out: int) -> float:
    """
    Calculate recency weight based on how long player has been out.

    Recent injury = full impact (team hasn't adjusted)
    Long-term injury = reduced impact (team has adjusted lineup/rotation)

    Args:
        days_out: Number of days player has been out

    Returns:
        Weight from 0.4 to 1.0
    """
    if days_out <= 7:
        return 1.0      # First week: full impact
    elif days_out <= 14:
        return 0.85     # Week 2: 85% impact
    elif days_out <= 21:
        return 0.70     # Week 3: 70% impact
    elif days_out <= 28:
        return 0.55     # Week 4: 55% impact
    else:
        return 0.40     # 4+ weeks: 40% impact (team fully adjusted)


async def fetch_all_injuries() -> dict[str, list[Injury]]:
    """Fetch all current injuries grouped by team abbreviation."""
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


async def get_player_stats(player_ids: list[int], client: BallDontLieClient) -> dict[int, dict]:
    """Fetch season averages for multiple players (current season only)."""
    if not player_ids:
        return {}

    # Only use current season (2025-26 = 2025)
    # If no data, player has been out all season = team fully adjusted = 0 impact
    return await client.get_player_season_averages_batch(player_ids, season=2025)


def calculate_team_injury_report(
    team_abbrev: str,
    injuries: list[Injury],
    player_stats: dict[int, dict],
) -> TeamInjuryReport:
    """
    Calculate comprehensive injury impact for a team using actual player stats.

    Args:
        team_abbrev: Team abbreviation (e.g., "LAL")
        injuries: List of injuries for this team
        player_stats: Dict mapping player_id to their season averages

    Returns:
        TeamInjuryReport with all impact calculations
    """
    injured_players = []

    total_ppg = 0.0
    total_rpg = 0.0
    total_apg = 0.0
    total_defense = 0.0
    total_minutes = 0.0

    for inj in injuries:
        # Get player's season averages
        stats = player_stats.get(inj.player_id, {})

        if not stats:
            # No stats available - use minimal default
            ppg, rpg, apg, spg, bpg, mpg = 2.0, 1.0, 0.5, 0.2, 0.1, 8.0
            games_played = 0
        else:
            ppg = stats.get('pts', 0) or 0
            rpg = stats.get('reb', 0) or 0
            apg = stats.get('ast', 0) or 0
            spg = stats.get('stl', 0) or 0
            bpg = stats.get('blk', 0) or 0
            games_played = stats.get('games_played', 0) or 0

            # Parse minutes (format: "32:15" or just number)
            min_str = stats.get('min', '0')
            if isinstance(min_str, str) and ':' in min_str:
                parts = min_str.split(':')
                mpg = float(parts[0]) + float(parts[1]) / 60
            else:
                mpg = float(min_str) if min_str else 0

        # Determine injury duration and impact
        # Key insight: if player hasn't played this season, team is fully adjusted
        is_season_ending = "season" in inj.status.lower() if inj.status else False

        if games_played == 0:
            # No games this season = out all year = team fully adjusted = 0 impact
            days_out = 120
            recency_weight = 0.0
        elif is_season_ending:
            # Out for season - team has had significant time to adjust
            days_out = 60
            recency_weight = 0.15
        else:
            # Active player who got injured - estimate recency based on games played
            # ~40 games expected by mid-season
            expected_games = 40
            games_missed_pct = max(0, (expected_games - games_played) / expected_games)

            if games_missed_pct > 0.7:
                # Missed >70% of season - team adjusted
                days_out = 45
                recency_weight = 0.30
            elif games_missed_pct > 0.4:
                # Missed 40-70% - partially adjusted
                days_out = 21
                recency_weight = 0.55
            else:
                # Recent injury, team not adjusted
                days_out = 7
                recency_weight = calculate_recency_weight(days_out)

        player_impact = PlayerInjuryImpact(
            player_id=inj.player_id,
            player_name=inj.player_name,
            position=inj.position,
            status=inj.status,
            ppg=ppg,
            rpg=rpg,
            apg=apg,
            spg=spg,
            bpg=bpg,
            mpg=mpg,
            games_played=games_played,
            days_out=days_out,
            recency_weight=recency_weight,
        )

        injured_players.append(player_impact)

        # Accumulate totals
        total_ppg += player_impact.scoring_impact
        total_rpg += player_impact.rebounding_impact
        total_apg += player_impact.playmaking_impact
        total_defense += player_impact.defense_impact
        total_minutes += player_impact.minutes_impact

    # Calculate impact scores as % of team production
    scoring_impact = min(1.0, total_ppg / TEAM_AVERAGES['ppg'])
    rebounding_impact = min(1.0, total_rpg / TEAM_AVERAGES['rpg'])
    playmaking_impact = min(1.0, total_apg / TEAM_AVERAGES['apg'])
    defense_impact = min(1.0, total_defense / (TEAM_AVERAGES['spg'] + TEAM_AVERAGES['bpg']))

    # Calculate composite scores for betting markets
    # Spread/ML: scoring and playmaking matter most
    spread_injury_score = (
        scoring_impact * 0.50 +      # Scoring is 50%
        playmaking_impact * 0.30 +   # Playmaking is 30%
        rebounding_impact * 0.10 +   # Rebounding is 10%
        defense_impact * 0.10        # Defense is 10%
    )

    # Totals: rebounding and defense matter more (affects pace, 2nd chance pts)
    totals_injury_score = (
        rebounding_impact * 0.35 +   # Rebounding is 35% (2nd chance pts, pace)
        defense_impact * 0.25 +      # Defense is 25% (opponent scoring)
        scoring_impact * 0.25 +      # Scoring is 25%
        playmaking_impact * 0.15     # Playmaking is 15%
    )

    # General score (average of both)
    injury_score = (spread_injury_score + totals_injury_score) / 2

    return TeamInjuryReport(
        team_id=ABBREV_TO_TEAM_ID.get(team_abbrev, 0),
        team_abbrev=team_abbrev,
        injured_players=injured_players,
        total_ppg_lost=total_ppg,
        total_rpg_lost=total_rpg,
        total_apg_lost=total_apg,
        total_defense_lost=total_defense,
        total_minutes_lost=total_minutes,
        scoring_impact=scoring_impact,
        rebounding_impact=rebounding_impact,
        playmaking_impact=playmaking_impact,
        defense_impact=defense_impact,
        spread_injury_score=spread_injury_score,
        totals_injury_score=totals_injury_score,
        injury_score=injury_score,
    )


async def get_all_team_injury_reports() -> dict[str, TeamInjuryReport]:
    """
    Get comprehensive injury reports for all teams using actual player stats.

    Returns:
        Dict mapping team abbreviation to TeamInjuryReport
    """
    client = BallDontLieClient()

    # Fetch all injuries
    injuries_by_team = await fetch_all_injuries()

    # Collect all injured player IDs
    all_player_ids = []
    for injuries in injuries_by_team.values():
        for inj in injuries:
            all_player_ids.append(inj.player_id)

    # Fetch stats for all injured players
    logger.info(f"Fetching stats for {len(all_player_ids)} injured players...")
    player_stats = await get_player_stats(all_player_ids, client)
    logger.info(f"Got stats for {len(player_stats)} players")

    # Build reports for all teams
    reports = {}
    for abbrev in TEAM_ID_TO_ABBREV.values():
        team_injuries = injuries_by_team.get(abbrev, [])
        reports[abbrev] = calculate_team_injury_report(abbrev, team_injuries, player_stats)

    return reports


async def get_game_injury_context(
    home_team: str,
    away_team: str,
    injury_reports: dict[str, TeamInjuryReport] | None = None,
) -> InjuryContext:
    """
    Get full injury context for a game.

    Args:
        home_team: Home team abbreviation
        away_team: Away team abbreviation
        injury_reports: Pre-fetched reports (optional)

    Returns:
        InjuryContext with both team reports and edge calculations
    """
    if injury_reports is None:
        injury_reports = await get_all_team_injury_reports()

    home_report = injury_reports.get(home_team, TeamInjuryReport(
        team_id=ABBREV_TO_TEAM_ID.get(home_team, 0),
        team_abbrev=home_team,
    ))
    away_report = injury_reports.get(away_team, TeamInjuryReport(
        team_id=ABBREV_TO_TEAM_ID.get(away_team, 0),
        team_abbrev=away_team,
    ))

    # Calculate edges (positive = home has advantage)
    home_injury_edge = away_report.injury_score - home_report.injury_score
    home_spread_edge = away_report.spread_injury_score - home_report.spread_injury_score
    home_totals_edge = away_report.totals_injury_score - home_report.totals_injury_score

    # Game uncertainty based on questionable players
    home_q = sum(1 for p in home_report.injured_players if p.status in ("Questionable", "Day-To-Day"))
    away_q = sum(1 for p in away_report.injured_players if p.status in ("Questionable", "Day-To-Day"))
    game_uncertainty = min(1.0, (home_q + away_q) / 10.0)

    return InjuryContext(
        home_team=home_report,
        away_team=away_report,
        home_injury_edge=home_injury_edge,
        home_spread_edge=home_spread_edge,
        home_totals_edge=home_totals_edge,
        game_uncertainty=game_uncertainty,
    )


# CLI test
if __name__ == "__main__":
    async def main():
        print("Fetching stats-based injury reports (with star player weighting)...\n")
        reports = await get_all_team_injury_reports()

        # Sort by injury score
        sorted_teams = sorted(
            reports.values(),
            key=lambda r: r.injury_score,
            reverse=True
        )

        print("Teams by Injury Impact (Stats-Based + Star Weighting):")
        print("=" * 80)

        for report in sorted_teams:
            if report.injury_score > 0.05:  # Only show teams with meaningful injuries
                # Count star players out
                stars_out = [p for p in report.injured_players if p.is_star and p.status == "Out"]
                star_label = f" [{len(stars_out)} STAR{'S' if len(stars_out) != 1 else ''} OUT]" if stars_out else ""

                print(f"\n{report.team_abbrev}: spread={report.spread_injury_score:.2f}, "
                      f"totals={report.totals_injury_score:.2f}{star_label}")
                print(f"  Lost: {report.total_ppg_lost:.1f} PPG, {report.total_rpg_lost:.1f} RPG, "
                      f"{report.total_apg_lost:.1f} APG, {report.total_defense_lost:.1f} D")

                # Show top injured players by impact
                top_players = sorted(report.injured_players,
                                    key=lambda p: p.scoring_impact, reverse=True)[:3]
                for p in top_players:
                    if p.ppg > 0:
                        star_marker = " *STAR*" if p.is_star else ""
                        mult_str = f" [x{p.star_multiplier:.2f}]" if p.star_multiplier > 1.0 else ""
                        print(f"    {p.player_name} ({p.status}): {p.ppg:.1f}ppg, "
                              f"{p.rpg:.1f}rpg, {p.apg:.1f}apg{mult_str}{star_marker}")

        # Example game context
        print("\n" + "=" * 80)
        print("\nExample Game Context: LAL @ DEN")
        context = await get_game_injury_context("DEN", "LAL", reports)
        print(f"  DEN: spread={context.home_team.spread_injury_score:.2f}, "
              f"totals={context.home_team.totals_injury_score:.2f}")
        print(f"  LAL: spread={context.away_team.spread_injury_score:.2f}, "
              f"totals={context.away_team.totals_injury_score:.2f}")
        print(f"  Home spread edge: {context.home_spread_edge:+.2f}")
        print(f"  Home totals edge: {context.home_totals_edge:+.2f}")

    asyncio.run(main())
