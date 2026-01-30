"""Player prop scoring service - identifies best value props."""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import structlog

from src.services.data.balldontlie import BallDontLieClient

logger = structlog.get_logger()

# Map prop types to season average fields
PROP_TO_STAT = {
    "points": "pts",
    "rebounds": "reb",
    "assists": "ast",
    "threes": "fg3m",
    "steals": "stl",
    "blocks": "blk",
    "turnovers": "turnover",
}


@dataclass
class ScoredProp:
    """A scored player prop with value analysis."""
    player_name: str
    prop_type: str
    line: float
    over_odds: float | None
    under_odds: float | None
    book: str
    game_id: str

    # Analysis
    season_avg: float | None
    edge: float | None  # Positive = over looks good, negative = under looks good
    edge_pct: float | None  # Edge as percentage of line
    recommendation: str  # "OVER", "UNDER", or "PASS"
    value_score: float  # 0-100 score
    reasoning: str


def calculate_implied_prob(decimal_odds: float) -> float:
    """Convert decimal odds to implied probability."""
    if decimal_odds <= 1:
        return 1.0
    return 1 / decimal_odds


def calculate_value_score(
    line: float,
    season_avg: float,
    over_odds: float | None,
    under_odds: float | None,
) -> tuple[float, str, str]:
    """
    Calculate value score for a prop.

    Returns:
        (value_score, recommendation, reasoning)
    """
    if season_avg is None or season_avg == 0:
        return 0, "PASS", "No season average available"

    # Calculate edge vs season average
    edge = season_avg - line  # Positive = player averages more than line (over)
    edge_pct = (edge / line) * 100 if line > 0 else 0

    # Base score from edge
    abs_edge_pct = abs(edge_pct)

    # Determine direction
    if edge > 0:
        # Over looks good
        direction = "OVER"
        relevant_odds = over_odds
    else:
        # Under looks good
        direction = "UNDER"
        relevant_odds = under_odds

    # Start with edge-based score (0-60 points)
    if abs_edge_pct >= 20:
        base_score = 60
    elif abs_edge_pct >= 15:
        base_score = 50
    elif abs_edge_pct >= 10:
        base_score = 40
    elif abs_edge_pct >= 5:
        base_score = 25
    else:
        base_score = 10

    # Odds bonus (0-30 points) - plus money = more value
    odds_bonus = 0
    if relevant_odds:
        implied_prob = calculate_implied_prob(relevant_odds)
        # Plus money (odds > 2.0) gets bonus
        if relevant_odds >= 2.20:  # +120 or better
            odds_bonus = 30
        elif relevant_odds >= 2.10:  # +110
            odds_bonus = 25
        elif relevant_odds >= 2.00:  # +100
            odds_bonus = 20
        elif relevant_odds >= 1.95:  # -105
            odds_bonus = 15
        elif relevant_odds >= 1.91:  # -110
            odds_bonus = 10
        else:
            odds_bonus = 5

    # Confidence bonus (0-10 points) for large edges
    confidence_bonus = min(10, int(abs_edge_pct / 2))

    value_score = base_score + odds_bonus + confidence_bonus
    value_score = min(100, max(0, value_score))

    # Don't recommend small edges
    if abs_edge_pct < 5:
        return value_score, "PASS", f"Edge too small ({edge_pct:+.1f}%)"

    # Build reasoning
    odds_str = ""
    if relevant_odds:
        if relevant_odds >= 2.0:
            american = f"+{int((relevant_odds - 1) * 100)}"
        else:
            american = f"{int(-100 / (relevant_odds - 1))}"
        odds_str = f" at {american}"

    reasoning = f"Avg {season_avg:.1f} vs line {line} ({edge_pct:+.1f}% edge){odds_str}"

    return value_score, direction, reasoning


async def score_props(props: list[dict], min_score: int = 50) -> list[ScoredProp]:
    """
    Score a list of player props and return top value plays.

    Args:
        props: List of prop dicts from database
        min_score: Minimum value score to include

    Returns:
        List of ScoredProp sorted by value_score descending
    """
    client = BallDontLieClient()
    scored = []

    # Cache player lookups to avoid duplicate API calls
    player_cache: dict[str, dict | None] = {}
    avg_cache: dict[int, dict | None] = {}

    for prop in props:
        player_name = prop.get("player_name", "")
        prop_type = prop.get("prop_type", "")
        line = float(prop.get("line", 0))
        over_odds = prop.get("over_odds")
        under_odds = prop.get("under_odds")

        if over_odds:
            over_odds = float(over_odds)
        if under_odds:
            under_odds = float(under_odds)

        # Skip unsupported prop types
        stat_key = PROP_TO_STAT.get(prop_type)
        if not stat_key:
            continue

        # Look up player (with caching)
        if player_name not in player_cache:
            try:
                player_cache[player_name] = await client.find_player_by_name(player_name)
            except Exception as e:
                logger.warning(f"Failed to find player {player_name}: {e}")
                player_cache[player_name] = None

        player = player_cache[player_name]
        if not player:
            continue

        player_id = player["id"]

        # Get season averages (with caching)
        if player_id not in avg_cache:
            try:
                avg_cache[player_id] = await client.get_season_averages(player_id, season=2025)
            except Exception as e:
                logger.warning(f"Failed to get averages for {player_name}: {e}")
                avg_cache[player_id] = None

        averages = avg_cache[player_id]
        season_avg = averages.get(stat_key) if averages else None

        # Calculate edge
        edge = None
        edge_pct = None
        if season_avg is not None and line > 0:
            edge = season_avg - line
            edge_pct = (edge / line) * 100

        # Calculate value score
        value_score, recommendation, reasoning = calculate_value_score(
            line, season_avg, over_odds, under_odds
        )

        if value_score >= min_score:
            scored.append(ScoredProp(
                player_name=player_name,
                prop_type=prop_type,
                line=line,
                over_odds=over_odds,
                under_odds=under_odds,
                book=prop.get("book", ""),
                game_id=prop.get("game_id", ""),
                season_avg=season_avg,
                edge=edge,
                edge_pct=edge_pct,
                recommendation=recommendation,
                value_score=value_score,
                reasoning=reasoning,
            ))

    # Sort by value score descending
    scored.sort(key=lambda x: x.value_score, reverse=True)

    return scored


async def get_top_props(limit: int = 10, min_score: int = 50) -> list[ScoredProp]:
    """
    Get top value props from today's games.

    Args:
        limit: Maximum number of props to return
        min_score: Minimum value score threshold

    Returns:
        List of top ScoredProp
    """
    import psycopg2
    import os

    DB_URL = os.environ.get('DATABASE_URL', 'postgresql://postgres:wzYHkiAOkykxiPitXKBIqPJxvifFtDPI@maglev.proxy.rlwy.net:46068/railway')

    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()

    # Get recent props (last 6 hours)
    cur.execute('''
        SELECT DISTINCT ON (player_name, prop_type)
            game_id, player_name, prop_type, line, over_odds, under_odds, book, snapshot_time
        FROM player_props
        WHERE snapshot_time > NOW() - INTERVAL '6 hours'
        ORDER BY player_name, prop_type, snapshot_time DESC
    ''')

    rows = cur.fetchall()
    cur.close()
    conn.close()

    props = [
        {
            "game_id": row[0],
            "player_name": row[1],
            "prop_type": row[2],
            "line": row[3],
            "over_odds": row[4],
            "under_odds": row[5],
            "book": row[6],
        }
        for row in rows
    ]

    logger.info(f"Scoring {len(props)} unique props")

    scored = await score_props(props, min_score=min_score)

    return scored[:limit]
