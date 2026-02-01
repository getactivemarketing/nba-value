"""Player prop scoring service - identifies best value props."""

import asyncio
import os
from dataclasses import dataclass
from datetime import datetime, timezone, date
import psycopg2
import structlog

from src.services.data.balldontlie import BallDontLieClient

logger = structlog.get_logger()

DB_URL = os.environ.get('DATABASE_URL', 'postgresql://postgres:wzYHkiAOkykxiPitXKBIqPJxvifFtDPI@maglev.proxy.rlwy.net:46068/railway')

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


async def snapshot_top_props(min_score: int = 50) -> int:
    """
    Score props and save predictions to prop_snapshots table.

    Args:
        min_score: Minimum value score to snapshot

    Returns:
        Number of props saved
    """
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()

    # Get recent props with game dates
    cur.execute('''
        SELECT DISTINCT ON (pp.player_name, pp.prop_type)
            pp.game_id, pp.player_name, pp.prop_type, pp.line,
            pp.over_odds, pp.under_odds, pp.book, pp.snapshot_time,
            g.game_date
        FROM player_props pp
        LEFT JOIN games g ON pp.game_id = g.game_id
        WHERE pp.snapshot_time > NOW() - INTERVAL '6 hours'
        ORDER BY pp.player_name, pp.prop_type, pp.snapshot_time DESC
    ''')

    rows = cur.fetchall()

    props = [
        {
            "game_id": row[0],
            "player_name": row[1],
            "prop_type": row[2],
            "line": row[3],
            "over_odds": row[4],
            "under_odds": row[5],
            "book": row[6],
            "game_date": row[8],
        }
        for row in rows
    ]

    logger.info(f"Scoring {len(props)} props for snapshot")

    # Score all props (no min_score filter for scoring, we'll filter on save)
    scored = await score_props(props, min_score=0)

    # Filter to value plays only
    value_props = [p for p in scored if p.value_score >= min_score and p.recommendation != "PASS"]

    logger.info(f"Found {len(value_props)} value props to snapshot")

    # Save to prop_snapshots
    saved = 0
    now = datetime.now(timezone.utc)

    for prop in value_props:
        # Get game_date from original props dict
        game_date = None
        for orig in props:
            if orig["player_name"] == prop.player_name and orig["prop_type"] == prop.prop_type:
                game_date = orig.get("game_date")
                break

        # Check if we already have a snapshot for this prop today
        cur.execute('''
            SELECT 1 FROM prop_snapshots
            WHERE player_name = %s AND prop_type = %s AND game_id = %s
            AND DATE(snapshot_time) = DATE(%s)
        ''', (prop.player_name, prop.prop_type, prop.game_id, now))

        if cur.fetchone():
            continue  # Already have snapshot for today

        cur.execute('''
            INSERT INTO prop_snapshots (
                game_id, player_name, prop_type, line, over_odds, under_odds, book,
                season_avg, edge, edge_pct, recommendation, value_score, reasoning,
                snapshot_time, game_date
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ''', (
            prop.game_id,
            prop.player_name,
            prop.prop_type,
            prop.line,
            prop.over_odds,
            prop.under_odds,
            prop.book,
            prop.season_avg,
            prop.edge,
            prop.edge_pct,
            prop.recommendation,
            prop.value_score,
            prop.reasoning,
            now,
            game_date,
        ))
        saved += 1

    conn.commit()
    cur.close()
    conn.close()

    logger.info(f"Saved {saved} prop snapshots")
    return saved


async def grade_prop_snapshots(days_back: int = 2) -> dict:
    """
    Grade prop snapshots from completed games.

    Fetches actual player stats from BallDontLie and compares to predictions.

    Args:
        days_back: How many days back to look for ungraded props

    Returns:
        Dict with grading summary
    """
    client = BallDontLieClient()
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()

    # Get ungraded prop snapshots from recent days
    cur.execute('''
        SELECT ps.snapshot_id, ps.game_id, ps.player_name, ps.prop_type,
               ps.line, ps.recommendation, ps.value_score, ps.game_date
        FROM prop_snapshots ps
        WHERE ps.result IS NULL
        AND ps.game_date >= CURRENT_DATE - INTERVAL '%s days'
        AND ps.game_date < CURRENT_DATE  -- Only grade past games
        ORDER BY ps.game_date DESC
    ''', (days_back,))

    ungraded = cur.fetchall()
    logger.info(f"Found {len(ungraded)} ungraded prop snapshots")

    if not ungraded:
        cur.close()
        conn.close()
        return {"graded": 0, "wins": 0, "losses": 0, "pushes": 0}

    # Get unique dates we need to fetch games for
    game_dates = set(row[7] for row in ungraded if row[7])
    logger.info(f"Fetching games for dates: {game_dates}")

    # Fetch all games for these dates from BallDontLie
    bdl_games_by_date: dict[date, list] = {}
    for game_date in game_dates:
        try:
            games = await client.get_games(start_date=game_date, end_date=game_date)
            bdl_games_by_date[game_date] = games
            logger.info(f"Found {len(games)} BDL games for {game_date}")
        except Exception as e:
            logger.error(f"Failed to fetch games for {game_date}: {e}")
            bdl_games_by_date[game_date] = []

    # Cache for player stats
    # Key: (player_name, game_date) -> stats dict or None
    stats_cache: dict[tuple, dict | None] = {}

    # Cache for player IDs
    player_id_cache: dict[str, int | None] = {}

    graded = 0
    wins = 0
    losses = 0
    pushes = 0
    now = datetime.now(timezone.utc)

    for row in ungraded:
        snapshot_id, game_id, player_name, prop_type, line, recommendation, value_score, game_date = row

        if not game_date:
            continue

        # Get stat key for this prop type
        stat_key = PROP_TO_STAT.get(prop_type)
        if not stat_key:
            logger.warning(f"Unknown prop type: {prop_type}")
            continue

        cache_key = (player_name, game_date)

        # Fetch player stats for this game if not cached
        if cache_key not in stats_cache:
            try:
                # Find player ID (with caching)
                if player_name not in player_id_cache:
                    player = await client.find_player_by_name(player_name)
                    player_id_cache[player_name] = player["id"] if player else None

                player_id = player_id_cache[player_name]
                if not player_id:
                    logger.warning(f"Could not find player: {player_name}")
                    stats_cache[cache_key] = None
                    continue

                # Get games for this date
                bdl_games = bdl_games_by_date.get(game_date, [])
                if not bdl_games:
                    logger.warning(f"No BDL games found for {game_date}")
                    stats_cache[cache_key] = None
                    continue

                # Get all game IDs for this date
                bdl_game_ids = [g.id for g in bdl_games]

                # Fetch stats for this player in these games
                stats_list = await client.get_player_stats(
                    game_ids=bdl_game_ids,
                    player_ids=[player_id]
                )

                if stats_list:
                    # Take the first (should only be one game per day for a player)
                    s = stats_list[0]
                    stats_cache[cache_key] = {
                        "pts": s.points,
                        "reb": s.rebounds,
                        "ast": s.assists,
                        "fg3m": s.fg3_made,
                        "stl": s.steals,
                        "blk": s.blocks,
                        "turnover": s.turnovers,
                    }
                else:
                    # Player didn't play (DNP, injury, etc.)
                    stats_cache[cache_key] = None

            except Exception as e:
                logger.error(f"Error fetching stats for {player_name} on {game_date}: {e}")
                stats_cache[cache_key] = None

        player_stats = stats_cache.get(cache_key)
        if not player_stats:
            continue

        # Get actual value
        actual_value = player_stats.get(stat_key)
        if actual_value is None:
            continue

        # Determine result
        line_float = float(line)
        if actual_value > line_float:
            result = "WIN" if recommendation == "OVER" else "LOSS"
        elif actual_value < line_float:
            result = "WIN" if recommendation == "UNDER" else "LOSS"
        else:
            result = "PUSH"

        # Update snapshot
        cur.execute('''
            UPDATE prop_snapshots
            SET actual_value = %s, result = %s, graded_at = %s
            WHERE snapshot_id = %s
        ''', (actual_value, result, now, snapshot_id))

        graded += 1
        if result == "WIN":
            wins += 1
        elif result == "LOSS":
            losses += 1
        else:
            pushes += 1

        logger.info(
            f"Graded {player_name} {prop_type}: {recommendation} {line_float} -> "
            f"Actual {actual_value} = {result}"
        )

    conn.commit()
    cur.close()
    conn.close()

    summary = {
        "graded": graded,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "win_rate": round(wins / (wins + losses) * 100, 1) if (wins + losses) > 0 else None,
    }

    logger.info(f"Grading complete: {summary}")
    return summary


def get_prop_performance(days: int = 7, min_score: int = 50) -> dict:
    """
    Get prop betting performance summary.

    Args:
        days: Number of days to analyze
        min_score: Minimum value score to include

    Returns:
        Performance summary dict
    """
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()

    # Overall stats
    cur.execute('''
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN result = 'LOSS' THEN 1 ELSE 0 END) as losses,
            SUM(CASE WHEN result = 'PUSH' THEN 1 ELSE 0 END) as pushes
        FROM prop_snapshots
        WHERE game_date >= CURRENT_DATE - INTERVAL '%s days'
        AND result IS NOT NULL
        AND value_score >= %s
    ''', (days, min_score))

    row = cur.fetchone()
    total, wins, losses, pushes = row
    # Handle NULL values from COUNT when no results
    total = total or 0
    wins = wins or 0
    losses = losses or 0
    pushes = pushes or 0

    # By value score bucket
    cur.execute('''
        SELECT
            CASE
                WHEN value_score >= 90 THEN '90+'
                WHEN value_score >= 80 THEN '80-89'
                WHEN value_score >= 70 THEN '70-79'
                WHEN value_score >= 60 THEN '60-69'
                ELSE '50-59'
            END as bucket,
            COUNT(*) as total,
            SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN result = 'LOSS' THEN 1 ELSE 0 END) as losses
        FROM prop_snapshots
        WHERE game_date >= CURRENT_DATE - INTERVAL '%s days'
        AND result IS NOT NULL
        AND value_score >= %s
        GROUP BY bucket
        ORDER BY bucket DESC
    ''', (days, min_score))

    buckets = []
    for row in cur.fetchall():
        bucket, total_b, wins_b, losses_b = row
        win_rate = round(wins_b / (wins_b + losses_b) * 100, 1) if (wins_b + losses_b) > 0 else None
        buckets.append({
            "bucket": bucket,
            "total": total_b,
            "wins": wins_b,
            "losses": losses_b,
            "win_rate": win_rate,
        })

    # Recent props
    cur.execute('''
        SELECT player_name, prop_type, line, recommendation, value_score,
               actual_value, result, game_date
        FROM prop_snapshots
        WHERE game_date >= CURRENT_DATE - INTERVAL '%s days'
        AND result IS NOT NULL
        AND value_score >= %s
        ORDER BY game_date DESC, value_score DESC
        LIMIT 20
    ''', (days, min_score))

    recent = []
    for row in cur.fetchall():
        recent.append({
            "player_name": row[0],
            "prop_type": row[1],
            "line": float(row[2]),
            "recommendation": row[3],
            "value_score": float(row[4]),
            "actual_value": float(row[5]) if row[5] else None,
            "result": row[6],
            "game_date": row[7].isoformat() if row[7] else None,
        })

    cur.close()
    conn.close()

    win_rate = round(wins / (wins + losses) * 100, 1) if (wins + losses) > 0 else None

    return {
        "days": days,
        "min_score": min_score,
        "total": total or 0,
        "wins": wins or 0,
        "losses": losses or 0,
        "pushes": pushes or 0,
        "win_rate": win_rate,
        "by_bucket": buckets,
        "recent": recent,
    }
