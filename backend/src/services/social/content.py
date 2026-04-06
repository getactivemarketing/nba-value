"""Content generation for TruLine social media posts."""

import structlog
from datetime import datetime, timezone, timedelta, date
from decimal import Decimal

from sqlalchemy import select, and_, desc, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.models import (
    MLBGame, MLBPredictionSnapshot, MLBPrediction, MLBPitcher, MLBGameContext,
)

logger = structlog.get_logger()

# MLB team hashtags
TEAM_HASHTAGS = {
    "ARI": "#Dbacks", "ATL": "#Braves", "BAL": "#Orioles", "BOS": "#RedSox",
    "CHC": "#Cubs", "CWS": "#WhiteSox", "CIN": "#Reds", "CLE": "#Guardians",
    "COL": "#Rockies", "DET": "#Tigers", "HOU": "#Astros", "KC": "#Royals",
    "LAA": "#Angels", "LAD": "#Dodgers", "MIA": "#Marlins", "MIL": "#Brewers",
    "MIN": "#Twins", "NYM": "#Mets", "NYY": "#Yankees", "OAK": "#Athletics",
    "PHI": "#Phillies", "PIT": "#Pirates", "SD": "#Padres", "SF": "#Giants",
    "SEA": "#Mariners", "STL": "#Cardinals", "TB": "#Rays", "TEX": "#Rangers",
    "TOR": "#BlueJays", "WSH": "#Nationals",
}

TEAM_NAMES = {
    "ARI": "D-backs", "ATL": "Braves", "BAL": "Orioles", "BOS": "Red Sox",
    "CHC": "Cubs", "CWS": "White Sox", "CIN": "Reds", "CLE": "Guardians",
    "COL": "Rockies", "DET": "Tigers", "HOU": "Astros", "KC": "Royals",
    "LAA": "Angels", "LAD": "Dodgers", "MIA": "Marlins", "MIL": "Brewers",
    "MIN": "Twins", "NYM": "Mets", "NYY": "Yankees", "OAK": "Athletics",
    "PHI": "Phillies", "PIT": "Pirates", "SD": "Padres", "SF": "Giants",
    "SEA": "Mariners", "STL": "Cardinals", "TB": "Rays", "TEX": "Rangers",
    "TOR": "Blue Jays", "WSH": "Nationals",
}


def _fmt_odds(decimal_odds: float) -> str:
    """Format decimal odds to American."""
    if decimal_odds >= 2.0:
        return f"+{round((decimal_odds - 1) * 100)}"
    else:
        return str(round(-100 / (decimal_odds - 1)))


async def generate_daily_picks_thread(session: AsyncSession, game_date: date) -> list[str]:
    """
    Generate a Twitter thread with today's best MLB value picks.

    Returns a list of tweet-length strings (max 280 chars each).
    """
    tweets = []

    # Get today's games with predictions
    stmt = select(MLBPrediction).where(
        MLBPrediction.game_id.in_(
            select(MLBGame.game_id).where(
                and_(
                    MLBGame.game_date == game_date,
                    MLBGame.status == "scheduled",
                )
            )
        )
    ).where(MLBPrediction.market_type == "moneyline")

    result = await session.execute(stmt)
    predictions = result.scalars().all()

    if not predictions:
        return []

    # Get games for context
    games_result = await session.execute(
        select(MLBGame).where(
            and_(
                MLBGame.game_date == game_date,
                MLBGame.status == "scheduled",
            )
        ).order_by(MLBGame.game_time)
    )
    games = {g.game_id: g for g in games_result.scalars().all()}

    date_str = game_date.strftime("%m/%d")

    # Header tweet
    header = (
        f"MLB Picks {date_str}\n\n"
        f"Today's AI value plays from the model.\n"
        f"{len(games)} games analyzed.\n\n"
        f"Full card + value scores: truline.app\n\n"
        f"#MLB #SportsBetting #GamblingX"
    )
    tweets.append(header)

    # Individual game tweets (top games by predicted run diff magnitude)
    pred_map = {p.game_id: p for p in predictions}
    scored_games = []

    for game_id, game in games.items():
        pred = pred_map.get(game_id)
        if not pred or pred.predicted_run_diff is None:
            continue

        run_diff = float(pred.predicted_run_diff)
        p_home = float(pred.p_home_win) if pred.p_home_win else 0.5
        p_away = float(pred.p_away_win) if pred.p_away_win else 0.5

        favored_team = game.home_team if run_diff > 0 else game.away_team
        favored_pct = max(p_home, p_away)

        scored_games.append({
            "game": game,
            "pred": pred,
            "favored_team": favored_team,
            "favored_pct": favored_pct,
            "abs_diff": abs(run_diff),
            "run_diff": run_diff,
        })

    # Sort by model confidence (biggest predicted edge)
    scored_games.sort(key=lambda x: x["abs_diff"], reverse=True)

    # Top 3-5 picks as individual tweets
    for sg in scored_games[:5]:
        game = sg["game"]
        away = game.away_team
        home = game.home_team
        pick = sg["favored_team"]
        pct = round(sg["favored_pct"] * 100)
        diff = sg["run_diff"]

        game_time = ""
        if game.game_time:
            et = game.game_time - timedelta(hours=4)  # UTC to ET approx
            game_time = et.strftime("%-I:%M %p ET")

        pick_name = TEAM_NAMES.get(pick, pick)
        away_name = TEAM_NAMES.get(away, away)
        home_name = TEAM_NAMES.get(home, home)
        hashtag = TEAM_HASHTAGS.get(pick, "")

        tweet = (
            f"{away_name} @ {home_name} | {game_time}\n\n"
            f"Model pick: {pick_name} ({pct}%)\n"
            f"Predicted run diff: {diff:+.1f}\n\n"
            f"{hashtag} #MLB"
        )
        tweets.append(tweet)

    return tweets


async def generate_nrfi_tweet(session: AsyncSession, game_date: date) -> str | None:
    """
    Generate a NRFI (No Run First Inning) picks tweet for today's games.

    Analyzes pitcher matchups and team first inning scoring rates.
    """
    # Get first inning stats
    fi_result = await session.execute(text("""
        SELECT team,
               COUNT(*) as games,
               SUM(CASE WHEN runs > 0 THEN 1 ELSE 0 END) as scored,
               SUM(CASE WHEN runs = 0 THEN 1 ELSE 0 END) as scoreless
        FROM (
            SELECT home_team as team, COALESCE(home_first_inning_runs, 0) as runs
            FROM mlb_games WHERE status = 'final' AND game_type = 'R' AND home_first_inning_runs IS NOT NULL
            UNION ALL
            SELECT away_team as team, COALESCE(away_first_inning_runs, 0) as runs
            FROM mlb_games WHERE status = 'final' AND game_type = 'R' AND away_first_inning_runs IS NOT NULL
        ) t
        GROUP BY team
    """))
    fi_stats = {row[0]: {"games": int(row[1]), "scored": int(row[2]), "scoreless": int(row[3])}
                for row in fi_result.fetchall()}

    if not fi_stats:
        return None

    # Get today's scheduled games
    games_result = await session.execute(
        select(MLBGame).where(
            and_(
                MLBGame.game_date == game_date,
                MLBGame.status == "scheduled",
            )
        ).order_by(MLBGame.game_time)
    )
    games = games_result.scalars().all()

    if not games:
        return None

    # Score each game for NRFI likelihood
    nrfi_picks = []
    for game in games:
        home_fi = fi_stats.get(game.home_team)
        away_fi = fi_stats.get(game.away_team)

        if not home_fi or not away_fi:
            continue

        home_nrfi = home_fi["scoreless"] / home_fi["games"] if home_fi["games"] > 0 else 0.5
        away_nrfi = away_fi["scoreless"] / away_fi["games"] if away_fi["games"] > 0 else 0.5

        # Combined NRFI probability (both teams need to be scoreless)
        combined_nrfi = home_nrfi * away_nrfi
        min_games = min(home_fi["games"], away_fi["games"])

        nrfi_picks.append({
            "game": game,
            "combined_nrfi": combined_nrfi,
            "home_nrfi": home_nrfi,
            "away_nrfi": away_nrfi,
            "min_games": min_games,
        })

    # Sort by NRFI likelihood
    nrfi_picks.sort(key=lambda x: x["combined_nrfi"], reverse=True)

    if not nrfi_picks:
        return None

    date_str = game_date.strftime("%m/%d")

    # Build the tweet
    lines = [f"NRFI Plays {date_str}\n"]

    for pick in nrfi_picks[:3]:
        game = pick["game"]
        away = TEAM_NAMES.get(game.away_team, game.away_team)
        home = TEAM_NAMES.get(game.home_team, game.home_team)
        pct = round(pick["combined_nrfi"] * 100)
        lines.append(f"{'*' if pct >= 70 else '-'} {away} @ {home}: {pct}% NRFI")

    lines.append(f"\nBased on {nrfi_picks[0]['min_games']}+ games per team")
    lines.append(f"\n#NRFI #MLB #GamblingX")

    tweet = "\n".join(lines)

    # Ensure under 280
    if len(tweet) > 280:
        tweet = tweet[:277] + "..."

    return tweet


async def generate_results_tweet(session: AsyncSession, game_date: date) -> str | None:
    """
    Generate a results recap tweet for yesterday's picks.
    """
    # Get graded snapshots for the date
    stmt = select(MLBPredictionSnapshot).where(
        and_(
            MLBPredictionSnapshot.game_date == game_date,
            MLBPredictionSnapshot.best_bet_result.isnot(None),
        )
    )
    result = await session.execute(stmt)
    snapshots = result.scalars().all()

    if not snapshots:
        return None

    wins = sum(1 for s in snapshots if s.best_bet_result == "win")
    losses = sum(1 for s in snapshots if s.best_bet_result == "loss")
    pushes = sum(1 for s in snapshots if s.best_bet_result == "push")
    profit = sum(float(s.best_bet_profit or 0) for s in snapshots)

    total = wins + losses
    win_rate = round(wins / total * 100, 1) if total > 0 else 0

    date_str = game_date.strftime("%m/%d")
    profit_str = f"+{profit:.1f}" if profit >= 0 else f"{profit:.1f}"

    # Get NRFI results
    nrfi_result = await session.execute(
        select(MLBGame).where(
            and_(
                MLBGame.game_date == game_date,
                MLBGame.status == "final",
                MLBGame.home_first_inning_runs.isnot(None),
            )
        )
    )
    nrfi_games = nrfi_result.scalars().all()
    nrfi_count = sum(1 for g in nrfi_games
                     if (g.home_first_inning_runs or 0) + (g.away_first_inning_runs or 0) == 0)

    tweet = (
        f"Results {date_str}\n\n"
        f"Record: {wins}-{losses}"
    )

    if pushes:
        tweet += f"-{pushes}"

    tweet += (
        f" ({win_rate}%)\n"
        f"P/L: {profit_str}u\n"
    )

    if nrfi_games:
        tweet += f"\nNRFI: {nrfi_count}/{len(nrfi_games)} games scoreless in 1st\n"

    tweet += f"\nFull breakdown: truline.app\n\n#MLB #SportsBetting #GamblingX"

    if len(tweet) > 280:
        tweet = tweet[:277] + "..."

    return tweet


async def generate_nrfi_results_tweet(session: AsyncSession, game_date: date) -> str | None:
    """Generate a tweet showing yesterday's NRFI results."""
    result = await session.execute(
        select(MLBGame).where(
            and_(
                MLBGame.game_date == game_date,
                MLBGame.status == "final",
                MLBGame.home_first_inning_runs.isnot(None),
            )
        ).order_by(MLBGame.game_time)
    )
    games = result.scalars().all()

    if not games:
        return None

    date_str = game_date.strftime("%m/%d")

    nrfi_games = []
    yrfi_games = []
    for g in games:
        first_inn_runs = (g.home_first_inning_runs or 0) + (g.away_first_inning_runs or 0)
        away = TEAM_NAMES.get(g.away_team, g.away_team)
        home = TEAM_NAMES.get(g.home_team, g.home_team)
        label = f"{away}@{home}"

        if first_inn_runs == 0:
            nrfi_games.append(label)
        else:
            yrfi_games.append(f"{label} ({g.away_first_inning_runs}-{g.home_first_inning_runs})")

    nrfi_pct = round(len(nrfi_games) / len(games) * 100) if games else 0

    tweet = (
        f"1st Inning Recap {date_str}\n\n"
        f"NRFI: {len(nrfi_games)}/{len(games)} ({nrfi_pct}%)\n\n"
    )

    if nrfi_games:
        tweet += "Scoreless 1st:\n"
        tweet += ", ".join(nrfi_games[:6])
        if len(nrfi_games) > 6:
            tweet += f" +{len(nrfi_games) - 6} more"
        tweet += "\n"

    tweet += f"\n#NRFI #MLB #GamblingX"

    if len(tweet) > 280:
        tweet = tweet[:277] + "..."

    return tweet
