"""One-off backfill script to score completed MLB games.

The scorer was broken for most of the MLB season due to two bugs
(MissingGreenlet + missing unique constraint on mlb_predictions).
As a result, no predictions were saved for completed games.

This script re-scores completed games retroactively so we can build
a historical track record for the evaluation page.

Usage:
    python -m src.tasks.backfill_mlb_predictions [--days N]

Default scores the last 30 days of completed games.
"""

import asyncio
import sys
from datetime import datetime, timezone, timedelta, date

import structlog
from sqlalchemy import select, and_

# Add parent to path for imports
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.database import async_session
from src.models import MLBGame
from src.services.mlb.scorer import MLBScorer

logger = structlog.get_logger()


async def backfill_completed_games(days: int = 30) -> dict:
    """Score all completed games from the last N days."""
    end_date = date.today()
    start_date = end_date - timedelta(days=days)

    print(f"[BACKFILL] Scoring completed games from {start_date} to {end_date}", flush=True)

    results = {
        "total_games": 0,
        "scored": 0,
        "save_failures": 0,
        "skipped": 0,
        "errors": [],
    }

    async with async_session() as session:
        stmt = select(MLBGame).where(
            and_(
                MLBGame.game_date >= start_date,
                MLBGame.game_date <= end_date,
                MLBGame.status == "final",
                MLBGame.home_score.isnot(None),
                MLBGame.away_score.isnot(None),
            )
        ).order_by(MLBGame.game_date)

        result = await session.execute(stmt)
        games = list(result.scalars().all())
        results["total_games"] = len(games)

        print(f"[BACKFILL] Found {len(games)} completed games to score", flush=True)

        scorer = MLBScorer(session)
        predictions = []

        for i, game in enumerate(games):
            try:
                prediction = await scorer.score_game(game)
                predictions.append(prediction)
                results["scored"] += 1

                if (i + 1) % 10 == 0:
                    print(f"[BACKFILL] Scored {i+1}/{len(games)}...", flush=True)
            except Exception as e:
                results["save_failures"] += 1
                error_msg = f"{game.game_id}: {str(e)[:100]}"
                results["errors"].append(error_msg)
                print(f"[BACKFILL] ERROR scoring {game.game_id}: {str(e)[:100]}", flush=True)

        # Save all predictions in one batch
        if predictions:
            print(f"[BACKFILL] Saving {len(predictions)} predictions...", flush=True)
            try:
                saved = await scorer.save_predictions(predictions)
                print(f"[BACKFILL] Saved {saved} prediction rows", flush=True)
                results["saved_rows"] = saved
            except Exception as e:
                print(f"[BACKFILL] SAVE ERROR: {e}", flush=True)
                results["save_error"] = str(e)

    return results


if __name__ == '__main__':
    days = 30
    if len(sys.argv) > 1 and sys.argv[1] == '--days':
        days = int(sys.argv[2])

    results = asyncio.run(backfill_completed_games(days))

    print(f"\n[BACKFILL] Complete!", flush=True)
    print(f"  Total games: {results['total_games']}", flush=True)
    print(f"  Scored: {results['scored']}", flush=True)
    print(f"  Failed: {results['save_failures']}", flush=True)
    if results.get('errors'):
        print(f"  First 5 errors:", flush=True)
        for e in results['errors'][:5]:
            print(f"    {e}", flush=True)
