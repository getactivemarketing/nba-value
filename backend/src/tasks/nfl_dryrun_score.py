"""NFL Phase 3 dry-run: score a completed week end-to-end and grade it.

Builds the feature frame for one past season+week, synthesizes nfl_markets rows
from that week's nflverse lines (spread/total at -110, moneylines as-is), scores
each game with the trained (calibrated) bundles, writes nfl_prediction_snapshots,
and grades vs the final scores. Confirms: best_bet is always a total (spread/ML
shadow), and no best_bet value_score is saturated at 100.

Run:  export DATABASE_URL=$(grep -oE "postgresql://[^\"']+" src/tasks/prediction_tracker.py | head -1)
      python3 -m src.tasks.nfl_dryrun_score 2024 10
"""
import asyncio
import sys
from datetime import datetime, timezone

from sqlalchemy.dialects.postgresql import insert

from src.config import settings
from src.database import async_session_maker
from src.models import NFLPredictionSnapshot
from src.services.nfl.training_data import load_training_frames, build_feature_frame
from src.services.nfl.model_training import load_bundle
from src.services.nfl.scorer import score_game
from src.services.nfl.snapshot import build_snapshot, grade_snapshot


def _am_to_dec(am):
    if am is None:
        return None
    am = float(am)
    return 1 + (am / 100) if am > 0 else 1 + (100 / abs(am))


async def main(season: int, week: int) -> None:
    mov = load_bundle(settings.nfl_mov_model_path)
    tot = load_bundle(settings.nfl_totals_model_path)
    print(f"totals calibrator loaded: {tot.get('calibrator') is not None}")

    async with async_session_maker() as s:
        frames = await load_training_frames(s, [season])
        frame = build_feature_frame(*frames)
        wk = frame[frame["week"] == week]
        print(f"{season} week {week}: {len(wk)} modelable games\n")

        rows, best_bet_types, saturated = [], [], 0
        for _, g in wk.iterrows():
            markets = [
                {"market_type": "spread", "line": g["spread_line"],
                 "home_odds": 1.909, "away_odds": 1.909},
                {"market_type": "moneyline", "line": None,
                 "home_odds": _am_to_dec(g["home_moneyline"]), "away_odds": _am_to_dec(g["away_moneyline"])},
                {"market_type": "total", "line": g["total_line"],
                 "over_odds": 1.909, "under_odds": 1.909},
            ]
            scored = score_game(g.to_dict(), markets, mov, tot)
            game = {"game_id": g["game_id"], "home_team": g["home_team"],
                    "away_team": g["away_team"], "snapshot_time": datetime.now(timezone.utc)}
            snap = build_snapshot(game, scored)
            # grade vs final score (from nfl_games via the frame's margin/total targets)
            home_score = int((g["margin"] + g["total"]) / 2)
            away_score = int(g["total"] - home_score)
            graded = grade_snapshot(snap, home_score, away_score, g["spread_line"], g["total_line"])
            snap.update(graded)

            stmt = insert(NFLPredictionSnapshot).values(**snap).on_conflict_do_update(
                index_elements=["game_id"],
                set_={k: snap[k] for k in snap if k != "game_id"})
            await s.execute(stmt)

            if snap["best_bet_type"]:
                best_bet_types.append(snap["best_bet_type"])
                if snap["best_bet_value_score"] and snap["best_bet_value_score"] >= 99.5:
                    saturated += 1
            bb = (f"{snap['best_bet_type']} {snap.get('best_total_direction') or ''} "
                  f"{snap['best_bet_line']} vs={snap['best_bet_value_score']}"
                  if snap["best_bet_type"] else "no bet")
            print(f"  {g['away_team']}@{g['home_team']:<4} {away_score}-{home_score}  best_bet: {bb}"
                  f"  -> {snap.get('best_bet_result') or '-'}")
        await s.commit()

    from collections import Counter
    print(f"\nbest_bet market mix: {dict(Counter(best_bet_types))}")
    print(f"saturated best_bet value_scores (>=99.5): {saturated}  (must be 0)")
    non_total = [t for t in best_bet_types if t != "total"]
    print(f"non-total best_bets (must be 0 — spread/ML are shadow): {len(non_total)}")


if __name__ == "__main__":
    season = int(sys.argv[1]) if len(sys.argv) > 1 else 2024
    week = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    asyncio.run(main(season, week))
