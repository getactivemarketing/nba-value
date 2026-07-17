# src/tasks/nfl_train.py
"""Build NFL features, run the walk-forward backtest, train + save final models."""
import asyncio

import structlog

from src.config import settings
from src.database import async_session_maker
from src.services.nfl.training_data import (
    load_training_frames, build_feature_frame, MOV_FEATURES, TOTALS_FEATURES)
from src.services.nfl.model_training import train_regressor, save_bundle
from src.services.nfl.backtest import walk_forward

logger = structlog.get_logger()
ALL_SEASONS = list(range(2010, 2025))
TEST_SEASONS = list(range(2019, 2025))
FINAL_TRAIN = list(range(2010, 2024))  # hold out 2024 as the headline walk-forward year


async def main() -> None:
    async with async_session_maker() as session:
        frames = await load_training_frames(session, ALL_SEASONS)
    frame = build_feature_frame(*frames)
    print(f"modelable games: {len(frame)} ({frame['season'].min()}-{frame['season'].max()})")

    report = walk_forward(frame, TEST_SEASONS, threshold=0.05)
    print("\n=== WALK-FORWARD BACKTEST (2019-2024, flat -110 units) ===")
    for mkt in ("spread", "totals"):
        m = report[mkt]
        print(f"  {mkt:7} {m['wins']}/{m['n']}  ATS={m['ats_pct']}%  units={m['units']:+.2f}")
    print("  reliability (spread, |edge| band -> realized win%):")
    for b in report["spread_reliability"]:
        print(f"    {b['edge_band']}  n={b['n']}  win%={b['win_pct']}")
    print(f"  saturation_max_prob={report['saturation_max_prob']} (should be < 1.0)")

    final = frame[frame["season"].isin(FINAL_TRAIN)]
    mov, mov_std = train_regressor(final, MOV_FEATURES, "margin")
    tot, tot_std = train_regressor(final, TOTALS_FEATURES, "total")
    save_bundle(settings.nfl_mov_model_path, mov, MOV_FEATURES, mov_std, FINAL_TRAIN)
    save_bundle(settings.nfl_totals_model_path, tot, TOTALS_FEATURES, tot_std, FINAL_TRAIN)
    print(f"\nsaved models: mov_std={mov_std:.2f} total_std={tot_std:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
