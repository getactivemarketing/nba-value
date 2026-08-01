"""Build a point-in-time-correct 2026 MLB training set.

Only possible since the game-log backfill (c4b26e5) gave mlb_pitcher_stats real
history. Before that, every row would have carried today's season-to-date
pitcher numbers regardless of when the game was played.

Starts at 2026-04-13: team stats history begins 2026-04-12 and the feature
lookup takes the most recent row STRICTLY BEFORE the game date, so earlier
games have no team stats to draw on.

    python -m src.tasks.build_mlb_training_2026 [--out models/mlb_training_2026.parquet]
"""

import asyncio
import sys
from datetime import date

import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.config import settings
from src.services.mlb.features import build_training_data

# Team stats history starts 2026-04-12; features look strictly backwards.
TRAIN_START = date(2026, 4, 13)


async def build(out_path: str, end: date) -> dict:
    engine = create_async_engine(settings.async_database_url, echo=False)
    Session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with Session() as session:
        rows = await build_training_data(session, TRAIN_START, end)

    await engine.dispose()

    df = pd.DataFrame(rows)
    df.to_parquet(out_path)

    # Coverage is the number that decides whether this set is usable at all:
    # a feature present on half the rows is mostly imputation.
    key = [
        "home_starter_era", "away_starter_era", "starter_era_diff",
        "home_ops", "home_era", "temperature", "is_dome",
        "weather_factor", "home_last_10_win_pct",
    ]
    print(f"\nrows: {len(df)}  ->  {out_path}\n")
    print(f"{'feature':<26}{'non-null':>10}{'coverage':>11}{'unique':>9}")
    for col in key:
        if col not in df.columns:
            print(f"{col:<26}{'MISSING':>10}")
            continue
        n = int(df[col].notna().sum())
        print(f"{col:<26}{n:>10}{n / len(df) * 100:>10.1f}%{df[col].nunique():>9}")

    return {"rows": len(df), "path": out_path}


if __name__ == "__main__":
    args = sys.argv[1:]
    out = "models/mlb_training_2026.parquet"
    for i, a in enumerate(args):
        if a == "--out" and i + 1 < len(args):
            out = args[i + 1]
    asyncio.run(build(out, end=date(2026, 7, 30)))
