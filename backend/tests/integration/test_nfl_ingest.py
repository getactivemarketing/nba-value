import pandas as pd
import pytest
from src.database import async_session_maker
from src.services.nfl.nfl_data import schedule_to_game_rows
from src.services.nfl.ingest import upsert_games


@pytest.mark.integration
@pytest.mark.asyncio
async def test_upsert_games_is_idempotent():
    sched = pd.read_parquet("tests/fixtures/nfl_schedule_sample.parquet")
    rows = schedule_to_game_rows(sched)
    async with async_session_maker() as session:
        n1 = await upsert_games(session, rows)
        await session.commit()
        n2 = await upsert_games(session, rows)   # re-run: no duplicates
        await session.commit()
    assert n1 == len(rows)
    assert n2 == len(rows)   # same count, upsert not insert
