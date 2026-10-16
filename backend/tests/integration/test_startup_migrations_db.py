"""Startup column migrations must never hang a deploy.

A boot runs while the previous container is still serving. Unconditional
ALTER TABLE ... ADD COLUMN IF NOT EXISTS takes an ACCESS EXCLUSIVE lock even
when the column exists, so one open transaction on the old container blocked
the new one's startup until the healthcheck failed (2026-10-16).
"""
import asyncio
import os
import time

import psycopg2
import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from src.database import add_missing_columns

URL = os.environ.get("EDGE_TEST_DB_URL")
pytestmark = pytest.mark.skipif(not URL, reason="EDGE_TEST_DB_URL not set")


def _engine():
    return create_async_engine(URL.replace("postgresql://", "postgresql+asyncpg://"))


@pytest.fixture
def holder():
    setup = psycopg2.connect(URL)
    setup.autocommit = True
    setup.cursor().execute("DROP TABLE IF EXISTS t_mig; CREATE TABLE t_mig (a int, b int)")
    setup.close()
    h = psycopg2.connect(URL)  # like the old container: a read inside an open transaction
    h.cursor().execute("SELECT * FROM t_mig")
    yield h
    h.rollback()
    h.close()


def test_existing_columns_take_no_lock_and_blocked_ones_are_skipped(holder):
    migrations = [("t_mig", "b", "INTEGER"), ("t_mig", "c", "INTEGER"), ("no_such_table", "x", "INTEGER")]

    async def run():
        engine = _engine()
        try:
            return await asyncio.wait_for(add_missing_columns(engine, migrations, lock_timeout_ms=500), 10)
        finally:
            await engine.dispose()

    t0 = time.monotonic()
    added, skipped = asyncio.run(run())
    assert time.monotonic() - t0 < 5          # boot is not held hostage
    assert added == [] and skipped == [("t_mig", "c")]

    holder.rollback()                          # old container lets go; next boot adds it
    added, skipped = asyncio.run(_once(migrations))
    assert added == [("t_mig", "c")] and skipped == []


async def _once(migrations):
    engine = _engine()
    try:
        return await add_missing_columns(engine, migrations, lock_timeout_ms=500)
    finally:
        await engine.dispose()
