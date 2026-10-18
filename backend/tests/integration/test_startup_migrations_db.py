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


def test_init_edge_tables_creates_missing_edge_tables_and_columns_only():
    """The worker can boot before the API (2026-10-18): it must build its own edge schema,
    and only that, so it never races the API over the rest of the app's tables."""
    from sqlalchemy import create_engine

    from src.database import Base, init_edge_tables
    import src.models  # noqa: F401  registers the tables

    sync = create_engine(URL)
    edge = [t for n, t in Base.metadata.tables.items() if n.startswith("edge_")]
    Base.metadata.create_all(sync, tables=edge)
    sync.dispose()
    setup = psycopg2.connect(URL)
    setup.autocommit = True
    cur = setup.cursor()
    cur.execute("DROP TABLE IF EXISTS edge_tag_status")
    cur.execute("ALTER TABLE edge_opportunities DROP COLUMN IF EXISTS tags")
    cur.execute("DROP TABLE IF EXISTS games CASCADE")

    async def run():
        engine = _engine()
        try:
            return await init_edge_tables(engine)
        finally:
            await engine.dispose()

    added, skipped = asyncio.run(run())
    assert ("edge_opportunities", "tags") in added and skipped == []
    cur.execute("SELECT to_regclass('edge_tag_status'), to_regclass('games')")
    assert cur.fetchone() == ("edge_tag_status", None)
    cur.execute("SELECT 1 FROM information_schema.columns WHERE table_name='edge_opportunities' AND column_name='tags'")
    assert cur.fetchone() == (1,)
    setup.close()
