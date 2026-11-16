"""headshots.refresh / fetch_png against a throwaway Postgres with a mocked ESPN. Needs EDGE_TEST_DB_URL."""
import asyncio
import os
from datetime import datetime, timedelta, timezone

import httpx
import psycopg2
import pytest
from sqlalchemy import create_engine

from src.database import Base
import src.models  # noqa: F401
from src.edge.social import headshots

URL = os.environ.get("EDGE_TEST_DB_URL")
pytestmark = pytest.mark.skipif(not URL, reason="EDGE_TEST_DB_URL not set")
EDGE_TABLES = [t for name, t in Base.metadata.tables.items() if name.startswith("edge_")]
NOW = datetime(2026, 11, 16, 12, tzinfo=timezone.utc)


@pytest.fixture
def conn():
    engine = create_engine(URL)
    Base.metadata.drop_all(engine, tables=EDGE_TABLES)
    Base.metadata.create_all(engine, tables=EDGE_TABLES)
    engine.dispose()
    c = psycopg2.connect(URL)
    with c.cursor() as cur:
        for n in ("Pascal Siakam", "Jalen Williams", "Nobody Here"):
            cur.execute("INSERT INTO edge_players (odds_name, team, match_method, confidence, updated_at) "
                        "VALUES (%s, 'IND', 'exact', 1, %s)", (n, NOW))
    c.commit()
    yield c
    c.close()


def _handler(calls):
    def h(req):
        u = str(req.url)
        calls.append(u)
        if "/roster" in u:
            return httpx.Response(200, json={"athletes": [
                {"displayName": "Pascal Siakam", "id": "3149673"},
                {"displayName": "Jalen Williams", "id": "1"}, {"displayName": "Jalen Williams", "id": "2"}]})
        return httpx.Response(200, content=b"\x89PNG") if "3149673" in u else httpx.Response(404)
    return h


def _rows(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT odds_name, espn_id FROM edge_headshots ORDER BY odds_name")
        return cur.fetchall()


def test_refresh_writes_rows_and_skips_fresh_teams(conn):
    calls = []

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(_handler(calls))) as http:
            n1 = await headshots.refresh(conn, "nba", {"IND"}, NOW, http)
            n2 = await headshots.refresh(conn, "nba", {"IND"}, NOW + timedelta(days=3), http)
            n3 = await headshots.refresh(conn, "nba", {"IND"}, NOW + timedelta(days=8), http)
            png = await headshots.fetch_png(conn, "nba", "Pascal Siakam", http)
            none_amb = await headshots.fetch_png(conn, "nba", "Jalen Williams", http)
            return n1, n2, n3, png, none_amb

    n1, n2, n3, png, none_amb = asyncio.run(run())
    assert (n1, n2, n3) == (3, 0, 3)
    assert sum("/roster" in c for c in calls) == 2
    assert _rows(conn) == [("Jalen Williams", None), ("Nobody Here", None), ("Pascal Siakam", 3149673)]
    assert png == b"\x89PNG" and none_amb is None


def test_refresh_survives_espn_outage(conn):
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(503))) as http:
            return await headshots.refresh(conn, "nba", {"IND"}, NOW, http)

    assert asyncio.run(run()) == 0 and _rows(conn) == []
