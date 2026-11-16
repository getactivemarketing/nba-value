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
        return httpx.Response(200, content=headshots.PNG_MAGIC) if "3149673" in u else httpx.Response(404)
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
            png = await headshots.fetch_png(conn, "nba", "Pascal Siakam", "IND", http)
            none_amb = await headshots.fetch_png(conn, "nba", "Jalen Williams", "IND", http)
            return n1, n2, n3, png, none_amb

    n1, n2, n3, png, none_amb = asyncio.run(run())
    assert (n1, n2, n3) == (3, 0, 3)
    assert sum("/roster" in c for c in calls) == 2
    assert _rows(conn) == [("Jalen Williams", None), ("Nobody Here", None), ("Pascal Siakam", 3149673)]
    assert png == headshots.PNG_MAGIC and none_amb is None


def test_refresh_survives_espn_outage(conn):
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(503))) as http:
            return await headshots.refresh(conn, "nba", {"IND"}, NOW, http)

    assert asyncio.run(run()) == 0 and _rows(conn) == []


def test_shared_nfl_name_gets_no_face_and_new_player_triggers_refresh(conn):
    calls = []

    def h(req):
        u = str(req.url)
        calls.append(u)
        if "/roster" in u:
            return httpx.Response(200, json={"athletes": [{"position": "offense", "items": [
                {"displayName": "Josh Allen", "id": "11"}, {"displayName": "Solo Guy", "id": "12"},
                {"displayName": "New Guy", "id": "13"}]}]})
        return httpx.Response(200, content=headshots.PNG_MAGIC)

    async def run():
        with conn.cursor() as cur:
            for n, t in (("Josh Allen", "BUF"), ("Josh Allen", "JAX"), ("Solo Guy", "BUF")):
                cur.execute("INSERT INTO edge_nfl_players (odds_name, team, match_method, confidence, updated_at) "
                            "VALUES (%s, %s, 'exact', 1, %s)", (n, t, NOW))
        conn.commit()
        async with httpx.AsyncClient(transport=httpx.MockTransport(h)) as http:
            await headshots.refresh(conn, "nfl", {"BUF", "JAX"}, NOW, http)
            a = await headshots.fetch_png(conn, "nfl", "Josh Allen", "BUF", http)
            b = await headshots.fetch_png(conn, "nfl", "Josh Allen", "JAX", http)
            ok = await headshots.fetch_png(conn, "nfl", "Solo Guy", "BUF", http)
            wrong_team = await headshots.fetch_png(conn, "nfl", "Solo Guy", "JAX", http)
            n_before = len(calls)
            with conn.cursor() as cur:
                cur.execute("INSERT INTO edge_nfl_players (odds_name, team, match_method, confidence, updated_at) "
                            "VALUES ('New Guy', 'BUF', 'exact', 1, %s)", (NOW,))
            conn.commit()
            await headshots.refresh(conn, "nfl", {"BUF"}, NOW + timedelta(days=1), http)  # new name -> stale
            return a, b, ok, wrong_team, n_before

    a, b, ok, wrong_team, n_before = asyncio.run(run())
    assert a is None and b is None and wrong_team is None and ok == headshots.PNG_MAGIC
    with conn.cursor() as cur:
        cur.execute("SELECT espn_id FROM edge_headshots WHERE odds_name = 'New Guy'")
        assert cur.fetchone() == (13,)


def test_refresh_malformed_payloads_do_not_raise(conn):
    for body in ([], "x"):
        async def run():
            async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=body))) as http:
                return await headshots.refresh(conn, "nba", {"IND"}, NOW, http)
        assert asyncio.run(run()) == 0


def test_refresh_stops_between_teams_when_told(conn):
    """Ruling 31: the refresh runs inside the social pass budget; past it no more ESPN calls start."""
    with conn.cursor() as cur:
        cur.execute("INSERT INTO edge_players (odds_name, team, match_method, confidence, updated_at) "
                    "VALUES ('Jaylen Brown', 'BOS', 'exact', 1, %s)", (NOW,))
    conn.commit()
    calls, asked = [], []

    def stop():
        asked.append(1)
        return len(asked) > 1  # budget runs out after the first team

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(_handler(calls))) as http:
            return await headshots.refresh(conn, "nba", {"IND", "BOS"}, NOW, http, stop=stop)

    asyncio.run(run())
    assert sum("/roster" in c for c in calls) == 1
