"""Edge storage SQL against a real Postgres.

Skipped unless EDGE_TEST_DB_URL points at a disposable database, e.g.
    EDGE_TEST_DB_URL=postgresql://edge@127.0.0.1:55432/edgetest
The edge_* tables are dropped and recreated on every run.
"""
import os
from datetime import datetime, timedelta, timezone

import psycopg2
import pytest
from sqlalchemy import create_engine

from src.database import Base
import src.models  # noqa: F401  registers the edge tables
from src.edge import opportunities, store
from src.edge.engine import Fair, Priced
from src.edge.quotes import Quote

URL = os.environ.get("EDGE_TEST_DB_URL")
pytestmark = pytest.mark.skipif(not URL, reason="EDGE_TEST_DB_URL not set")
NOW = datetime(2026, 10, 20, 18, 0, tzinfo=timezone.utc)
EDGE_TABLES = [t for name, t in Base.metadata.tables.items() if name.startswith("edge_")]


@pytest.fixture
def conn():
    engine = create_engine(URL)
    Base.metadata.drop_all(engine, tables=EDGE_TABLES)
    Base.metadata.create_all(engine, tables=EDGE_TABLES)
    engine.dispose()
    c = psycopg2.connect(URL)
    yield c
    c.close()


def q(book="fanduel", price=-110, side="over", at=NOW):
    return Quote("e1", "player_points", "Jayson Tatum", side, 27.5, book, price, "https://x", at)


def _event():
    return {"event_id": "e1", "sport": "basketball_nba", "home_team": "Detroit Pistons",
            "away_team": "Boston Celtics", "commence_time": NOW + timedelta(hours=2)}


def test_events_upsert_and_upcoming(conn):
    assert store.upsert_events(conn, [_event()]) == 1
    assert store.upsert_events(conn, [_event()]) == 1  # idempotent
    ev = store.upcoming_events(conn, NOW)
    assert [e["event_id"] for e in ev] == ["e1"] and ev[0]["last_props_poll"] is None
    store.mark_polled(conn, ["e1"], "last_props_poll", NOW)
    assert store.upcoming_events(conn, NOW)[0]["last_props_poll"] == NOW


def test_history_only_on_change(conn):
    store.upsert_events(conn, [_event()])
    assert store.write_quotes(conn, [q(), q(side="under", price=-110)]) == 2
    later = NOW + timedelta(minutes=5)
    assert store.write_quotes(conn, [q(at=later), q(side="under", price=-115, at=later)]) == 1
    latest = {x.side: x for x in store.load_quotes(conn, ["e1"])}
    assert latest["over"].observed_at == later and latest["under"].price == -115
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM edge_quote_history")
        assert cur.fetchone()[0] == 3


def test_opportunities_upsert_keeps_first_seen_and_max_ev(conn):
    p1 = Priced(q(book="betmgm", price=120), Fair(0.5, 4, True), 0.10, True, False)
    assert opportunities.record(conn, [p1], NOW) == 1
    p2 = Priced(q(book="betmgm", price=120), Fair(0.52, 4, True), 0.144, True, False)
    opportunities.record(conn, [p2], NOW + timedelta(minutes=5))
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*), MIN(first_seen_at), MAX(last_seen_at), MAX(max_ev), MAX(ev), "
                    "MAX(devig_method) FROM edge_opportunities")
        n, first, last, max_ev, ev, method = cur.fetchone()
    assert n == 1 and first == NOW and last == NOW + timedelta(minutes=5)
    assert float(max_ev) == pytest.approx(0.144) and float(ev) == pytest.approx(0.10)
    assert method == "power"


def test_state_and_credit_log(conn):
    store.set_state(conn, "guard_factor", "1", NOW)
    store.set_state(conn, "guard_factor", "2", NOW)
    assert store.get_state(conn) == {"guard_factor": "2"}
    store.log_api_call(conn, "odds", 6, 4_999_994, NOW - timedelta(hours=1))
    store.log_api_call(conn, "event_odds", 16, 4_999_978, NOW - timedelta(hours=30))
    assert store.cost_last_24h(conn, NOW) == 6
