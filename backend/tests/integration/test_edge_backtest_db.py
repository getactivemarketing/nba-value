"""Backtest SQL against a real Postgres (EDGE_TEST_DB_URL, disposable)."""
import asyncio
import os
from datetime import datetime, timezone

import psycopg2
import pytest
from sqlalchemy import create_engine

from src.database import Base
import src.models  # noqa: F401
from src.edge.backtest import fetch, replay, report
from src.edge.backtest.nflverse import Game
from src.edge.client import Usage

URL = os.environ.get("EDGE_TEST_DB_URL")
pytestmark = pytest.mark.skipif(not URL, reason="EDGE_TEST_DB_URL not set")
UTC = timezone.utc
GAME = Game("2023_10_CLE_BAL", 2023, 10, "BAL", "CLE", datetime(2023, 11, 12, 18, 0, tzinfo=UTC), 31, 33)
EDGE_TABLES = [t for name, t in Base.metadata.tables.items() if name.startswith("edge_")]
LIVE = ("edge_quotes", "edge_quote_history", "edge_opportunities", "edge_api_calls", "edge_worker_state")


@pytest.fixture
def conn():
    engine = create_engine(URL)
    Base.metadata.drop_all(engine, tables=EDGE_TABLES)
    Base.metadata.create_all(engine, tables=EDGE_TABLES)
    engine.dispose()
    c = psycopg2.connect(URL)
    yield c
    c.close()


EVENT = {"id": "e1", "home_team": "Baltimore Ravens", "away_team": "Cleveland Browns",
         "commence_time": "2023-11-12T18:03:00Z"}
SOFT = {"pinnacle": (-110, -110), "draftkings": (-110, -110), "betmgm": (-110, -110), "fanduel": (120, -150)}
STAT = {"name": "amari cooper", "team": "CLE", "pass_att": 0, "pass_cmp": 0, "pass_yds": 0, "pass_tds": 0,
        "pass_int": 0, "rush_att": 0, "rush_yds": 0, "receptions": 4, "rec_yds": 60}


def props_event():
    return {**EVENT, "bookmakers": [{"key": b, "markets": [{"key": "player_reception_yds", "outcomes": [
        {"name": "Over", "description": "Amari Cooper", "price": o, "point": 55.5},
        {"name": "Under", "description": "Amari Cooper", "price": u, "point": 55.5}]}]}
        for b, (o, u) in SOFT.items()]}


class PropsClient:
    """Self-contained: tests/ is not a package, so unit-test helpers can't be imported here."""
    async def get_historical_events(self, sport, date):
        return {"timestamp": date, "data": [EVENT]}, Usage(4_000_000, None, 1)

    async def get_historical_game_odds(self, sport, date):
        return {"timestamp": date, "data": [{**EVENT, "bookmakers": []}]}, Usage(4_000_000, None, 60)

    async def get_historical_event_odds(self, event_id, sport, date, markets=None):
        return {"timestamp": date, "data": props_event()}, Usage(4_000_000, None, 200)


async def _nosleep(_):
    return None


def test_ensure_tables_is_idempotent(conn):
    fetch.ensure_tables(URL)
    fetch.ensure_tables(URL)


def test_repo_round_trip_and_conflict(conn):
    repo = fetch.Repo(conn)
    row = {"season": 2023, "week": 10, "game_id": "g", "event_id": "e1", "kind": "props", "label": "t6h",
           "kickoff": GAME.kickoff, "requested_at": GAME.kickoff, "snapshot_at": GAME.kickoff, "home": "BAL",
           "away": "CLE", "credits": 200, "payload": {"id": "e1", "bookmakers": []}}
    repo.insert([row])
    repo.insert([{**row, "credits": 999}])
    assert repo.done_keys() == {("e1", "props", "t6h")} and repo.spent() == 200


def test_fetch_replay_report_round_trip_leaves_live_tables_empty(conn, tmp_path):
    s = asyncio.run(fetch.run_fetch(fetch.Repo(conn), PropsClient(), [GAME], cap=800_000,
                                    min_remaining=0, sleep=_nosleep))
    assert s.stopped is None and s.unmatched == []
    n = replay.run_replay(conn, "r1", {GAME.game_id: GAME}, {GAME.game_id: [STAT]},
                          {GAME.game_id: {"amari cooper"}}, 0.02, 0.15)
    assert n > 0
    with conn.cursor() as cur:
        cur.execute("SELECT label, book, side, result, clv FROM edge_bt_picks WHERE book = 'fanduel' "
                    "AND side = 'over' ORDER BY label")
        rows = cur.fetchall()
        assert [r[0] for r in rows] == ["t1h", "t24h", "t6h"]
        assert all(r[3] == "win" and r[4] is not None for r in rows)
        for t in LIVE:
            cur.execute(f"SELECT COUNT(*) FROM {t}")
            assert cur.fetchone()[0] == 0, t
    assert report.run_report(conn, None, tmp_path / "r.md") == "r1"
    assert "## Recommendations" in (tmp_path / "r.md").read_text()
