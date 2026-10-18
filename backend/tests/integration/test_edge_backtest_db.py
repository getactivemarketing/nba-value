"""Backtest SQL against a real Postgres (EDGE_TEST_DB_URL, disposable)."""
import asyncio
import json
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


def _row(kind, event_id, label, credits, payload=None):
    return {"season": 2023, "week": 10, "game_id": GAME.game_id, "event_id": event_id, "kind": kind,
            "label": label, "kickoff": GAME.kickoff, "requested_at": GAME.kickoff, "snapshot_at": GAME.kickoff,
            "home": "BAL", "away": "CLE", "credits": credits, "payload": payload or {"id": event_id, "bookmakers": []}}


def test_spent_is_scoped_to_the_requested_kinds(conn):
    repo = fetch.Repo(conn)
    repo.insert([_row("props", "e1", "t6h", 200), _row("lines", "e1", "t6h", 60),
                 _row("td", "e1", "t6h", 20), _row("events", "day:2023-11-12", "list", 1)])
    assert repo.spent() == 281
    assert repo.spent(("td",)) == 21
    assert repo.spent(("props", "lines")) == 261


def test_replay_reads_a_td_snapshot_row(conn):
    td = {**EVENT, "bookmakers": [{"key": b, "markets": [{"key": "player_anytime_td", "outcomes": [
        {"name": "Yes", "description": "Amari Cooper", "price": y},
        *([{"name": "No", "description": "Amari Cooper", "price": n}] if n else [])]}]}
        for b, (y, n) in {"pinnacle": (150, -170), "novig": (150, -170), "fanduel": (190, None)}.items()]}
    at = datetime(2023, 11, 12, 12, 3, tzinfo=UTC)
    r = _row("td", "e1", "t6h", 20, td)
    fetch.Repo(conn).insert([{**r, "snapshot_at": at}])
    stat = {**STAT, "rec_tds": 1, "rush_tds": 0, "ret_tds": 0}
    assert replay.run_replay(conn, "r2", {GAME.game_id: GAME}, {GAME.game_id: [stat]},
                             {GAME.game_id: {"amari cooper"}}, 0.02, 0.15) > 0
    with conn.cursor() as cur:
        cur.execute("SELECT market, side, line, result FROM edge_bt_picks WHERE book = 'fanduel'")
        assert cur.fetchall() == [("player_anytime_td", "over", 0.5, "win")]


def test_bt_pick_tags_round_trip_jsonb(conn):
    asked = []

    class Ctx:
        pass

    def contexts(game):
        asked.append((game.season, game.week))
        return Ctx()

    import src.edge.backtest.replay as rp
    seen = []

    def fake_tags_for(ctx, player, teams, market, side):
        seen.append((type(ctx).__name__, teams))
        return [rp.Tag("def_soft", "tailwind", "d")]

    orig = rp.tags_for
    rp.tags_for = fake_tags_for
    try:
        fetch.Repo(conn).insert([_row("props", "e1", "t6h", 200, {**EVENT, "bookmakers": [
            {"key": b, "markets": [{"key": "player_reception_yds", "outcomes": [
                {"name": "Over", "description": "Amari Cooper", "price": o, "point": 55.5},
                {"name": "Under", "description": "Amari Cooper", "price": u, "point": 55.5}]}]}
            for b, (o, u) in SOFT.items()]}) | {"snapshot_at": datetime(2023, 11, 12, 12, 3, tzinfo=UTC)}])
        assert replay.run_replay(conn, "r3", {GAME.game_id: GAME}, {GAME.game_id: [STAT]},
                                 {GAME.game_id: {"amari cooper"}}, 0.02, 0.15, contexts=contexts) > 0
    finally:
        rp.tags_for = orig
    assert asked == [(2023, 10)] and seen and all(t == ("Ctx", ("BAL", "CLE")) for t in seen)
    with conn.cursor() as cur:
        cur.execute("SELECT tags FROM edge_bt_picks WHERE book = 'fanduel' AND side = 'over'")
        assert cur.fetchall()[0][0] == [{"key": "def_soft", "direction": "tailwind"}]
        cur.execute("SELECT COUNT(*) FROM edge_bt_picks WHERE tags IS NULL")
        assert cur.fetchone()[0] == 0


def test_run_report_upserts_edge_tag_status(conn, tmp_path):
    def ins(cur, i, tags, clv):
        cur.execute(
            "INSERT INTO edge_bt_picks (run_id, season, week, event_id, label, market, player, side, line, "
            "book, price, fair_prob, ev, verify, sharps, n_books, clv, result, profit, tags) VALUES ('r9', 2023, 1, 'e1', 't6h', "
            "'player_reception_yds', %s, 'over', 55.5, 'fanduel', -110, 0.5, 0.03, false, '', 4, %s, 'win', 0.9, %s)",
            (f"P{i}", clv, None if tags is None else json.dumps(tags)))

    with conn.cursor() as cur:
        for i in range(210):
            ins(cur, i, [{"key": "def_soft", "direction": "tailwind"}], 0.03 + (0.01 if i % 2 else -0.01))
        for i in range(210, 420):
            ins(cur, i, None, 0.0 + (0.01 if i % 2 else -0.01))  # legacy NULL tags count as untagged
        cur.execute("INSERT INTO edge_tag_status (tag, direction, verdict, n, run_id, updated_at) VALUES "
                    "('def_soft','tailwind','negative',1,'old',now()), ('usage_up','headwind','proven',1,'old',now())")
    conn.commit()
    assert report.run_report(conn, "r9", tmp_path / "r.md") == "r9"
    with conn.cursor() as cur:
        cur.execute("SELECT tag, direction, verdict, clv_lift, n, run_id FROM edge_tag_status")
        rows = cur.fetchall()
    assert len(rows) == 1  # stale pair removed, existing pair updated in place
    tag, direction, verdict, lift, n, run = rows[0]
    assert (tag, direction, verdict, n, run) == ("def_soft", "tailwind", "neutral", 210, "r9")
    assert lift == pytest.approx(0.03)
    assert "## Matchup tags" in (tmp_path / "r.md").read_text()
