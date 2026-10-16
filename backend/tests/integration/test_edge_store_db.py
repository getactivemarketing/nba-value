"""Edge storage SQL against a real Postgres.

Skipped unless EDGE_TEST_DB_URL points at a disposable database, e.g.
    EDGE_TEST_DB_URL=postgresql://edge@127.0.0.1:55432/edgetest
The edge_* tables are dropped and recreated on every run.
"""
import os
from datetime import date, datetime, timedelta, timezone

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


def test_run_grading_closes_and_grades(conn):
    from src.edge import grading

    tip = NOW
    store.upsert_events(conn, [{**_event(), "commence_time": tip}])
    at = tip - timedelta(minutes=3)
    closing = []  # the frozen pre-tip snapshot: 3 books incl. a sharp one, both markets
    for b in ("pinnacle", "fanduel", "draftkings"):
        closing += [Quote("e1", "player_points", "Jayson Tatum", "over", 27.5, b, -120, None, at),
                    Quote("e1", "player_points", "Jayson Tatum", "under", 27.5, b, 100, None, at),
                    Quote("e1", "h2h", "", "home", 0.0, b, -130, None, at),
                    Quote("e1", "h2h", "", "away", 0.0, b, 110, None, at)]
    store.write_quotes(conn, closing)
    early = tip - timedelta(hours=3)
    opportunities.record(conn, [
        Priced(Quote("e1", "player_points", "Jayson Tatum", "over", 27.5, "betmgm", 120, None, early),
               Fair(0.5, 3, True), 0.10, True, False),
        Priced(Quote("e1", "player_points", "Ghost Player", "over", 10.5, "betmgm", 120, None, early),
               Fair(0.5, 3, True), 0.10, True, False),
        Priced(Quote("e1", "h2h", "", "away", 0.0, "betmgm", 150, None, early),
               Fair(0.45, 3, True), 0.125, True, False),
    ], early)
    with conn.cursor() as cur:
        cur.execute("""INSERT INTO edge_players (odds_name, bdl_player_id, team, match_method, confidence, updated_at)
                       VALUES ('Jayson Tatum', 434, 'BOS', 'exact', 1, now()),
                              ('Ghost Player', 999, 'BOS', 'exact', 1, now())""")
        cur.execute("""INSERT INTO edge_player_games (bdl_player_id, bdl_game_id, season, game_date, team, opponent,
                       is_home, minutes, pts, reb, ast, fg3m, stl, blk, tov)
                       VALUES (434, 1, 2026, '2026-10-20', 'BOS', 'DET', false, 36, 31, 8, 6, 4, 1, 0, 3)""")
        cur.execute("UPDATE edge_events SET completed = TRUE, home_score = 110, away_score = 104")
    conn.commit()

    out = grading.run_grading(conn, tip + timedelta(hours=4))
    assert out == {"closed": 3, "graded": 3}
    with conn.cursor() as cur:
        cur.execute("SELECT player, market, result, profit_units, clv, clv_note FROM edge_opportunities ORDER BY id")
        rows = cur.fetchall()
    tatum, ghost, ml = rows
    assert tatum[2] == "win" and float(tatum[3]) == pytest.approx(1.2) and tatum[4] is not None
    assert ghost[2] == "void" and ghost[4] is None and ghost[5].startswith("thin close")
    assert ml[2] == "loss" and float(ml[3]) == -1.0 and ml[4] is not None


def test_views_on_real_sql(conn):
    from src.edge import views

    now = NOW
    store.upsert_events(conn, [_event()])  # tips at NOW+2h
    at = now - timedelta(minutes=1)
    quotes = []
    for b, o, u in (("pinnacle", -110, -110), ("fanduel", -110, -110), ("draftkings", -110, -110),
                    ("betmgm", 120, -145), ("prizepicks", -137, -137)):
        quotes += [Quote("e1", "player_points", "Jayson Tatum", "over", 27.5, b, o, f"https://{b}", at),
                   Quote("e1", "player_points", "Jayson Tatum", "under", 27.5, b, u, None, at)]
    store.write_quotes(conn, quotes)
    with conn.cursor() as cur:
        cur.execute("""INSERT INTO edge_players (odds_name, bdl_player_id, team, match_method, confidence, updated_at)
                       VALUES ('Jayson Tatum', 434, 'BOS', 'exact', 1, now())""")
        for i, pts in enumerate([30, 25, 20]):
            cur.execute("""INSERT INTO edge_player_games (bdl_player_id, bdl_game_id, season, game_date, team,
                           opponent, is_home, minutes, pts, reb, ast, fg3m, stl, blk, tov)
                           VALUES (434, %s, 2025, %s, 'BOS', 'DET', false, 34, %s, 5, 5, 2, 1, 0, 2)""",
                        (i, datetime(2026, 4, 10 - i).date(), pts))
    conn.commit()

    games = views.games(conn, now)
    assert games[0]["event_id"] == "e1" and games[0]["prop_markets"] == 1

    rows = views.ev_rows(conn, now)
    assert [(r["book"], r["side"]) for r in rows] == [("betmgm", "over")]
    assert rows[0]["bet"] == "Jayson Tatum Over 27.5 Points" and rows[0]["bdl_player_id"] == 434
    assert rows[0]["link"] == "https://betmgm"

    grid = views.odds(conn, now, "e1", "player_points")
    over = [r for r in grid["rows"] if r["side"] == "over"][0]
    assert over["best_book"] == "betmgm" and set(over["prices"]) >= {"pinnacle", "prizepicks"}
    assert over["fair_prob"] is not None and not over["prices"]["fanduel"]["stale"]

    page = views.player(conn, now, 434, "player_points", None, 0)
    assert page["line"] == 27.5 and page["opponent"] == "DET" and page["is_home"] is False
    assert page["hit_rates"]["l5"]["games"] == 3 and len(page["prices"]) == 10

    store.set_state(conn, "heartbeat_at", now.isoformat(), now)
    st = views.status(conn, now)
    assert st["heartbeat_age_s"] == 0 and st["unmatched_players"] == []
    assert views.public_status(conn, now) == {"delayed": False}
    assert views.results(conn, now, 30, None, None)["n"] == 0


def test_pulled_quotes_are_retired_on_the_next_poll(conn):
    store.upsert_events(conn, [_event()])
    t0, t1 = NOW, NOW + timedelta(minutes=5)
    store.write_quotes(conn, [q(book="draftkings", at=t0), q(book="draftkings", side="under", at=t0)])
    moved = [Quote("e1", "player_points", "Jayson Tatum", s, 28.5, "draftkings", -110, None, t1)
             for s in ("over", "under")]
    store.write_quotes(conn, moved)
    assert store.retire_missing(conn, ["e1"], ["player_points"], t1) == 2
    assert {x.line for x in store.load_quotes(conn, ["e1"])} == {28.5}


class _FakeOdds:
    def __init__(self, events):
        self.events, self.calls = events, []

    async def get_events(self):
        from src.edge.client import Usage
        return self.events, Usage(None, None, 0)

    async def get_game_odds(self):
        from src.edge.client import Usage
        return [], Usage(4_000_000, 1, 6)

    async def get_event_odds(self, eid):
        from src.edge.client import OddsApiError, Usage
        self.calls.append(eid)
        if eid == "gone":
            raise OddsApiError(404, "EVENT_NOT_FOUND")
        ev = {"id": eid, "sport_key": "basketball_nba", "home_team": "Detroit Pistons",
              "away_team": "Boston Celtics", "commence_time": "x", "bookmakers": []}
        return ev, Usage(4_000_000, 1, 16)


class _FakeBdl:
    async def roster(self, team):
        return []


@pytest.mark.asyncio
async def test_one_bad_event_does_not_stop_the_tick(conn):
    from src.edge import worker

    now = NOW.replace(hour=5)
    tip = (now + timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    evs = [{"id": i, "sport_key": "basketball_nba", "home_team": "Detroit Pistons",
            "away_team": "Boston Celtics", "commence_time": tip} for i in ("gone", "ok")]
    client = _FakeOdds(evs)
    out = await worker.tick(conn, client, _FakeBdl(), now)
    assert sorted(client.calls) == ["gone", "ok"] and "graded" in out
    status = {e["event_id"]: e for e in store.upcoming_events(conn, now)}
    assert "gone" not in status and status["ok"]["last_props_poll"] == now
    # the next tick does not retry the dead event
    client.calls.clear()
    await worker.tick(conn, client, _FakeBdl(), now + timedelta(minutes=6))
    assert client.calls == ["ok"]


@pytest.mark.asyncio
async def test_events_missing_from_the_feed_drop_out(conn):
    from src.edge import worker

    now = NOW.replace(hour=5)
    tip = (now + timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    ev = lambda i: {"id": i, "sport_key": "basketball_nba", "home_team": "Detroit Pistons",
                    "away_team": "Boston Celtics", "commence_time": tip}
    await worker.tick(conn, _FakeOdds([ev("a"), ev("b")]), _FakeBdl(), now)
    later = now + timedelta(minutes=61)
    await worker.tick(conn, _FakeOdds([ev("a")]), _FakeBdl(), later)  # b postponed
    assert [e["event_id"] for e in store.upcoming_events(conn, later)] == ["a"]


class _BoxBdl:
    """Fake BallDontLie: box scores for a date appear only once `ready` holds it."""

    def __init__(self):
        self.ready, self.calls = set(), []

    async def teams(self):
        return {2: "BOS", 9: "DET"}

    async def stats_for_date(self, d):
        self.calls.append(d)
        if d not in self.ready:
            return []
        return [{"player": {"id": 434}, "team": {"id": 2, "abbreviation": "BOS"},
                 "game": {"id": 1, "date": d.isoformat(), "season": 2026, "home_team_id": 9, "visitor_team_id": 2},
                 "min": "36", "pts": 30, "reb": 8, "ast": 6, "fg3m": 4, "stl": 1, "blk": 0, "turnover": 3},
                {"player": {"id": 777}, "team": {"id": 9, "abbreviation": "DET"},
                 "game": {"id": 1, "date": d.isoformat(), "season": 2026, "home_team_id": 9, "visitor_team_id": 2},
                 "min": "30", "pts": 20, "reb": 5, "ast": 5, "fg3m": 2, "stl": 1, "blk": 0, "turnover": 1}]


@pytest.mark.asyncio
async def test_missed_box_score_day_is_retried_until_loaded(conn):
    from src.edge import worker

    game_day = datetime(2026, 10, 20, 23, 0, tzinfo=timezone.utc)  # 7pm ET Oct 20
    store.upsert_events(conn, [{**_event(), "commence_time": game_day}])
    with conn.cursor() as cur:
        cur.execute("UPDATE edge_events SET completed = TRUE, status = 'final'")
    conn.commit()
    bdl = _BoxBdl()
    t = datetime(2026, 10, 21, 9, 0, tzinfo=timezone.utc)
    await worker.box_scores_step(conn, bdl, t)                       # BDL late: no rows yet
    assert bdl.calls[-1].isoformat() == "2026-10-20"
    n = len(bdl.calls)
    await worker.box_scores_step(conn, bdl, t + timedelta(minutes=30))  # throttled (hourly)
    assert len(bdl.calls) == n
    bdl.ready.add(game_day.date())
    await worker.box_scores_step(conn, bdl, t + timedelta(minutes=61))  # retried, now loads
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM edge_player_games WHERE game_date = '2026-10-20'")
        assert cur.fetchone()[0] == 2
    bdl.calls.clear()
    await worker.box_scores_step(conn, bdl, t + timedelta(minutes=125))  # complete: no refetch
    assert date(2026, 10, 20) not in bdl.calls


class _FlakyBdl(_BoxBdl):
    def __init__(self, fail_on):
        super().__init__()
        self.fail_on = fail_on

    async def stats_for_date(self, d):
        if d == self.fail_on:
            self.fail_on = None
            raise RuntimeError("network blip")
        self.calls.append(d)
        return []


@pytest.mark.asyncio
async def test_backfill_runs_in_steps_and_resumes_after_errors(conn):
    from src.edge import worker

    now = datetime(2026, 10, 18, 12, 0, tzinfo=timezone.utc)
    bdl = _FlakyBdl(fail_on=date(2025, 10, 18))
    await worker.backfill_step(conn, bdl, now, days=3)  # 10-15, 10-16, 10-17
    assert bdl.calls == [date(2025, 10, 15), date(2025, 10, 16), date(2025, 10, 17)]
    await worker.backfill_step(conn, bdl, now, days=3)  # 10-18 blows up: cursor stays
    assert store.get_state(conn)["backfill_2025_next"] == "2025-10-18"
    await worker.backfill_step(conn, bdl, now, days=3)  # resumes at 10-18
    assert bdl.calls[3] == date(2025, 10, 18)


def test_only_one_worker_holds_the_lock(conn):
    other = psycopg2.connect(URL)
    try:
        assert store.try_worker_lock(conn) is True
        assert store.try_worker_lock(conn) is True  # re-entrant for the holder
        assert store.try_worker_lock(other) is False
        conn.close()  # the lock dies with the holder's session
        assert store.try_worker_lock(other) is True
    finally:
        other.close()
