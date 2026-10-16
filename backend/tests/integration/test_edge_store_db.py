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
    def __init__(self, events, nfl_events=(), fail_nfl=None):
        self.events, self.nfl_events, self.fail_nfl = events, list(nfl_events), fail_nfl
        self.calls = []

    async def get_events(self, sport=None):
        from src.edge.client import OddsApiError, Usage
        if sport is not None and sport.slug == "nfl":
            if self.fail_nfl:
                raise OddsApiError(self.fail_nfl, "nfl upstream error")
            return self.nfl_events, Usage(None, None, 0)
        return self.events, Usage(None, None, 0)

    async def get_game_odds(self, sport=None):
        from src.edge.client import Usage
        return [], Usage(4_000_000, 1, 6)

    async def get_event_odds(self, eid, sport=None):
        from src.edge.client import OddsApiError, Usage
        self.calls.append(eid)
        if eid == "gone":
            raise OddsApiError(404, "EVENT_NOT_FOUND")
        key = sport.key if sport is not None else "basketball_nba"
        ev = {"id": eid, "sport_key": key, "home_team": "Detroit Pistons",
              "away_team": "Boston Celtics", "commence_time": "x", "bookmakers": []}
        return ev, Usage(4_000_000, 1, 16)

    async def get_scores(self, days_from=1, sport=None):
        from src.edge.client import Usage
        self.calls.append(f"scores:{sport.slug if sport else 'nba'}")
        return [], Usage(4_000_000, 1, 2)


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
        conn.close()  # the lock dies with the holder's session (released asynchronously)
        import time
        deadline = time.monotonic() + 3
        while not store.try_worker_lock(other) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert store.try_worker_lock(other) is True
    finally:
        other.close()


def test_api_serves_from_the_real_pool(conn, monkeypatch):
    from fastapi.testclient import TestClient

    from src.config import settings
    from src.edge import api as edge_api
    from src.main import app

    monkeypatch.setattr(edge_api, "_pool", None)
    monkeypatch.setattr(edge_api, "get_sync_database_url", lambda: URL)
    edge_api._cache.clear()
    edge_api._limiter.reset()
    c = TestClient(app)
    for path in ("/edge/games", "/edge/ev", "/edge/health", "/edge/results"):
        edge_api._cache.clear()
        r = c.get(f"{settings.api_v1_prefix}{path}")
        assert r.status_code == 200, (path, r.text[:300])
    pool = edge_api._pool
    assert pool is not None and len(pool._used) == 0  # every connection went back
    pool.closeall()


def _hist(conn, event_id, rows):
    with conn.cursor() as cur:
        for book, price, at in rows:
            cur.execute("""INSERT INTO edge_quote_history (event_id, market, player, side, line, book, price,
                           link, observed_at) VALUES (%s,'player_points','Jayson Tatum','over',27.5,%s,%s,NULL,%s)""",
                        (event_id, book, price, at))
    conn.commit()


def test_prune_history_keeps_open_close_and_hourly(conn):
    old_tip = NOW - timedelta(days=5)
    store.upsert_events(conn, [dict(_event(), event_id="old", commence_time=old_tip),
                               dict(_event(), event_id="new", commence_time=NOW + timedelta(hours=2))])
    t0 = old_tip - timedelta(hours=3)
    m = lambda mins: t0 + timedelta(minutes=mins)  # noqa: E731
    _hist(conn, "old", [("fanduel", -110, m(0)), ("fanduel", -112, m(10)), ("fanduel", -115, m(20)),
                        ("fanduel", -118, m(65)), ("fanduel", -120, m(90)), ("fanduel", -125, m(170)),
                        ("betmgm", -105, m(5))])
    _hist(conn, "new", [("fanduel", -110, NOW - timedelta(minutes=m_)) for m_ in (30, 20, 10)])

    deleted = store.prune_history(conn, before=NOW - timedelta(days=3))
    with conn.cursor() as cur:
        cur.execute("SELECT book, price FROM edge_quote_history WHERE event_id='old' ORDER BY observed_at")
        kept = cur.fetchall()
        cur.execute("SELECT COUNT(*) FROM edge_quote_history WHERE event_id='new'")
        new_count = cur.fetchone()[0]
    # fanduel: open (-110), first of the next hour (-118), close (-125); betmgm's only row stays.
    assert kept == [("fanduel", -110), ("betmgm", -105), ("fanduel", -118), ("fanduel", -125)]
    assert deleted == 3 and new_count == 3  # games inside the window are untouched
    assert store.prune_history(conn, before=NOW - timedelta(days=3)) == 0  # idempotent


def test_prune_history_respects_the_lower_bound(conn):
    tip = NOW - timedelta(days=10)
    store.upsert_events(conn, [dict(_event(), event_id="ancient", commence_time=tip)])
    _hist(conn, "ancient", [("fanduel", p, tip - timedelta(minutes=30 - i)) for i, p in enumerate((-110, -112, -114))])
    assert store.prune_history(conn, before=NOW, after=NOW - timedelta(days=5)) == 0
    assert store.prune_history(conn, before=NOW) == 1


def test_worker_prunes_once_a_day_from_its_cursor(conn):
    from src.edge import worker

    tip = NOW - timedelta(days=4)
    store.upsert_events(conn, [dict(_event(), event_id="old", commence_time=tip)])
    _hist(conn, "old", [("fanduel", p, tip - timedelta(minutes=30 - i)) for i, p in enumerate((-110, -112, -114))])
    assert worker.prune_step(conn, NOW) == 1
    _hist(conn, "old", [("betmgm", p, tip - timedelta(minutes=30 - i)) for i, p in enumerate((-110, -112, -114))])
    assert worker.prune_step(conn, NOW + timedelta(hours=1)) == 0  # ran today already
    assert worker.prune_step(conn, NOW + timedelta(days=1)) == 1   # next day; overlap catches stragglers
    assert store.get_state(conn)["history_pruned_before"]


def test_a_stale_lock_holder_is_evicted(conn):
    holder = psycopg2.connect(URL)
    try:
        assert store.try_worker_lock(holder)
        store.set_state(holder, "heartbeat_at", (NOW - timedelta(minutes=3)).isoformat(), NOW)
        assert store.take_over_stale_lock(conn, NOW, timedelta(minutes=10)) is False  # alive: leave it
        assert store.try_worker_lock(conn) is False
        store.set_state(holder, "heartbeat_at", (NOW - timedelta(minutes=20)).isoformat(), NOW)
        assert store.take_over_stale_lock(conn, NOW, timedelta(minutes=10)) is True
        import time  # pg_terminate_backend releases the lock asynchronously
        deadline = time.monotonic() + 3
        while not store.try_worker_lock(conn) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert store.try_worker_lock(conn) is True
    finally:
        holder.close()


def test_a_failing_prune_backs_off_instead_of_blocking_the_tick(conn, monkeypatch):
    from src.edge import worker

    def boom(*a, **k):
        raise psycopg2.errors.QueryCanceled("canceling statement due to statement timeout")

    monkeypatch.setattr(store, "prune_history", boom)
    assert worker.prune_step(conn, NOW) == 0  # no exception escapes into tick
    state = store.get_state(conn)              # and the connection is still usable
    assert state["last_prune"] and "QueryCanceled" in state["prune_error"]
    assert worker.prune_step(conn, NOW + timedelta(hours=1)) == 0  # backs off, no retry storm


def test_first_prune_is_bounded(conn):
    from src.edge import worker

    tip = NOW - timedelta(days=30)
    store.upsert_events(conn, [dict(_event(), event_id="ancient", commence_time=tip)])
    _hist(conn, "ancient", [("fanduel", p, tip - timedelta(minutes=30 - i)) for i, p in enumerate((-110, -112, -114))])
    assert worker.prune_step(conn, NOW) == 0  # first run looks back a bounded window only


def _nfl_event(event_id="n1", hours=2):
    return {"event_id": event_id, "sport": "americanfootball_nfl", "home_team": "Atlanta Falcons",
            "away_team": "Chicago Bears", "commence_time": NOW + timedelta(hours=hours)}


def test_one_sports_refresh_never_drops_the_others_events(conn):
    store.upsert_events(conn, [_event(), _nfl_event()])
    # NBA's feed lists only e1: NFL's n1 must survive.
    store.drop_missing_events(conn, ["e1"], NOW, sport="basketball_nba")
    nfl = store.upcoming_events(conn, NOW, hours=168, sport="americanfootball_nfl")
    nba = store.upcoming_events(conn, NOW, sport="basketball_nba")
    assert [e["event_id"] for e in nfl] == ["n1"] and [e["event_id"] for e in nba] == ["e1"]


def test_games_in_progress_is_per_sport(conn):
    store.upsert_events(conn, [_nfl_event(hours=-1)])
    from src.edge import grading
    assert grading.games_in_progress(conn, NOW, "americanfootball_nfl") is True
    assert grading.games_in_progress(conn, NOW, "basketball_nba") is False


def test_nfl_props_are_never_graded_even_when_the_name_matches_an_nba_player(conn):
    from src.edge import grading
    store.upsert_events(conn, [_nfl_event(hours=-5)])
    with conn.cursor() as cur:
        cur.execute("""INSERT INTO edge_players (odds_name, bdl_player_id, team, match_method, confidence, updated_at)
                       VALUES ('Josh Allen', 999, 'ORL', 'exact', 1.0, %s)""", (NOW,))
        cur.execute("""INSERT INTO edge_opportunities (event_id, market, player, side, line, book, price,
                       first_seen_at, last_seen_at, fair_prob, ev, max_ev, n_books, devig_method, closing_done)
                       VALUES ('n1', 'player_pass_yds', 'Josh Allen', 'over', 250.5, 'fanduel', 110,
                               %s, %s, 0.5, 0.05, 0.05, 5, 'power', TRUE)""", (NOW, NOW))
        # The NBA Josh Allen played that night: unscoped grading would find this row.
        from src.edge.players import et_date
        cur.execute("""INSERT INTO edge_player_games (bdl_player_id, bdl_game_id, season, game_date, team,
                       opponent, is_home, minutes, pts, reb, ast, fg3m, stl, blk, tov)
                       VALUES (999, 1, 2026, %s, 'ORL', 'MIA', TRUE, 30, 12, 5, 3, 1, 1, 0, 2)""",
                    (et_date(NOW - timedelta(hours=5)),))
    conn.commit()
    grading.run_grading(conn, NOW + timedelta(hours=1))
    with conn.cursor() as cur:
        cur.execute("SELECT result FROM edge_opportunities WHERE event_id = 'n1'")
        assert cur.fetchone()[0] is None


def _nfl_feed_event(i, tip):
    return {"id": i, "sport_key": "americanfootball_nfl", "home_team": "Atlanta Falcons",
            "away_team": "Chicago Bears", "commence_time": tip}


@pytest.mark.asyncio
async def test_tick_polls_both_sports_and_keeps_nba_state_names(conn):
    from src.edge import worker
    from src.edge.config import NBA, NFL
    now = NOW.replace(hour=5)
    tip = (now + timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    nba = [{"id": "b1", "sport_key": "basketball_nba", "home_team": "Detroit Pistons",
            "away_team": "Boston Celtics", "commence_time": tip}]
    client = _FakeOdds(nba, nfl_events=[_nfl_feed_event("f1", tip)])
    out = await worker.tick(conn, client, _FakeBdl(), now, sports=[NBA, NFL])
    assert sorted(client.calls) == ["b1", "f1"] and "nba" in out and "nfl" in out
    state = store.get_state(conn)
    assert "last_events_refresh" in state and "nfl:last_events_refresh" in state
    # NBA's refresh did not drop the NFL game, and vice versa
    assert {e["event_id"] for e in store.upcoming_events(conn, now, hours=168)} == {"b1", "f1"}


@pytest.mark.asyncio
async def test_an_nfl_failure_does_not_stop_nba(conn):
    from src.edge import worker
    from src.edge.config import NBA, NFL
    now = NOW.replace(hour=5)
    tip = (now + timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    nba = [{"id": "b1", "sport_key": "basketball_nba", "home_team": "Detroit Pistons",
            "away_team": "Boston Celtics", "commence_time": tip}]
    client = _FakeOdds(nba, fail_nfl=500)
    await worker.tick(conn, client, _FakeBdl(), now, sports=[NBA, NFL])
    assert client.calls == ["b1"]
    assert "500" in store.get_state(conn)["sport_error:nfl"]


@pytest.mark.asyncio
async def test_account_wide_errors_still_raise_for_global_backoff(conn):
    from src.edge import worker
    from src.edge.client import OddsApiError
    from src.edge.config import NBA, NFL
    with pytest.raises(OddsApiError):
        await worker.tick(conn, _FakeOdds([], fail_nfl=401), _FakeBdl(), NOW, sports=[NBA, NFL])


def _seed_two_sports(conn):
    """One NBA and one NFL game 2 h out, each with a clear +EV over at fanduel."""
    store.upsert_events(conn, [_event(), _nfl_event()])

    def qs(eid, market, player):
        return [Quote(eid, market, player, side, 10.5, book, price, "https://x", NOW - timedelta(minutes=1))
                for book, over, under in (("pinnacle", -110, -110), ("novig", -110, -110),
                                          ("draftkings", -112, -108), ("fanduel", 140, -170))
                for side, price in (("over", over), ("under", under))]
    store.write_quotes(conn, qs("e1", "player_points", "Jayson Tatum") + qs("n1", "player_rush_yds", "Bijan Robinson"))


def test_views_never_mix_sports(conn):
    from src.edge import views
    from src.edge.config import NBA, NFL
    _seed_two_sports(conn)
    nba = views.ev_rows(conn, NOW, min_ev=0.0, sport=NBA)
    nfl = views.ev_rows(conn, NOW, min_ev=0.0, sport=NFL)
    assert nba and {r["event_id"] for r in nba} == {"e1"}
    assert nfl and {r["event_id"] for r in nfl} == {"n1"}
    assert all(r["bdl_player_id"] is None for r in nfl)
    assert nfl[0]["market_label"] == "Rush Yds"
    # an NFL event id under NBA returns nothing
    assert views.ev_rows(conn, NOW, min_ev=0.0, event_id="n1", sport=NBA) == []
    assert views.odds(conn, NOW, "n1", "player_rush_yds", sport=NBA)["rows"] == []
    assert views.odds(conn, NOW, "n1", "player_rush_yds", sport=NFL)["rows"]
    assert [g["event_id"] for g in views.games(conn, NOW, sport=NFL)] == ["n1"]
    assert [g["event_id"] for g in views.games(conn, NOW, sport=NBA)] == ["e1"]


def test_nfl_lines_two_days_out_are_priced(conn):
    from src.edge import views
    from src.edge.config import NFL
    store.upsert_events(conn, [_nfl_event(hours=48)])
    qs = [Quote("n1", "spreads", "", side, -3.5, book, price, None, NOW - timedelta(minutes=20))
          for book, h, a in (("pinnacle", -110, -110), ("novig", -110, -110), ("draftkings", -110, -110),
                             ("fanduel", 130, -160))
          for side, price in (("home", h), ("away", a))]
    store.write_quotes(conn, qs)
    rows = views.ev_rows(conn, NOW, min_ev=0.0, sport=NFL)
    assert any(r["market"] == "spreads" and r["book"] == "fanduel" for r in rows)


def test_results_are_per_sport(conn):
    from src.edge import views
    from src.edge.config import NBA, NFL
    store.upsert_events(conn, [_event(), _nfl_event()])
    with conn.cursor() as cur:
        for eid in ("e1", "n1"):
            cur.execute("""INSERT INTO edge_opportunities (event_id, market, player, side, line, book, price,
                           first_seen_at, last_seen_at, fair_prob, ev, max_ev, n_books, devig_method,
                           closing_done, clv) VALUES (%s, 'h2h', '', 'home', 0, 'fanduel', 110, %s, %s,
                           0.5, 0.05, 0.05, 5, 'power', TRUE, 0.01)""", (eid, NOW, NOW))
    conn.commit()
    assert views.results(conn, NOW, 30, None, None, sport=NBA)["n"] == 1
    assert views.results(conn, NOW, 30, None, None, sport=NFL)["n"] == 1


@pytest.mark.asyncio
async def test_a_long_game_still_gets_its_final_score(conn):
    """MNF overtime ends ~3h40m after kickoff; the next NFL game is days away.
    Scores must keep polling until the game is final, not stop at kickoff + 4 h."""
    from src.edge import worker
    from src.edge.config import NFL
    kickoff = NOW - timedelta(hours=4, minutes=10)
    store.upsert_events(conn, [dict(_nfl_event(), commence_time=kickoff)])
    store.set_state(conn, "nfl:last_scores_poll", (NOW - timedelta(minutes=40)).isoformat(), NOW)
    store.set_state(conn, "nfl:last_events_refresh", NOW.isoformat(), NOW)
    client = _FakeOdds([])
    await worker.tick(conn, client, _FakeBdl(), NOW, sports=[NFL])
    assert "scores:nfl" in client.calls


class _FakeNflBdl:
    def __init__(self, rosters, fail=None):
        self.rosters, self.fail, self.calls = rosters, fail, []

    async def roster(self, team):
        from src.edge.players import BdlError
        self.calls.append(team)
        if self.fail:
            raise BdlError(self.fail, f"BallDontLie NFL {self.fail} rate limited")
        return self.rosters.get(team, [])


def _q(player, eid="n1", market="player_rush_yds"):
    return Quote(eid, market, player, "over", 60.5, "fanduel", -110, None, NOW)


@pytest.mark.asyncio
async def test_nfl_names_resolve_per_team_and_never_touch_nba_players(conn):
    from src.edge import nfl_players
    ev = {"home_team": "Atlanta Falcons", "away_team": "Chicago Bears"}
    rosters = {"ATL": [{"id": 10, "first_name": "Bijan", "last_name": "Robinson", "team_abbr": "ATL", "position": "RB"}],
               "CHI": [{"id": 20, "first_name": "D.J.", "last_name": "Moore", "team_abbr": "CHI", "position": "WR"}]}
    n = await nfl_players.resolve_new_players(conn, _FakeNflBdl(rosters), [_q("Bijan Robinson"), _q("DJ Moore"),
                                                                          _q("Nobody Here")], ev, NOW)
    assert n == 3
    with conn.cursor() as cur:
        cur.execute("SELECT odds_name, team, bdl_player_id, match_method FROM edge_nfl_players ORDER BY odds_name")
        rows = cur.fetchall()
        cur.execute("SELECT COUNT(*) FROM edge_players")
        assert cur.fetchone()[0] == 0  # NBA table untouched
    assert ("Bijan Robinson", "ATL", 10, "exact") in rows
    assert ("DJ Moore", "CHI", 20, "normalized") in rows
    assert any(r[0] == "Nobody Here" and r[3] == "unmatched" for r in rows)
    assert nfl_players.player_ids(conn, {"ATL", "CHI"}) == {"Bijan Robinson": 10, "DJ Moore": 20}


@pytest.mark.asyncio
async def test_a_traded_player_resolves_again_on_his_new_team(conn):
    from src.edge import nfl_players
    old = {"KC": [{"id": 30, "first_name": "Amari", "last_name": "Cooper", "team_abbr": "KC", "position": "WR"}]}
    await nfl_players.resolve_new_players(conn, _FakeNflBdl(old), [_q("Amari Cooper")],
                                          {"home_team": "Kansas City Chiefs", "away_team": "Denver Broncos"}, NOW)
    new = {"BUF": [{"id": 30, "first_name": "Amari", "last_name": "Cooper", "team_abbr": "BUF", "position": "WR"}]}
    n = await nfl_players.resolve_new_players(conn, _FakeNflBdl(new), [_q("Amari Cooper")],
                                              {"home_team": "Buffalo Bills", "away_team": "Miami Dolphins"}, NOW)
    assert n == 1 and nfl_players.player_ids(conn, {"BUF"}) == {"Amari Cooper": 30}


@pytest.mark.asyncio
async def test_a_roster_error_leaves_names_unseen_not_unmatched(conn):
    from src.edge import nfl_players
    n = await nfl_players.resolve_new_players(conn, _FakeNflBdl({}, fail=429), [_q("Bijan Robinson")],
                                              {"home_team": "Atlanta Falcons", "away_team": "Chicago Bears"}, NOW)
    assert n == 0
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM edge_nfl_players")
        assert cur.fetchone()[0] == 0
