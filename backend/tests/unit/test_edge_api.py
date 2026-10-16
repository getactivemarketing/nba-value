# backend/tests/unit/test_edge_api.py
from contextlib import contextmanager

from fastapi.testclient import TestClient

from src.api import admin as admin_api
from src.config import settings
from src.edge import api as edge_api
from src.edge import views
from src.main import app

P = f"{settings.api_v1_prefix}/edge"


def test_describe():
    assert views.describe("player_points", "Jayson Tatum", "over", 27.5, "DET", "BOS") == "Jayson Tatum Over 27.5 Points"
    assert views.describe("spreads", "", "away", -5.5, "Detroit Pistons", "Boston Celtics") == "Boston Celtics +5.5"
    assert views.describe("spreads", "", "home", -5.5, "Detroit Pistons", "Boston Celtics") == "Detroit Pistons -5.5"
    assert views.describe("h2h", "", "home", 0.0, "Detroit Pistons", "Boston Celtics") == "Detroit Pistons ML"
    assert views.describe("totals", "", "under", 221.5, "Detroit Pistons", "Boston Celtics") == "Under 221.5"


def test_results_summary_ci_and_roi():
    rows = [{"clv": 0.02, "result": "win", "profit_units": 1.2, "market": "player_points", "book": "fanduel"},
            {"clv": 0.00, "result": "loss", "profit_units": -1.0, "market": "player_points", "book": "fanduel"},
            {"clv": None, "result": None, "profit_units": None, "market": "h2h", "book": "betmgm"}]
    s = views.results_summary(rows)
    assert s["n"] == 3 and s["clv"]["measured"] == 2
    assert s["clv"]["ci95"][0] < s["clv"]["mean_clv"] < s["clv"]["ci95"][1]
    assert s["record"] == {"wins": 1, "losses": 1, "pushes": 0, "voids": 0, "win_rate": 0.5,
                           "units": 0.2, "roi": 0.1}
    assert set(s["by_market"]) == {"player_points", "h2h"}


@contextmanager
def _fake_conn():
    yield object()


def test_ev_route_uses_views_and_cache(monkeypatch):
    calls = []
    monkeypatch.setattr(edge_api, "_conn", _fake_conn)
    monkeypatch.setattr(views, "ev_rows", lambda conn, now, **kw: calls.append(kw) or [{"ev": 0.05}])
    edge_api._cache.clear()
    c = TestClient(app)
    assert c.get(f"{P}/ev", params={"min_ev": 0.03}).json() == {"rows": [{"ev": 0.05}], "access_tier": "full"}
    c.get(f"{P}/ev", params={"min_ev": 0.03})
    assert len(calls) == 1  # second call served from cache


def test_status_is_admin_only(monkeypatch):
    monkeypatch.setattr(edge_api, "_conn", _fake_conn)
    monkeypatch.setattr(views, "status", lambda conn, now: {"heartbeat_age_s": 5})
    c = TestClient(app)
    assert c.get(f"{P}/status").status_code == 404
    r = c.get(f"{P}/status", headers={"X-Cron-Key": admin_api.CRON_API_KEY})
    assert r.json() == {"heartbeat_age_s": 5}


def test_errors_never_leak_the_key(monkeypatch):
    monkeypatch.setattr(settings, "odds_api_key", "leakyleakyleaky1", raising=False)
    monkeypatch.setattr(edge_api, "_conn", _fake_conn)

    def boom(conn, now, **kw):
        raise RuntimeError("upstream https://x?apiKey=leakyleakyleaky1")

    monkeypatch.setattr(views, "games", boom)
    edge_api._cache.clear()
    r = TestClient(app, raise_server_exceptions=False).get(f"{P}/games")
    assert r.status_code == 503 and "leakyleakyleaky1" not in r.text


def test_ci_clusters_the_same_bet_across_books_and_prices():
    def row(book, price, clv, player="A"):
        return {"clv": clv, "result": "win", "profit_units": 1.0, "market": "player_points", "book": book,
                "event_id": "e1", "player": player, "line": 25.5, "side": "over", "price": price}

    same_bet = [row("fanduel", 120, 0.03), row("betmgm", 125, 0.031), row("caesars", 118, 0.029)]
    s = views.results_summary(same_bet)
    assert s["clv"]["measured"] == 3 and s["clv"]["bets"] == 1
    assert s["clv"]["ci95"] is None  # one independent bet: no interval yet
    two = same_bet + [row("fanduel", 110, -0.01, player="B")]
    assert views.results_summary(two)["clv"]["bets"] == 2


# --- Hardening: bounded cache, pooled connections, per-IP rate limit.

def test_cache_is_bounded(monkeypatch):
    monkeypatch.setattr(edge_api, "_conn", _fake_conn)
    edge_api._cache.clear()
    for i in range(edge_api.MAX_CACHE_ENTRIES + 100):
        edge_api._cached(("k", i), lambda c, now, i=i: i)
    assert len(edge_api._cache) <= edge_api.MAX_CACHE_ENTRIES
    assert ("k", edge_api.MAX_CACHE_ENTRIES + 99) in edge_api._cache  # newest kept


class _FakeConn:
    def __init__(self):
        self.closed = 0
        self.rolled_back = 0

    def rollback(self):
        self.rolled_back += 1


class _FakePool:
    def __init__(self):
        self.out, self.returned = [], []

    def getconn(self):
        c = _FakeConn()
        self.out.append(c)
        return c

    def putconn(self, conn, close=False):
        self.returned.append((conn, close))


def test_connections_come_from_a_shared_pool_and_go_back(monkeypatch):
    pool = _FakePool()
    monkeypatch.setattr(edge_api, "_get_pool", lambda: pool)
    with edge_api._conn() as c:
        assert c is pool.out[0]
    assert pool.returned == [(c, False)] and c.rolled_back == 1  # no open transaction left behind


def test_a_plain_error_keeps_the_connection_but_a_dead_one_is_discarded(monkeypatch):
    import psycopg2

    pool = _FakePool()
    monkeypatch.setattr(edge_api, "_get_pool", lambda: pool)
    for exc in (RuntimeError("bad query"), psycopg2.OperationalError("server closed the connection")):
        try:
            with edge_api._conn():
                raise exc
        except type(exc):
            pass
    (c1, close1), (c2, close2) = pool.returned
    assert close1 is False and c1.rolled_back == 1  # rolled back and reused
    assert close2 is True                            # broken: thrown away


def test_pool_is_created_once_with_timeouts_and_keepalives(monkeypatch):
    made = []
    monkeypatch.setattr(edge_api, "_pool", None)
    monkeypatch.setattr(edge_api, "ThreadedConnectionPool",
                        lambda *a, **k: made.append(k) or _FakePool())
    assert edge_api._get_pool() is edge_api._get_pool() and len(made) == 1
    kw = made[0]
    assert kw["connect_timeout"] <= 10 and kw["keepalives"] == 1 and "statement_timeout" in kw["options"]


def test_a_stale_pooled_connection_is_retried_once(monkeypatch):
    import psycopg2

    pool = _FakePool()
    monkeypatch.setattr(edge_api, "_get_pool", lambda: pool)
    edge_api._cache.clear()
    calls = []

    def build(conn, now):
        calls.append(conn)
        if len(calls) == 1:
            raise psycopg2.OperationalError("SSL connection has been closed unexpectedly")
        return "ok"

    assert edge_api._cached(("stale",), build) == "ok"
    assert len(calls) == 2 and calls[0] is not calls[1]


def _health(c, headers):
    return c.get(f"{P}/health", headers=headers).status_code


def test_users_behind_the_vercel_proxy_get_their_own_budget(monkeypatch):
    """Browsers reach Railway via Vercel's rewrite: X-Real-IP is Vercel, XFF has the user."""
    monkeypatch.setattr(edge_api, "_conn", _fake_conn)
    monkeypatch.setattr(views, "public_status", lambda conn, now, sport=None: {"ok": True})
    edge_api._reset_limits()
    c = TestClient(app)
    vercel = "76.76.21.21"
    user_a = {"X-Real-IP": vercel, "X-Forwarded-For": "1.2.3.4, 76.76.21.21"}
    user_b = {"X-Real-IP": vercel, "X-Forwarded-For": "5.6.7.8"}
    codes = [_health(c, user_a) for _ in range(edge_api.RATE_BURST + 5)]
    assert codes[:edge_api.RATE_BURST] == [200] * edge_api.RATE_BURST and codes[-1] == 429
    assert _health(c, user_b) == 200  # a second user via the same proxy is unaffected
    edge_api._reset_limits()


def test_rotating_a_spoofed_forwarded_for_still_hits_the_peer_cap(monkeypatch):
    monkeypatch.setattr(edge_api, "_conn", _fake_conn)
    monkeypatch.setattr(views, "public_status", lambda conn, now, sport=None: {"ok": True})
    monkeypatch.setattr(edge_api, "_peer_limiter", edge_api.RateLimiter(1.0, 10))
    edge_api._limiter.reset()
    c = TestClient(app)
    codes = [_health(c, {"X-Real-IP": "9.9.9.9", "X-Forwarded-For": f"10.0.0.{i}"}) for i in range(12)]
    assert codes[:10] == [200] * 10 and codes[-1] == 429
    edge_api._reset_limits()


def test_admin_requests_are_not_rate_limited(monkeypatch):
    monkeypatch.setattr(edge_api, "_conn", _fake_conn)
    monkeypatch.setattr(views, "status", lambda conn, now: {"heartbeat_age_s": 5})
    monkeypatch.setattr(edge_api, "_limiter", edge_api.RateLimiter(1.0, 1))
    c = TestClient(app)
    h = {"X-Cron-Key": admin_api.CRON_API_KEY, "X-Real-IP": "1.1.1.1"}
    assert [c.get(f"{P}/status", headers=h).status_code for _ in range(3)] == [200, 200, 200]


def test_limiter_refills_and_stays_bounded():
    lim = edge_api.RateLimiter(rate_per_s=1.0, burst=2, max_clients=3)
    assert lim.allow("a", now=0) and lim.allow("a", now=0) and not lim.allow("a", now=0)
    assert lim.allow("a", now=1.0)  # one token back after a second
    for ip in "bcdef":
        lim.allow(ip, now=2.0)
    assert len(lim) <= 3


def test_unknown_sport_is_422_and_sport_is_part_of_the_cache_key(monkeypatch):
    monkeypatch.setattr(edge_api, "_conn", _fake_conn)
    seen = []
    monkeypatch.setattr(views, "games", lambda conn, now, sport: seen.append(sport.slug) or [])
    edge_api._cache.clear()
    edge_api._reset_limits()
    c = TestClient(app)
    assert c.get(f"{P}/games", params={"sport": "nhl"}).status_code == 422
    c.get(f"{P}/games")
    c.get(f"{P}/games", params={"sport": "nfl"})
    assert seen == ["nba", "nfl"]  # separate cache entries per sport
