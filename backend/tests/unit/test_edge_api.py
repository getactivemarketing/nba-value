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
