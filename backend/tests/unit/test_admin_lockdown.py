"""Every /admin route needs the cron key, and nothing it returns carries a secret."""
from fastapi.testclient import TestClient

from src.api import admin as admin_api
from src.config import settings
from src.main import app

P = f"{settings.api_v1_prefix}/admin"
KEY = "leakyleakyleaky1"


def test_admin_routes_404_without_key():
    c = TestClient(app)
    for method, path in (("post", "/ingest/odds"), ("post", "/ingest/stats"), ("post", "/sync/results"),
                         ("post", "/scoring/run"), ("post", "/stats/calculate"),
                         ("post", "/backfill/games"), ("get", "/status")):
        assert getattr(c, method)(f"{P}{path}").status_code == 404, path


def test_ingest_odds_error_is_redacted(monkeypatch):
    monkeypatch.setattr(settings, "odds_api_key", KEY, raising=False)

    async def fake_ingest():
        return {"status": "error", "games_fetched": 0, "snapshots_stored": 0,
                "error": f"Client error '401' for url 'https://api.the-odds-api.com/v4/x?apiKey={KEY}&regions=us'"}

    monkeypatch.setattr(admin_api, "_ingest_odds_async", fake_ingest)
    r = TestClient(app).post(f"{P}/ingest/odds", headers={"X-Cron-Key": admin_api.CRON_API_KEY})
    assert r.status_code == 200 and KEY not in r.text


def test_admin_exception_detail_is_redacted(monkeypatch):
    monkeypatch.setattr(settings, "odds_api_key", KEY, raising=False)

    async def boom():
        raise RuntimeError(f"upstream https://x?apiKey={KEY}")

    monkeypatch.setattr(admin_api, "_update_nba_stats_async", boom)
    r = TestClient(app).post(f"{P}/ingest/stats", headers={"X-Cron-Key": admin_api.CRON_API_KEY})
    assert r.status_code == 500 and KEY not in r.text


def test_mlb_debug_tracebacks_are_redacted():
    import inspect
    from src.api import mlb
    src = inspect.getsource(mlb)
    assert "traceback.format_exc()" not in src.replace("redact(traceback.format_exc())", "")
