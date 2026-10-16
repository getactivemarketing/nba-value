from fastapi.testclient import TestClient

from src.api import admin as admin_api
from src.config import settings
from src.main import app

P = settings.api_v1_prefix
LEGACY = [f"{P}/markets", f"{P}/markets/live", f"{P}/evaluation/summary",
          f"{P}/props/top", f"{P}/games/upcoming", f"{P}/trends/model", f"{P}/backtest/results"]


def test_legacy_nba_pick_routes_404_when_not_public(monkeypatch):
    monkeypatch.setattr(settings, "nba_picks_public", False, raising=False)
    client = TestClient(app)
    for path in LEGACY:
        assert client.get(path).status_code == 404, path


def test_admin_key_still_reaches_them(monkeypatch):
    # Checked at the dependency, not through a route: every legacy handler opens
    # a DB connection, and there is no DB in unit tests.
    import pytest
    from fastapi import HTTPException
    from src.api.deps import require_nba_picks_public

    monkeypatch.setattr(settings, "nba_picks_public", False, raising=False)
    assert require_nba_picks_public(admin_api.CRON_API_KEY) is None
    with pytest.raises(HTTPException):
        require_nba_picks_public("wrong-key")
    monkeypatch.setattr(settings, "nba_picks_public", True, raising=False)
    assert require_nba_picks_public(None) is None


def test_mlb_and_nfl_are_untouched(monkeypatch):
    monkeypatch.setattr(settings, "nba_picks_public", False, raising=False)
    schema = TestClient(app).get("/openapi.json").json()
    assert f"{P}/nfl/games" in schema["paths"]
    assert any(p.startswith(f"{P}/mlb/") for p in schema["paths"])


def test_social_skips_nba_when_not_public(monkeypatch):
    from src.tasks import social_scheduler as ss
    monkeypatch.setattr(settings, "nba_picks_public", False, raising=False)
    assert ss.nba_posting_allowed() is False
    monkeypatch.setattr(settings, "nba_picks_public", True, raising=False)
    assert ss.nba_posting_allowed() is True
