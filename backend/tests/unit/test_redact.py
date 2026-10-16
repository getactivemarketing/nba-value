from fastapi.testclient import TestClient

from src.config import settings
from src.main import app
from src.utils.redact import redact


def test_query_param_keys_are_masked():
    msg = "Client error '401' for url 'https://api.the-odds-api.com/v4/sports/x/odds?apiKey=b1f7f077c2189762&regions=us'"
    out = redact(msg)
    assert "b1f7f077c2189762" not in out
    assert "apiKey=***" in out
    assert "regions=us" in out


def test_configured_secret_is_masked_anywhere(monkeypatch):
    monkeypatch.setattr(settings, "odds_api_key", "abcd1234efgh5678", raising=False)
    assert "abcd1234efgh5678" not in redact("boom abcd1234efgh5678 boom")


def test_short_or_empty_secrets_do_not_mangle_text(monkeypatch):
    monkeypatch.setattr(settings, "odds_api_key", "", raising=False)
    assert redact("plain message") == "plain message"


def test_nfl_debug_requires_admin_key():
    client = TestClient(app)
    r = client.get(f"{settings.api_v1_prefix}/nfl/debug/odds", params={"live": "false"})
    assert r.status_code == 404


def test_mlb_debug_requires_admin_key():
    client = TestClient(app)
    assert client.get(f"{settings.api_v1_prefix}/mlb/debug/odds").status_code == 404
