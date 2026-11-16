# backend/tests/unit/test_edge_signups.py
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

from src.api import admin as admin_api
from src.config import settings
from src.edge import api as edge_api
from src.edge import signups
from src.main import app

P = f"{settings.api_v1_prefix}/edge"


@pytest.mark.parametrize("raw,expected", [
    ("Fan@Example.com", "fan@example.com"),
    ("  bettor+nba@mail.co.uk ", "bettor+nba@mail.co.uk"),
    ("", None),
    ("not-an-email", None),
    ("a@b", None),
    ("two@@example.com", None),
    ("spaces in@example.com", None),
    ("x" * 250 + "@example.com", None),
])
def test_normalize_email(raw, expected):
    assert signups.normalize_email(raw) == expected


def test_normalize_source_keeps_known_values_only():
    assert signups.normalize_source("edge_home") == "edge_home"
    assert signups.normalize_source("anything else") == "other"
    assert signups.normalize_source(None) == "other"


@contextmanager
def _fake_conn():
    yield object()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(edge_api, "_conn", _fake_conn)
    edge_api._reset_limits()
    yield TestClient(app)
    edge_api._reset_limits()


def test_signup_stores_normalized_email(client, monkeypatch):
    saved = []
    monkeypatch.setattr(signups, "add", lambda conn, email, source: saved.append((email, source)) or True)
    r = client.post(f"{P}/signup", json={"email": "Fan@Example.com", "source": "edge_home"})
    assert r.status_code == 200 and r.json() == {"ok": True}
    assert saved == [("fan@example.com", "edge_home")]


def test_signup_does_not_reveal_an_existing_email(client, monkeypatch):
    monkeypatch.setattr(signups, "add", lambda conn, email, source: False)
    r = client.post(f"{P}/signup", json={"email": "fan@example.com"})
    assert r.status_code == 200 and r.json() == {"ok": True}


def test_signup_rejects_a_bad_email(client, monkeypatch):
    monkeypatch.setattr(signups, "add", lambda *a: pytest.fail("must not store"))
    assert client.post(f"{P}/signup", json={"email": "nope"}).status_code == 422


def test_signup_honeypot_is_accepted_but_not_stored(client, monkeypatch):
    monkeypatch.setattr(signups, "add", lambda *a: pytest.fail("bots must not be stored"))
    r = client.post(f"{P}/signup", json={"email": "bot@example.com", "website": "http://spam"})
    assert r.status_code == 200 and r.json() == {"ok": True}


def test_signup_has_its_own_tight_rate_limit(client, monkeypatch):
    monkeypatch.setattr(signups, "add", lambda conn, email, source: True)
    codes = [client.post(f"{P}/signup", json={"email": f"f{i}@example.com"}).status_code
             for i in range(signups.SIGNUP_BURST + 1)]
    assert codes[:-1] == [200] * signups.SIGNUP_BURST and codes[-1] == 429


def test_signup_list_is_admin_only(client, monkeypatch):
    monkeypatch.setattr(signups, "recent", lambda conn, limit: [{"email": "fan@example.com"}])
    monkeypatch.setattr(signups, "count", lambda conn: 1)
    assert client.get(f"{P}/signups").status_code == 404
    r = client.get(f"{P}/signups", headers={"X-Cron-Key": admin_api.CRON_API_KEY})
    assert r.json() == {"count": 1, "signups": [{"email": "fan@example.com"}]}
