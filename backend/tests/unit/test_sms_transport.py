"""Textbelt SMS transport: env gating, payload shape, success-field handling."""

import httpx

import src.services.notifications.sms as sms


class FakeResponse:
    def __init__(self, status_code, json_data=None, text=""):
        self.status_code = status_code
        self._json = json_data
        self.text = text

    def json(self):
        if self._json is None:
            raise ValueError("no JSON")
        return self._json


def _set_env(monkeypatch):
    monkeypatch.setenv("TEXTBELT_API_KEY", "testkey123")
    monkeypatch.setenv("PICK_ALERT_TO_NUMBER", "+15552223333")


def test_missing_env_no_ops(monkeypatch):
    for var in ("TEXTBELT_API_KEY", "PICK_ALERT_TO_NUMBER"):
        monkeypatch.delenv(var, raising=False)
    called = []
    monkeypatch.setattr(sms.httpx, "post", lambda *a, **k: called.append(1))
    assert sms.send_sms("hi") is False
    assert not called


def test_sends_correct_payload_on_success(monkeypatch):
    _set_env(monkeypatch)
    captured = {}

    def fake_post(url, data=None, timeout=None):
        captured.update(url=url, data=data)
        return FakeResponse(200, {"success": True, "quotaRemaining": 500, "textId": "123"})

    monkeypatch.setattr(sms.httpx, "post", fake_post)
    assert sms.send_sms("test body") is True
    assert captured["url"] == "https://textbelt.com/text"
    assert captured["data"] == {"phone": "+15552223333", "message": "test body", "key": "testkey123"}


def test_success_false_returns_false(monkeypatch):
    _set_env(monkeypatch)
    monkeypatch.setattr(
        sms.httpx, "post",
        lambda *a, **k: FakeResponse(200, {"success": False, "error": "Out of quota"}),
    )
    assert sms.send_sms("hi") is False


def test_non_2xx_returns_false(monkeypatch):
    _set_env(monkeypatch)
    monkeypatch.setattr(sms.httpx, "post", lambda *a, **k: FakeResponse(503, None, "unavailable"))
    assert sms.send_sms("hi") is False


def test_non_json_response_returns_false(monkeypatch):
    _set_env(monkeypatch)
    monkeypatch.setattr(sms.httpx, "post", lambda *a, **k: FakeResponse(200, None, "<html>"))
    assert sms.send_sms("hi") is False


def test_httpx_error_returns_false(monkeypatch):
    _set_env(monkeypatch)

    def boom(*a, **k):
        raise httpx.ConnectError("no network")

    monkeypatch.setattr(sms.httpx, "post", boom)
    assert sms.send_sms("hi") is False


def test_low_quota_still_returns_true(monkeypatch):
    _set_env(monkeypatch)
    monkeypatch.setattr(
        sms.httpx, "post",
        lambda *a, **k: FakeResponse(200, {"success": True, "quotaRemaining": 5}),
    )
    assert sms.send_sms("hi") is True
