"""Twilio SMS transport: env gating, payload shape, error handling."""

import httpx

import src.services.notifications.sms as sms


class FakeResponse:
    def __init__(self, status_code, text=""):
        self.status_code = status_code
        self.text = text


def _set_env(monkeypatch):
    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "ACtest")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "tok")
    monkeypatch.setenv("TWILIO_FROM_NUMBER", "+15550001111")
    monkeypatch.setenv("PICK_ALERT_TO_NUMBER", "+15552223333")


def test_missing_env_no_ops(monkeypatch):
    for var in ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN",
                "TWILIO_FROM_NUMBER", "PICK_ALERT_TO_NUMBER"):
        monkeypatch.delenv(var, raising=False)
    called = []
    monkeypatch.setattr(sms.httpx, "post", lambda *a, **k: called.append(1))
    assert sms.send_sms("hi") is False
    assert not called


def test_sends_correct_payload_on_201(monkeypatch):
    _set_env(monkeypatch)
    captured = {}

    def fake_post(url, auth=None, data=None, timeout=None):
        captured.update(url=url, auth=auth, data=data)
        return FakeResponse(201)

    monkeypatch.setattr(sms.httpx, "post", fake_post)
    assert sms.send_sms("test body") is True
    assert captured["url"] == "https://api.twilio.com/2010-04-01/Accounts/ACtest/Messages.json"
    assert captured["auth"] == ("ACtest", "tok")
    assert captured["data"] == {"From": "+15550001111", "To": "+15552223333", "Body": "test body"}


def test_non_2xx_returns_false(monkeypatch):
    _set_env(monkeypatch)
    monkeypatch.setattr(sms.httpx, "post", lambda *a, **k: FakeResponse(401, "auth error"))
    assert sms.send_sms("hi") is False


def test_httpx_error_returns_false(monkeypatch):
    _set_env(monkeypatch)

    def boom(*a, **k):
        raise httpx.ConnectError("no network")

    monkeypatch.setattr(sms.httpx, "post", boom)
    assert sms.send_sms("hi") is False
