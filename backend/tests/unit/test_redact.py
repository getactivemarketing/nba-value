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


# --- Process-wide scrubbing: ~110 legacy print/log sites pass raw httpx errors,
# whose message embeds the request URL and therefore the Odds API key.

import io
import logging

from src.utils.redact import RedactingStream, install_redaction

LEAK = "Client error '401' for url 'https://api.the-odds-api.com/v4/sports?apiKey=zz9988776655&x=1'"


def test_redacting_stream_scrubs_print_output():
    raw = io.StringIO()
    out = RedactingStream(raw)
    print(LEAK, file=out)
    assert "zz9988776655" not in raw.getvalue()
    assert "apiKey=***" in raw.getvalue()


def test_log_records_are_scrubbed_including_args_and_tracebacks():
    install_redaction()
    install_redaction()  # idempotent
    raw = io.StringIO()
    handler = logging.StreamHandler(raw)  # bound to a plain stream, not a wrapped one
    log = logging.getLogger("test.redaction")
    log.addHandler(handler)
    try:
        log.error("odds failed: %s", LEAK)
        try:
            raise RuntimeError(LEAK)
        except RuntimeError:
            log.exception("tick failed")
    finally:
        log.removeHandler(handler)
    text = raw.getvalue()
    assert "zz9988776655" not in text
    assert text.count("apiKey=***") == 2  # message arg + exception line in the traceback
    assert "Traceback" in text


import subprocess
import sys as _sys

import pytest


@pytest.mark.parametrize("setup", ["import src.main", "import src.edge.worker as w; w.configure_logging()"])
def test_entry_points_install_scrubbing(setup):
    code = f"{setup}\nimport logging\nprint({LEAK!r})\nlogging.getLogger('x').error({LEAK!r})"
    p = subprocess.run([_sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr[-500:]
    assert "apiKey=***" in p.stdout
    assert "zz9988776655" not in p.stdout + p.stderr


def test_uvicorn_access_log_still_formats():
    """Uvicorn's AccessFormatter unpacks record.args; scrubbing must not clear them."""
    from uvicorn.logging import AccessFormatter

    install_redaction()
    raw = io.StringIO()
    handler = logging.StreamHandler(raw)
    handler.setFormatter(AccessFormatter('%(client_addr)s - "%(request_line)s" %(status_code)s'))
    log = logging.getLogger("test.uvicorn.access")
    log.setLevel(logging.INFO)
    log.addHandler(handler)
    log.propagate = False
    try:
        log.info('%s - "%s %s HTTP/%s" %d', "1.2.3.4", "GET", "/x?apiKey=zz9988776655", "1.1", 200)
    finally:
        log.removeHandler(handler)
    assert '1.2.3.4 - "GET /x?apiKey=*** HTTP/1.1" 200' in raw.getvalue()


def test_bad_log_format_does_not_raise_into_the_caller():
    install_redaction()
    log = logging.getLogger("test.badformat")
    log.addHandler(logging.NullHandler())
    log.propagate = False
    log.warning("value %d", "notanint")  # must not raise TypeError here


def test_backtest_errors_do_not_echo_the_key(monkeypatch):
    from src.api import admin as admin_api
    from src.api import backtest

    def boom():
        raise RuntimeError("upstream https://x?apiKey=zz9988776655")

    monkeypatch.setattr(backtest, "get_backtest_results", lambda *a, **k: boom())
    paths = [r.path for r in app.routes if getattr(r, "path", "").endswith("/results")
             and "backtest" in getattr(r, "path", "")]
    r = TestClient(app, raise_server_exceptions=False).get(
        paths[0], headers={"X-Cron-Key": admin_api.CRON_API_KEY})
    assert r.status_code == 500 and "zz9988776655" not in r.text
