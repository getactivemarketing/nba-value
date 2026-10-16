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
