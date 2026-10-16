"""Strip secrets from text before it reaches a response, a log line or the DB.

httpx's raise_for_status() message embeds the full request URL, and The Odds
API takes its key as a query parameter, so any `str(e)` of an Odds API error
contains the key. Everything that surfaces an error goes through redact().
"""
import re

from src.config import settings

_QUERY_SECRET = re.compile(r"(?i)\b(apiKey|api_key|token)=([^&\s'\"]+)")


def redact(text: object) -> str:
    s = _QUERY_SECRET.sub(lambda m: f"{m.group(1)}=***", str(text))
    for secret in (
        settings.odds_api_key,
        settings.balldontlie_api_key,
        settings.blotato_api_key,
        settings.typefully_api_key,
        settings.jwt_secret,
    ):
        if secret and len(secret) >= 8:
            s = s.replace(secret, "***")
    return s


class RedactingStream:
    """Wraps stdout/stderr so print() and structlog output are scrubbed too."""

    def __init__(self, stream):
        self._stream = stream

    def write(self, s):
        return self._stream.write(redact(s))

    def __getattr__(self, name):
        return getattr(self._stream, name)


_installed = False


def install_redaction() -> None:
    """Scrub every formatted log line and all stdout/stderr output, process-wide.

    Legacy code passes raw httpx errors to print() and loggers in ~110 places;
    this catches them all instead of patching each one. Redaction happens on
    the finished string in Formatter.format, so records keep their args (uvicorn's
    access formatter unpacks them) and %-format errors stay inside the handler's
    error path instead of raising into the caller. Idempotent.
    """
    global _installed
    if _installed:
        return
    _installed = True
    import logging
    import sys

    original = logging.Formatter.format

    def format(self, record):  # noqa: A001 - mirrors logging.Formatter.format
        return redact(original(self, record))

    logging.Formatter.format = format
    sys.stdout = RedactingStream(sys.stdout)
    sys.stderr = RedactingStream(sys.stderr)
