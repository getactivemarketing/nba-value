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
    """Scrub every log record and all stdout/stderr output, process-wide.

    Legacy code passes raw httpx errors to print() and loggers in ~110 places;
    this catches them all instead of patching each one. Idempotent.
    """
    global _installed
    if _installed:
        return
    _installed = True
    import logging
    import sys

    base = logging.getLogRecordFactory()
    fmt = logging.Formatter()

    def factory(*args, **kwargs):
        record = base(*args, **kwargs)
        record.msg = redact(record.getMessage())
        record.args = ()
        if record.exc_info and record.exc_info[0] is not None:
            record.exc_text = redact(fmt.formatException(record.exc_info))
            record.exc_info = None
        return record

    logging.setLogRecordFactory(factory)
    sys.stdout = RedactingStream(sys.stdout)
    sys.stderr = RedactingStream(sys.stderr)
