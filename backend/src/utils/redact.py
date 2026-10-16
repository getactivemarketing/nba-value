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
