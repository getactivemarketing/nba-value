# backend/src/edge/client.py
"""The Odds API v4 client for Edge. The key travels as a query param, so every
error message is redacted before it leaves this module."""
from __future__ import annotations

from dataclasses import dataclass

import httpx

from src.config import settings
from src.edge.config import BOOKS, LINE_MARKETS, PROP_MARKETS, SPORT_KEY
from src.utils.redact import redact

BASE = "https://api.the-odds-api.com/v4"


@dataclass(frozen=True)
class Usage:
    remaining: int | None
    used: int | None
    last: int | None


class OddsApiError(Exception):
    def __init__(self, status: int, message: str):
        self.status = status
        super().__init__(redact(message))


def _int(v: str | None) -> int | None:
    try:
        return int(float(v)) if v is not None else None
    except ValueError:
        return None


class EdgeOddsClient:
    def __init__(self, api_key: str | None = None, transport: httpx.AsyncBaseTransport | None = None):
        self.api_key = api_key if api_key is not None else settings.odds_api_key
        self.transport = transport

    async def _get(self, path: str, params: dict) -> tuple[object, Usage]:
        params = {"apiKey": self.api_key, **params}
        try:
            async with httpx.AsyncClient(transport=self.transport, timeout=30.0) as c:
                r = await c.get(f"{BASE}{path}", params=params)
        except httpx.HTTPError as e:
            raise OddsApiError(0, f"network error: {e}") from None
        if r.status_code != 200:
            raise OddsApiError(r.status_code, f"{r.status_code} on {path}: {r.text[:200]}")
        h = r.headers
        return r.json(), Usage(_int(h.get("x-requests-remaining")), _int(h.get("x-requests-used")),
                               _int(h.get("x-requests-last")))

    async def get_events(self):
        return await self._get(f"/sports/{SPORT_KEY}/events", {})

    async def get_game_odds(self):
        return await self._get(f"/sports/{SPORT_KEY}/odds", {
            "bookmakers": ",".join(BOOKS), "markets": ",".join(LINE_MARKETS),
            "oddsFormat": "american", "includeLinks": "true"})

    async def get_event_odds(self, event_id: str):
        return await self._get(f"/sports/{SPORT_KEY}/events/{event_id}/odds", {
            "bookmakers": ",".join(BOOKS), "markets": ",".join(PROP_MARKETS),
            "oddsFormat": "american", "includeLinks": "true"})

    async def get_scores(self, days_from: int = 1):
        return await self._get(f"/sports/{SPORT_KEY}/scores", {"daysFrom": days_from})
