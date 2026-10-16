# backend/src/edge/api.py
"""Read-only /api/v1/edge routes (spec §8). 20-30s in-process cache per query."""
from __future__ import annotations

import time
from contextlib import contextmanager
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.deps import require_admin
from src.edge import store, views
from src.utils.redact import redact

router = APIRouter(prefix="/edge", tags=["Edge"])
_cache: dict[tuple, tuple[float, object]] = {}
TTL = 25.0


@contextmanager
def _conn():
    conn = store.connect()
    try:
        yield conn
    finally:
        conn.close()


def access_tier() -> str:
    """Paywall seam: Stripe will decide this. v1 is a free beta."""
    return "full"


def _cached(key: tuple, build):
    hit = _cache.get(key)
    if hit and time.monotonic() - hit[0] < TTL:
        return hit[1]
    try:
        with _conn() as conn:
            value = build(conn, datetime.now(timezone.utc))
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"edge data unavailable: {redact(e)[:200]}") from None
    _cache[key] = (time.monotonic(), value)
    return value


@router.get("/games")
def get_games(tier: str = Depends(access_tier)) -> dict:
    return {"games": _cached(("games",), lambda c, now: views.games(c, now)), "access_tier": tier}


@router.get("/ev")
def get_ev(market: str | None = None, book: str | None = None,
           min_ev: float | None = Query(None, ge=0, le=1), event_id: str | None = None,
           tier: str = Depends(access_tier)) -> dict:
    rows = _cached(("ev", market, book, min_ev, event_id), lambda c, now: views.ev_rows(
        c, now, market=market, book=book, min_ev=min_ev, event_id=event_id))
    return {"rows": rows, "access_tier": tier}


@router.get("/odds/{event_id}")
def get_odds(event_id: str, market: str = "h2h") -> dict:
    return _cached(("odds", event_id, market), lambda c, now: views.odds(c, now, event_id, market))


@router.get("/players/{bdl_id}")
def get_player(bdl_id: int, market: str = "player_points", line: float | None = None,
               min_minutes: float = Query(0, ge=0, le=48)) -> dict:
    return _cached(("player", bdl_id, market, line, min_minutes),
                   lambda c, now: views.player(c, now, bdl_id, market, line, min_minutes))


@router.get("/results")
def get_results(days: int = Query(365, ge=1, le=730), market: str | None = None,
                book: str | None = None) -> dict:
    return _cached(("results", days, market, book), lambda c, now: views.results(c, now, days, market, book))


@router.get("/health")
def get_public_health() -> dict:
    return _cached(("health",), lambda c, now: views.public_status(c, now))


@router.get("/status", dependencies=[Depends(require_admin)])
def get_status() -> dict:
    with _conn() as conn:
        return views.status(conn, datetime.now(timezone.utc))
