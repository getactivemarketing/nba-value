# backend/src/edge/api.py
"""Read-only /api/v1/edge routes (spec §8). 20-30s in-process cache per query.

Hardened for public traffic: per-IP rate limit, bounded cache, pooled
connections (the Postgres is cross-region, so a new connection per request
costs a TLS handshake over ~200 ms of latency).
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from psycopg2.pool import ThreadedConnectionPool

from src.api.deps import require_admin
from src.config import get_sync_database_url
from src.edge import views
from src.utils.redact import redact

TTL = 25.0
MAX_CACHE_ENTRIES = 512
POOL_MAX = 8
RATE_PER_S = 1.0  # sustained: 60 requests/minute per client
RATE_BURST = 60   # an open Edge page makes ~6 requests/minute


class RateLimiter:
    """Token bucket per client key, holding at most max_clients buckets."""

    def __init__(self, rate_per_s: float, burst: int, max_clients: int = 10_000):
        self.rate, self.burst, self.max_clients = rate_per_s, burst, max_clients
        self._buckets: OrderedDict[str, tuple[float, float]] = OrderedDict()
        self._lock = threading.Lock()

    def allow(self, key: str, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        with self._lock:
            tokens, last = self._buckets.pop(key, (float(self.burst), now))
            tokens = min(float(self.burst), tokens + (now - last) * self.rate)
            ok = tokens >= 1.0
            self._buckets[key] = (tokens - 1.0 if ok else tokens, now)
            while len(self._buckets) > self.max_clients:
                self._buckets.popitem(last=False)  # least recently seen
            return ok

    def reset(self) -> None:
        with self._lock:
            self._buckets.clear()

    def __len__(self) -> int:
        return len(self._buckets)


_limiter = RateLimiter(RATE_PER_S, RATE_BURST)


def rate_limit(request: Request) -> None:
    # Railway's edge sets X-Real-IP to the caller's address.
    ip = request.headers.get("x-real-ip") or (request.client.host if request.client else "unknown")
    if not _limiter.allow(ip):
        raise HTTPException(status_code=429, detail="Too many requests", headers={"Retry-After": "5"})


router = APIRouter(prefix="/edge", tags=["Edge"], dependencies=[Depends(rate_limit)])
_cache: OrderedDict[tuple, tuple[float, object]] = OrderedDict()
_cache_lock = threading.Lock()
_pool: ThreadedConnectionPool | None = None
_pool_lock = threading.Lock()
_slots = threading.BoundedSemaphore(POOL_MAX)


def _get_pool() -> ThreadedConnectionPool:
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = ThreadedConnectionPool(1, POOL_MAX, get_sync_database_url())
        return _pool


@contextmanager
def _conn():
    # The semaphore makes requests wait for a free connection instead of
    # failing: ThreadedConnectionPool raises when all POOL_MAX are out.
    if not _slots.acquire(timeout=10):
        raise HTTPException(status_code=503, detail="edge data busy, retry shortly")
    try:
        pool = _get_pool()
        conn = pool.getconn()
        broken = False
        try:
            yield conn
        except BaseException:
            broken = True
            raise
        finally:
            if not broken and not conn.closed:
                try:
                    conn.rollback()  # reads leave a transaction open
                except Exception:  # noqa: BLE001
                    broken = True
            pool.putconn(conn, close=broken or bool(conn.closed))
    finally:
        _slots.release()


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
    with _cache_lock:
        _cache.pop(key, None)
        _cache[key] = (time.monotonic(), value)
        while len(_cache) > MAX_CACHE_ENTRIES:
            _cache.popitem(last=False)  # oldest entry first
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
