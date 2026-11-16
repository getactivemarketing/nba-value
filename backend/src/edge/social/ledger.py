# backend/src/edge/social/ledger.py
"""Post ledger, switches and brakes for automatic X posting. Table: edge_social_posts.

claim() is the once-per-key gate: a (kind, key) is posted at most once, a failed
attempt may be retried up to MAX_ATTEMPTS times, and brake_tripped() stops a
runaway loop. dry_run rows count toward neither brake.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from psycopg2.extras import Json

from src.config import settings

MAX_ATTEMPTS = 3
STALE_PENDING = timedelta(minutes=15)  # a worker that died mid-post
ET = ZoneInfo("America/New_York")
_TERMINAL = frozenset({"posted", "dry_run", "skipped", "failed"})


def _csv(raw: str) -> set[str]:
    return {p.strip() for p in raw.split(",") if p.strip()}


def enabled(kind: str, sport: str) -> bool:
    return kind in _csv(settings.social_kinds) and sport in _csv(settings.social_sports)


def claim(conn, sport: str, kind: str, key: str, now: datetime) -> int | None:
    """Insert a pending row, or re-claim a failed/stale-pending one with attempts left. None if not claimable."""
    with conn.cursor() as cur:
        cur.execute("""INSERT INTO edge_social_posts (sport, kind, key, status, attempts, created_at, updated_at)
                       VALUES (%s, %s, %s, 'pending', 1, %s, %s)
                       ON CONFLICT (kind, key) DO UPDATE
                         SET status = 'pending', attempts = edge_social_posts.attempts + 1, updated_at = EXCLUDED.updated_at
                       WHERE edge_social_posts.attempts < %s
                         AND (edge_social_posts.status = 'failed'
                              OR (edge_social_posts.status = 'pending' AND edge_social_posts.updated_at < %s))
                       RETURNING id""",
                    (sport, kind, key, now, now, MAX_ATTEMPTS, now - STALE_PENDING))
        row = cur.fetchone()
    conn.commit()
    return row[0] if row else None


def finish(conn, row_id: int, status: str, now: datetime, *, text: str, image: bytes | None = None,
           media_url: str | None = None, submission_id: str | None = None, post_url: str | None = None,
           detail: dict | None = None) -> None:
    if status not in _TERMINAL:
        raise ValueError(f"bad status {status!r}")
    with conn.cursor() as cur:
        cur.execute("""UPDATE edge_social_posts
                       SET status = %s, text = %s, image = %s, media_url = %s, submission_id = %s,
                           post_url = %s, detail = %s, updated_at = %s
                       WHERE id = %s""",
                    (status, text, image, media_url, submission_id, post_url,
                     Json(detail) if detail is not None else None, now, row_id))
    conn.commit()


def brake_tripped(conn, sport: str, now: datetime) -> bool:
    """True when posted+pending rows reach the per-ET-day or per-hour cap for this sport."""
    day_start = now.astimezone(ET).replace(hour=0, minute=0, second=0, microsecond=0)
    with conn.cursor() as cur:
        cur.execute("""SELECT count(*) FILTER (WHERE updated_at >= %s),
                              count(*) FILTER (WHERE updated_at >= %s)
                       FROM edge_social_posts
                       WHERE sport = %s AND status IN ('posted', 'pending') AND updated_at >= %s""",
                    (day_start, now - timedelta(hours=1), sport, min(day_start, now - timedelta(hours=1))))
        per_day, per_hour = cur.fetchone()
    return per_day >= settings.social_max_per_day or per_hour >= settings.social_max_per_hour


def find(conn, kind: str, key: str) -> dict | None:
    with conn.cursor() as cur:
        cur.execute("""SELECT id, sport, kind, key, status, attempts, text, image, media_url, submission_id,
                              post_url, detail, created_at, updated_at
                       FROM edge_social_posts WHERE kind = %s AND key = %s""", (kind, key))
        row = cur.fetchone()
        if row is None:
            return None
        return dict(zip([d[0] for d in cur.description], row))
