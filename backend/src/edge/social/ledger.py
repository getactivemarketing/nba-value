# backend/src/edge/social/ledger.py
"""Post ledger, switches and brakes for automatic X posting. Table: edge_social_posts.

claim() is the once-per-key gate: a (kind, key) is posted at most once, a failed
attempt may be retried up to MAX_ATTEMPTS times, and brake_tripped() stops a
runaway loop. dry_run rows count toward neither brake.

A live row left `pending` past STALE_PENDING (the worker died mid-post) is never
re-claimed: the post may have gone out, so it becomes `unknown` (logged at ERROR)
and only reconcile_unknown() can turn it into `posted`, from Blotato's own answer.
Dry runs post nothing, so a stale dry-run row may be re-claimed.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from psycopg2.extras import Json

from src.config import settings

MAX_ATTEMPTS = 3
STALE_PENDING = timedelta(minutes=15)  # a worker that died mid-post
log = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")
_FINISH_STATUSES = frozenset({"posted", "dry_run", "skipped", "failed", "unknown"})
COUNTED_AS_POSTED = ("posted", "pending", "unknown")  # what the brakes count
HOURLY_CAP_EXEMPT = ("final", "receipt")  # kinds only the daily cap counts


@dataclass(frozen=True)
class Claim:
    """Proof of a claim: finish() only applies while this exact attempt still owns the row."""
    id: int
    attempt: int


def _csv(raw: str) -> set[str]:
    return {p.strip() for p in raw.split(",") if p.strip()}


def enabled(kind: str, sport: str) -> bool:
    return kind in _csv(settings.social_kinds) and sport in _csv(settings.social_sports)


def claim(conn, sport: str, kind: str, key: str, now: datetime, *, dry_run: bool = False) -> Claim | None:
    """Insert a pending row, or re-claim a failed one with attempts left. None if not claimable.

    Live (dry_run False): a stale pending row becomes `unknown` and is never re-claimed (it may have
    posted). Dry run: a stale pending row with attempts left is re-claimed. A stale pending row
    already at MAX_ATTEMPTS stays pending (fail-closed: the key is dead).
    """
    with conn.cursor() as cur:
        if not dry_run:
            cur.execute("""UPDATE edge_social_posts SET status = 'unknown'
                           WHERE kind = %s AND key = %s AND status = 'pending' AND updated_at < %s
                           RETURNING id, submission_id""", (kind, key, now - STALE_PENDING))
            lost = cur.fetchone()
            if lost:
                log.error("social ledger: %s %s was left pending mid-post (row %s, submission %s); marked "
                          "unknown, never re-posted", kind, key, lost[0], lost[1])
        cur.execute("""INSERT INTO edge_social_posts (sport, kind, key, status, attempts, created_at, updated_at)
                       VALUES (%s, %s, %s, 'pending', 1, %s, %s)
                       ON CONFLICT (kind, key) DO UPDATE
                         SET status = 'pending', attempts = edge_social_posts.attempts + 1, updated_at = EXCLUDED.updated_at
                       WHERE edge_social_posts.attempts < %s
                         AND (edge_social_posts.status = 'failed'
                              OR (%s AND edge_social_posts.status = 'pending' AND edge_social_posts.updated_at < %s))
                       RETURNING id, attempts""",
                    (sport, kind, key, now, now, MAX_ATTEMPTS, dry_run, now - STALE_PENDING))
        row = cur.fetchone()
    conn.commit()
    return Claim(row[0], row[1]) if row else None


def note_submission(conn, claim: Claim, submission_id: str | None, media_url: str | None = None) -> None:
    """Store Blotato's submission id the moment the post is accepted, before finish(), so a row the
    worker loses after this point can still be reconciled."""
    if not submission_id:
        return
    with conn.cursor() as cur:
        cur.execute("""UPDATE edge_social_posts SET submission_id = %s, media_url = COALESCE(%s, media_url)
                       WHERE id = %s AND status = 'pending' AND attempts = %s""",
                    (submission_id, media_url, claim.id, claim.attempt))
    conn.commit()


def finish(conn, claim: Claim, status: str, now: datetime, *, text: str, image: bytes | None = None,
           media_url: str | None = None, submission_id: str | None = None, post_url: str | None = None,
           detail: dict | None = None) -> bool:
    """Record the outcome. False (nothing written) if the claim was superseded or already finished."""
    if status not in _FINISH_STATUSES:
        raise ValueError(f"bad status {status!r}")
    with conn.cursor() as cur:
        cur.execute("""UPDATE edge_social_posts
                       SET status = %s, text = %s, image = %s, media_url = %s, submission_id = %s,
                           post_url = %s, detail = %s, updated_at = %s
                       WHERE id = %s AND status = 'pending' AND attempts = %s""",
                    (status, text, image, media_url, submission_id, post_url,
                     Json(detail) if detail is not None else None, now, claim.id, claim.attempt))
        updated = cur.rowcount == 1
    conn.commit()
    if not updated:
        log.warning("social ledger: stale finish ignored for row %s attempt %s", claim.id, claim.attempt)
    return updated


def brake_tripped(conn, sport: str, now: datetime) -> bool:
    """True when posted+pending+unknown rows reach the per-ET-day or per-hour cap for this sport.
    Finals and receipts follow game ends (a busy night has many at once), so the hourly cap leaves
    them out; the daily cap counts every kind."""
    day_start = now.astimezone(ET).replace(hour=0, minute=0, second=0, microsecond=0)
    with conn.cursor() as cur:
        cur.execute("""SELECT count(*) FILTER (WHERE updated_at >= %s),
                              count(*) FILTER (WHERE updated_at >= %s AND kind <> ALL(%s))
                       FROM edge_social_posts
                       WHERE sport = %s AND status = ANY(%s) AND updated_at >= %s""",
                    (day_start, now - timedelta(hours=1), list(HOURLY_CAP_EXEMPT), sport,
                     list(COUNTED_AS_POSTED), min(day_start, now - timedelta(hours=1))))
        per_day, per_hour = cur.fetchone()
    conn.rollback()
    return per_day >= settings.social_max_per_day or per_hour >= settings.social_max_per_hour


def find(conn, kind: str, key: str) -> dict | None:
    with conn.cursor() as cur:
        cur.execute("""SELECT id, sport, kind, key, status, attempts, text, image, media_url, submission_id,
                              post_url, detail, created_at, updated_at
                       FROM edge_social_posts WHERE kind = %s AND key = %s""", (kind, key))
        row = cur.fetchone()
        cols = [d[0] for d in cur.description]
    conn.rollback()
    return dict(zip(cols, row)) if row is not None else None


URL_LOOKUP_WINDOW = timedelta(hours=48)


def unresolved(conn, now: datetime, limit: int = 5) -> list[tuple[int, str]]:
    """(id, submission_id) of recent posted rows whose public URL is not known yet and that
    Blotato has not reported as failed."""
    with conn.cursor() as cur:
        cur.execute("""SELECT id, submission_id FROM edge_social_posts
                       WHERE status = 'posted' AND post_url IS NULL AND submission_id IS NOT NULL
                         AND updated_at >= %s AND COALESCE(detail->>'blotato_state', '') <> 'failed'
                       ORDER BY updated_at LIMIT %s""", (now - URL_LOOKUP_WINDOW, limit))
        rows = cur.fetchall()
    conn.rollback()
    return [(r[0], r[1]) for r in rows]


def note_lookup(conn, row_id: int, state: str, post_url: str | None = None, error: str | None = None) -> None:
    """Store what Blotato said about a posted row: its public URL, or that publishing failed."""
    patch = {"blotato_state": state, **({"blotato_error": error} if error else {})}
    with conn.cursor() as cur:
        cur.execute("""UPDATE edge_social_posts SET post_url = COALESCE(%s, post_url),
                              detail = COALESCE(detail, '{}'::jsonb) || %s
                       WHERE id = %s AND status = 'posted'""", (post_url, Json(patch), row_id))
    conn.commit()


def unknown_submissions(conn, now: datetime, limit: int = 5) -> list[tuple[int, str]]:
    """(id, submission_id) of recent `unknown` rows that Blotato can be asked about."""
    with conn.cursor() as cur:
        cur.execute("""SELECT id, submission_id FROM edge_social_posts
                       WHERE status = 'unknown' AND submission_id IS NOT NULL AND updated_at >= %s
                       ORDER BY updated_at LIMIT %s""", (now - URL_LOOKUP_WINDOW, limit))
        rows = cur.fetchall()
    conn.rollback()
    return [(r[0], r[1]) for r in rows]


def confirm_posted(conn, row_id: int, post_url: str | None) -> bool:
    """An `unknown` row Blotato confirms as published becomes `posted`. Nothing else changes it."""
    with conn.cursor() as cur:
        cur.execute("""UPDATE edge_social_posts SET status = 'posted', post_url = COALESCE(%s, post_url),
                              detail = COALESCE(detail, '{}'::jsonb) || '{"reconciled": true}'::jsonb
                       WHERE id = %s AND status = 'unknown'""", (post_url, row_id))
        ok = cur.rowcount == 1
    conn.commit()
    if ok:
        log.warning("social ledger: row %s confirmed published by Blotato; now posted", row_id)
    return ok


def unknown_without_submission(conn) -> int:
    """`unknown` rows Blotato can't be asked about: a live post lost before its submission id was stored."""
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM edge_social_posts WHERE status = 'unknown' AND submission_id IS NULL")
        n = cur.fetchone()[0]
    conn.rollback()
    return int(n)


IMAGE_KEEP = timedelta(days=7)  # receipts reuse a card for RECEIPT_DAYS; after a week only the text stays


def prune_images(conn, now: datetime) -> int:
    """Drop card PNGs from ledger rows created more than IMAGE_KEEP ago. Returns rows cleared."""
    with conn.cursor() as cur:
        cur.execute("UPDATE edge_social_posts SET image = NULL WHERE image IS NOT NULL AND created_at < %s",
                    (now - IMAGE_KEEP,))
        n = cur.rowcount
    conn.commit()
    return n
