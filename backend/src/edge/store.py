# backend/src/edge/store.py
"""psycopg2 reads/writes for Edge. Thin on purpose: logic lives in pure modules."""
from __future__ import annotations

from datetime import datetime, timedelta

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values

from src.config import get_sync_database_url
from src.edge.ingest import best_per_key
from src.edge.quotes import Quote

_QCOLS = "event_id, market, player, side, line, book, price, link, observed_at"


# The database is cross-region: bound connect time, and keep idle sessions
# alive in both directions so dead ones are noticed in ~1 minute, not hours.
# Server-side keepalives matter for the worker lock: Postgres holds a dead
# session's advisory lock until it notices the client is gone.
CONNECT_KWARGS = {
    "connect_timeout": 5,
    "keepalives": 1, "keepalives_idle": 30, "keepalives_interval": 10, "keepalives_count": 3,
}
SERVER_KEEPALIVES = "-c tcp_keepalives_idle=30 -c tcp_keepalives_interval=10 -c tcp_keepalives_count=3"


def connect():
    return psycopg2.connect(get_sync_database_url(), options=SERVER_KEEPALIVES, **CONNECT_KWARGS)


# Arbitrary constant: one Edge worker per database. Two would double credit use.
WORKER_LOCK_ID = 7_311_020


def try_worker_lock(conn) -> bool:
    """Session-level advisory lock; released automatically if the holder disconnects."""
    with conn.cursor() as cur:
        cur.execute("SELECT pg_try_advisory_lock(%s)", (WORKER_LOCK_ID,))
        held = cur.fetchone()[0]
    conn.commit()
    return bool(held)


def take_over_stale_lock(conn, now: datetime, stale_after: timedelta) -> bool:
    """Evict the lock holder if its heartbeat is older than `stale_after`.

    Covers a holder whose session died without Postgres noticing (it keeps the
    lock until its keepalives give up), including this worker's own old session.
    A missing heartbeat is not stale: a new holder writes one right after locking.
    """
    beat = get_state(conn).get("heartbeat_at")
    if not beat or now - datetime.fromisoformat(beat) < stale_after:
        return False
    with conn.cursor() as cur:
        cur.execute("""SELECT pg_terminate_backend(pid) FROM pg_locks
                       WHERE locktype = 'advisory' AND objid = %s AND granted
                         AND pid <> pg_backend_pid()""", (WORKER_LOCK_ID,))
        evicted = any(r[0] for r in cur.fetchall())
    conn.commit()
    return evicted


def _quote(r) -> Quote:
    return Quote(r[0], r[1], r[2], r[3], float(r[4]), r[5], int(r[6]), r[7], r[8])


def _qtuple(q: Quote) -> tuple:
    return (q.event_id, q.market, q.player, q.side, q.line, q.book, q.price, q.link, q.observed_at)


def changed_quotes(existing: dict[tuple, int], quotes: list[Quote]) -> list[Quote]:
    return [q for q in quotes if existing.get(q.key) != q.price]


def upsert_events(conn, rows: list[dict]) -> int:
    if not rows:
        return 0
    with conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO edge_events (event_id, sport, home_team, away_team, commence_time, status, completed)
            VALUES %s
            ON CONFLICT (event_id) DO UPDATE SET commence_time = EXCLUDED.commence_time,
                home_team = EXCLUDED.home_team, away_team = EXCLUDED.away_team,
                status = CASE WHEN edge_events.status = 'unavailable' THEN 'scheduled'
                              ELSE edge_events.status END
        """, [(r["event_id"], r["sport"], r["home_team"], r["away_team"], r["commence_time"],
               "scheduled", False) for r in rows])
    conn.commit()
    return len(rows)


def upcoming_events(conn, now: datetime, hours: int = 36, sport: str | None = None) -> list[dict]:
    sql = """SELECT * FROM edge_events WHERE commence_time > %s AND commence_time <= %s
             AND status <> 'unavailable'"""
    params: list = [now, now + timedelta(hours=hours)]
    if sport:
        sql += " AND sport = %s"
        params.append(sport)
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql + " ORDER BY commence_time", params)
        return [dict(r) for r in cur.fetchall()]


def write_quotes(conn, quotes: list[Quote]) -> int:
    quotes = best_per_key(quotes)  # one upsert can't touch a key twice (exchange ladders)
    if not quotes:
        return 0
    event_ids = sorted({q.event_id for q in quotes})
    with conn.cursor() as cur:
        cur.execute("""SELECT event_id, market, player, line, side, book, price FROM edge_quotes
                       WHERE event_id = ANY(%s)""", (event_ids,))
        existing = {(r[0], r[1], r[2], float(r[3]), r[4], r[5]): r[6] for r in cur.fetchall()}
        changed = changed_quotes(existing, quotes)
        execute_values(cur, f"""
            INSERT INTO edge_quotes ({_QCOLS}) VALUES %s
            ON CONFLICT (event_id, market, player, side, line, book) DO UPDATE SET
                price = EXCLUDED.price, link = EXCLUDED.link, observed_at = EXCLUDED.observed_at
        """, [_qtuple(q) for q in quotes])
        if changed:
            execute_values(cur, f"INSERT INTO edge_quote_history ({_QCOLS}) VALUES %s",
                           [_qtuple(q) for q in changed])
    conn.commit()
    return len(changed)


def mark_unavailable(conn, event_ids: list[str]) -> None:
    """Postponed, re-IDed or removed upstream: stop polling it (upsert revives it)."""
    if not event_ids:
        return
    with conn.cursor() as cur:
        cur.execute("UPDATE edge_events SET status = 'unavailable' WHERE event_id = ANY(%s)", (event_ids,))
    conn.commit()


def drop_missing_events(conn, live_ids: list[str], now: datetime, sport: str | None = None) -> int:
    """Future events of `sport` no longer in its feed are marked unavailable.

    Scoped per sport: each sport's feed only lists its own events, so an
    unscoped update would drop every other sport's games.
    """
    sql = """UPDATE edge_events SET status = 'unavailable'
             WHERE commence_time > %s AND status = 'scheduled' AND NOT (event_id = ANY(%s))"""
    params: list = [now, live_ids]
    if sport:
        sql += " AND sport = %s"
        params.append(sport)
    with conn.cursor() as cur:
        cur.execute(sql, params)
        n = cur.rowcount
    conn.commit()
    return n


def retire_missing(conn, event_ids: list[str], markets: list[str], polled_at: datetime) -> int:
    """Delete quotes a poll covered but did not return: the book pulled or moved them.

    write_quotes stamps every returned key with observed_at = polled_at, so in
    the polled (event, market) scope anything older is gone. History keeps it.
    """
    if not event_ids or not markets:
        return 0
    with conn.cursor() as cur:
        cur.execute("""DELETE FROM edge_quotes WHERE event_id = ANY(%s) AND market = ANY(%s)
                       AND observed_at < %s""", (event_ids, markets, polled_at))
        n = cur.rowcount
    conn.commit()
    return n


def mark_polled(conn, event_ids: list[str], field: str, now: datetime) -> None:
    if field not in ("last_lines_poll", "last_props_poll") or not event_ids:
        return
    with conn.cursor() as cur:
        cur.execute(f"UPDATE edge_events SET {field} = %s WHERE event_id = ANY(%s)", (now, event_ids))
    conn.commit()


def load_quotes(conn, event_ids: list[str], market: str | None = None) -> list[Quote]:
    sql = f"SELECT {_QCOLS} FROM edge_quotes WHERE event_id = ANY(%s)"
    params: list = [event_ids]
    if market:
        sql += " AND market = %s"
        params.append(market)
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return [_quote(r) for r in cur.fetchall()]


def log_api_call(conn, endpoint: str, cost: int, remaining: int | None, now: datetime) -> None:
    with conn.cursor() as cur:
        cur.execute("INSERT INTO edge_api_calls (called_at, endpoint, cost, remaining) VALUES (%s,%s,%s,%s)",
                    (now, endpoint, cost, remaining))
    conn.commit()


def cost_last_24h(conn, now: datetime) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT COALESCE(SUM(cost), 0) FROM edge_api_calls WHERE called_at > %s",
                    (now - timedelta(hours=24),))
        return int(cur.fetchone()[0])


def set_state(conn, key: str, value: str, now: datetime) -> None:
    with conn.cursor() as cur:
        cur.execute("""INSERT INTO edge_worker_state (key, value, updated_at) VALUES (%s, %s, %s)
                       ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = EXCLUDED.updated_at""",
                    (key, value, now))
    conn.commit()


def get_state(conn) -> dict[str, str]:
    with conn.cursor() as cur:
        cur.execute("SELECT key, value FROM edge_worker_state")
        return {k: v for k, v in cur.fetchall()}


def prune_history(conn, before: datetime, after: datetime | None = None) -> int:
    """Thin out price history for games that tipped before `before`.

    Keeps, per quote key: the opening price, the closing price and the first
    change in each hour, which is enough for line-movement charts. Nothing
    reads the rest, and it is most of the table's growth. Idempotent. `after`
    bounds the scan to recent games so daily runs stay cheap as history grows.
    """
    with conn.cursor() as cur:
        cur.execute("""
            DELETE FROM edge_quote_history h USING (
                SELECT id FROM (
                    SELECT hh.id,
                           row_number() OVER (PARTITION BY hh.event_id, hh.market, hh.player, hh.side, hh.line, hh.book
                                              ORDER BY hh.observed_at) AS rn_open,
                           row_number() OVER (PARTITION BY hh.event_id, hh.market, hh.player, hh.side, hh.line, hh.book
                                              ORDER BY hh.observed_at DESC) AS rn_close,
                           row_number() OVER (PARTITION BY hh.event_id, hh.market, hh.player, hh.side, hh.line, hh.book,
                                                           date_trunc('hour', hh.observed_at)
                                              ORDER BY hh.observed_at) AS rn_hour
                    FROM edge_quote_history hh
                    JOIN edge_events e ON e.event_id = hh.event_id
                    WHERE e.commence_time < %s AND (%s::timestamptz IS NULL OR e.commence_time >= %s)
                ) ranked
                WHERE rn_open > 1 AND rn_close > 1 AND rn_hour > 1
            ) doomed
            WHERE h.id = doomed.id
        """, (before, after, after))
        deleted = cur.rowcount
    conn.commit()
    return deleted
