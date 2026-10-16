# backend/src/edge/store.py
"""psycopg2 reads/writes for Edge. Thin on purpose: logic lives in pure modules."""
from __future__ import annotations

from datetime import datetime, timedelta

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values

from src.config import get_sync_database_url
from src.edge.quotes import Quote

_QCOLS = "event_id, market, player, side, line, book, price, link, observed_at"


def connect():
    return psycopg2.connect(get_sync_database_url())


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


def upcoming_events(conn, now: datetime, hours: int = 36) -> list[dict]:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""SELECT * FROM edge_events WHERE commence_time > %s AND commence_time <= %s
                       AND status <> 'unavailable' ORDER BY commence_time""", (now, now + timedelta(hours=hours)))
        return [dict(r) for r in cur.fetchall()]


def write_quotes(conn, quotes: list[Quote]) -> int:
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


def drop_missing_events(conn, live_ids: list[str], now: datetime) -> int:
    """Future events no longer in the feed are marked unavailable."""
    with conn.cursor() as cur:
        cur.execute("""UPDATE edge_events SET status = 'unavailable'
                       WHERE commence_time > %s AND status = 'scheduled' AND NOT (event_id = ANY(%s))""",
                    (now, live_ids))
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
