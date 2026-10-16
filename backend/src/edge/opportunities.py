# backend/src/edge/opportunities.py
"""Log every +EV price the first time it is seen (spec §6.4).

A new price at the same book/line is a new opportunity; a repeat sighting only
bumps last_seen_at and max_ev. Logging at first sight — not at the best moment
in hindsight — is what keeps the published record honest.
"""
from __future__ import annotations

from datetime import datetime

from psycopg2.extras import execute_values

from src.edge.config import DEVIG_METHOD
from src.edge.engine import Priced


def opportunity_rows(priced: list[Priced], now: datetime) -> list[tuple]:
    rows = []
    for p in priced:
        if not p.is_ev:
            continue
        q = p.quote
        ev = round(p.ev, 5)
        rows.append((q.event_id, q.market, q.player, q.side, q.line, q.book, q.price, q.link,
                     now, round(p.fair.prob, 5), ev, p.fair.n_books, now, ev, DEVIG_METHOD, False))
    return rows


def record(conn, priced: list[Priced], now: datetime) -> int:
    rows = opportunity_rows(priced, now)
    if not rows:
        return 0
    with conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO edge_opportunities (event_id, market, player, side, line, book, price, link,
                first_seen_at, fair_prob, ev, n_books, last_seen_at, max_ev, devig_method, closing_done)
            VALUES %s
            ON CONFLICT (event_id, market, player, side, line, book, price) DO UPDATE SET
                last_seen_at = EXCLUDED.last_seen_at,
                max_ev = GREATEST(edge_opportunities.max_ev, EXCLUDED.ev)
        """, rows)
    conn.commit()
    return len(rows)
