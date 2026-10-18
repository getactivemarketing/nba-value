# backend/src/edge/opportunities.py
"""Log every +EV price the first time it is seen (spec §6.4).

A new price at the same book/line is a new opportunity; a repeat sighting only
bumps last_seen_at and max_ev. Logging at first sight — not at the best moment
in hindsight — is what keeps the published record honest.
"""
from __future__ import annotations

from datetime import datetime

from psycopg2.extras import Json, execute_values

from src.edge.config import DEVIG_METHOD
from src.edge import tagging
from src.edge.engine import Priced


def _stored_tags(p: Priced, tagger, teams: dict[str, tuple[str, str] | None]):
    """[{key, direction}] for an NFL player opportunity; None when there is nothing to record."""
    q = p.quote
    pair = teams.get(q.event_id)
    if tagger is None or pair is None or not q.player:
        return None
    tags = tagging.row_tags(tagger, q.player, q.market, q.side, *pair)
    return Json([{"key": t["key"], "direction": t["direction"]} for t in tags]) if tags else None


def opportunity_rows(priced: list[Priced], now: datetime, tagger=None,
                     events: list[dict] | None = None) -> list[tuple]:
    teams = {e["event_id"]: (e["home_team"], e["away_team"]) for e in events or []}
    rows = []
    for p in priced:
        if not p.is_ev:
            continue
        q = p.quote
        ev = round(p.ev, 5)
        rows.append((q.event_id, q.market, q.player, q.side, q.line, q.book, q.price, q.link,
                     now, round(p.fair.prob, 5), ev, p.fair.n_books, now, ev, DEVIG_METHOD, False,
                     _stored_tags(p, tagger, teams)))
    return rows


def record(conn, priced: list[Priced], now: datetime, tagger=None, events: list[dict] | None = None) -> int:
    rows = opportunity_rows(priced, now, tagger, events)
    if not rows:
        return 0
    with conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO edge_opportunities (event_id, market, player, side, line, book, price, link,
                first_seen_at, fair_prob, ev, n_books, last_seen_at, max_ev, devig_method, closing_done, tags)
            VALUES %s
            ON CONFLICT (event_id, market, player, side, line, book, price) DO UPDATE SET
                last_seen_at = EXCLUDED.last_seen_at,
                max_ev = GREATEST(edge_opportunities.max_ev, EXCLUDED.ev)
        """, rows)
    conn.commit()
    return len(rows)
