# backend/src/edge/signups.py
"""Email list from the signup form on the Edge pages. Table: edge_email_signups."""
from __future__ import annotations

import re

# Per client: 5 signups, then one more every 10 minutes. The router-wide limit is
# sized for page reads (60/min), far too loose for a form that writes rows.
SIGNUP_BURST = 5
SIGNUP_RATE_PER_S = 1 / 600

SOURCES = frozenset({"edge_home", "edge_board", "edge_results", "edge_odds", "edge_player", "nfl_edge"})

_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")


def normalize_email(raw: str | None) -> str | None:
    """Trimmed, lowercased address, or None when it doesn't look like one."""
    email = (raw or "").strip().lower()
    if len(email) > 254 or not _EMAIL.match(email):
        return None
    return email


def normalize_source(raw: str | None) -> str:
    return raw if raw in SOURCES else "other"


def add(conn, email: str, source: str) -> bool:
    """Store an address once. True when it is new."""
    with conn.cursor() as cur:
        cur.execute("""INSERT INTO edge_email_signups (email, source) VALUES (%s, %s)
                       ON CONFLICT (email) DO NOTHING""", (email, source))
        new = cur.rowcount == 1
    conn.commit()
    return new


def recent(conn, limit: int) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute("""SELECT email, source, created_at FROM edge_email_signups
                       ORDER BY created_at DESC LIMIT %s""", (limit,))
        return [{"email": e, "source": s, "created_at": c.isoformat()} for e, s, c in cur.fetchall()]


def count(conn) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM edge_email_signups")
        return int(cur.fetchone()[0])
