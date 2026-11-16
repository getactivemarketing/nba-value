# backend/src/edge/social/__main__.py
"""Social CLI.

    python -m src.edge.social preview --out DIR [--since YYYY-MM-DD]
    python -m src.edge.social reset-dry-run --sport {nba,nfl}

preview writes every dry-run ledger row created since the date (ET, default today) as
NNNNN_<kind>_<key>.txt (the post text, thread posts separated by ---) and .png (the card).

reset-dry-run clears a sport's dry run before go-live, so live blind numbering starts after the
last real post (#1 when there is none): it deletes the sport's blind plays that never went out
live (no posted/unknown/pending ledger row) with their blind and blind-receipt ledger rows, and
every dry_run ledger row of the sport, in one transaction. With SOCIAL_DRY_RUN off it refuses when
a live post (posted/unknown/pending) is newer than anything it would delete.
"""
from __future__ import annotations

import argparse
import re
from datetime import date, datetime
from pathlib import Path

from psycopg2.extras import RealDictCursor

from src.config import settings
from src.edge.players import ET, et_date

SPORTS = ("nba", "nfl")
LIVE_STATUSES = ["posted", "unknown", "pending"]  # a post that went (or may have gone) out


def _dry_runs(conn, since: date) -> list[dict]:
    start = datetime(since.year, since.month, since.day, tzinfo=ET)
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""SELECT id, kind, key, text, image FROM edge_social_posts
                       WHERE status = 'dry_run' AND created_at >= %s ORDER BY id""", (start,))
        rows = [dict(r) for r in cur.fetchall()]
    conn.rollback()
    return rows


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-")


def export(conn, out: Path, since: date) -> int:
    out.mkdir(parents=True, exist_ok=True)
    rows = _dry_runs(conn, since)
    for r in rows:
        stem = f"{r['id']:05d}_{_slug(r['kind'])}_{_slug(r['key'])}"
        (out / f"{stem}.txt").write_text(r["text"] or "", encoding="utf-8")
        if r["image"] is not None:
            (out / f"{stem}.png").write_bytes(bytes(r["image"]))
    return len(rows)


class ResetRefused(RuntimeError):
    """reset-dry-run would delete rows older than a live post."""


def reset_dry_run(conn, sport: str) -> dict[str, int]:
    """Delete `sport`'s dry-run blind plays and dry-run ledger rows. Returns {"plays", "ledger"} counts.
    Raises ResetRefused (nothing deleted) in live mode when a live post is newer than the oldest row
    it would delete."""
    if sport not in SPORTS:
        raise ValueError(f"unknown sport {sport!r}")
    blind_key = "'blind:' || b.sport || ':' || b.n"
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (f"edge_blind_plays:{sport}",))
            cur.execute(f"""SELECT b.n, b.created_at FROM edge_blind_plays b
                            WHERE b.sport = %s AND NOT EXISTS (
                              SELECT 1 FROM edge_social_posts sp WHERE sp.kind = 'blind' AND sp.key = {blind_key}
                                AND sp.status = ANY(%s))""", (sport, LIVE_STATUSES))
            doomed = cur.fetchall()
            ns = [n for n, _ in doomed]
            blind_keys = [f"blind:{sport}:{n}" for n in ns] + [f"receipt:blind:{sport}:{n}" for n in ns]
            cur.execute("""SELECT id, created_at FROM edge_social_posts
                           WHERE (sport = %s AND status = 'dry_run')
                              OR (key = ANY(%s) AND status <> ALL(%s))""", (sport, blind_keys, LIVE_STATUSES))
            rows = cur.fetchall()
            stamps = [t for _, t in doomed] + [t for _, t in rows]
            if stamps and not settings.social_dry_run:
                cur.execute("""SELECT count(*) FROM edge_social_posts
                               WHERE sport = %s AND status = ANY(%s) AND created_at >= %s""",
                            (sport, LIVE_STATUSES, min(stamps)))
                if cur.fetchone()[0]:
                    raise ResetRefused(f"SOCIAL_DRY_RUN is off and {sport} has a live post newer than the "
                                       "dry-run rows; nothing deleted")
            cur.execute("DELETE FROM edge_social_posts WHERE id = ANY(%s)", ([i for i, _ in rows],))
            n_ledger = cur.rowcount
            cur.execute("DELETE FROM edge_blind_plays WHERE sport = %s AND n = ANY(%s)", (sport, ns))
            n_plays = cur.rowcount
    except Exception:
        conn.rollback()
        raise
    conn.commit()
    return {"plays": n_plays, "ledger": n_ledger}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m src.edge.social")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pv = sub.add_parser("preview", help="export dry-run posts as PNG + TXT")
    pv.add_argument("--out", required=True, type=Path)
    pv.add_argument("--since", type=date.fromisoformat, default=None)
    rs = sub.add_parser("reset-dry-run", help="delete a sport's dry-run blind plays and ledger rows")
    rs.add_argument("--sport", required=True, choices=SPORTS)
    args = ap.parse_args(argv)
    from src.edge import store
    conn = store.connect()
    try:
        if args.cmd == "reset-dry-run":
            try:
                out = reset_dry_run(conn, args.sport)
            except ResetRefused as e:
                raise SystemExit(f"refused: {e}") from None
            print(f"deleted {out['plays']} blind play(s) and {out['ledger']} ledger row(s) for {args.sport}")
            return
        since = args.since or et_date(datetime.now(ET))
        n = export(conn, args.out, since)
    finally:
        conn.close()
    print(f"wrote {n} dry-run post(s) to {args.out}")


if __name__ == "__main__":
    main()
