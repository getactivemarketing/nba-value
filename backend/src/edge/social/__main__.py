# backend/src/edge/social/__main__.py
"""Social CLI.

    python -m src.edge.social preview --out DIR [--since YYYY-MM-DD]

Writes every dry-run ledger row created since the date (ET, default today) as
NNNNN_<kind>_<key>.txt (the post text, thread posts separated by ---) and .png (the card).
"""
from __future__ import annotations

import argparse
import re
from datetime import date, datetime
from pathlib import Path

from psycopg2.extras import RealDictCursor

from src.edge.players import ET, et_date


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


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m src.edge.social")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pv = sub.add_parser("preview", help="export dry-run posts as PNG + TXT")
    pv.add_argument("--out", required=True, type=Path)
    pv.add_argument("--since", type=date.fromisoformat, default=None)
    args = ap.parse_args(argv)
    from src.edge import store
    since = args.since or et_date(datetime.now(ET))
    conn = store.connect()
    try:
        n = export(conn, args.out, since)
    finally:
        conn.close()
    print(f"wrote {n} dry-run post(s) to {args.out}")


if __name__ == "__main__":
    main()
