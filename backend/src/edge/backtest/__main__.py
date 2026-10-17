"""python -m src.edge.backtest fetch|replay|report (spec §10).

Run from backend/ with DATABASE_URL pointing at the target database and
ODDS_API_KEY set (fetch only). BT_CREDIT_CAP / BT_MIN_REMAINING bound spend.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.config import get_sync_database_url, settings
from src.edge import store
from src.edge.backtest import SEASONS, fetch, nflverse, plan, replay, report
from src.edge.client import EdgeOddsClient
from src.utils.redact import install_redaction

DEFAULT_OUT = Path(__file__).resolve().parents[3] / "docs" / "backtests" / "nfl-2023-2025.md"


def _games(seasons, week):
    games = nflverse.load_schedule(seasons)
    return [g for g in games if week is None or g.week == week]


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    p = argparse.ArgumentParser(prog="python -m src.edge.backtest")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--season", type=int, action="append")
    f.add_argument("--week", type=int)
    f.add_argument("--dry-run", action="store_true")
    r = sub.add_parser("replay")
    r.add_argument("--season", type=int, action="append")
    rp = sub.add_parser("report")
    rp.add_argument("--run")
    rp.add_argument("--out", default=str(DEFAULT_OUT))
    a = p.parse_args(argv)
    seasons = a.season if getattr(a, "season", None) else list(SEASONS)

    if a.cmd == "fetch":
        games = _games(seasons, a.week)
        if a.dry_run:
            for k, v in plan.dry_run(games).items():
                print(f"{k}: {v:,}")
            return 0
        fetch.ensure_tables(get_sync_database_url())
        conn = store.connect()
        try:
            s = asyncio.run(fetch.run_fetch(
                fetch.Repo(conn), EdgeOddsClient(), games,
                cap=int(os.environ.get("BT_CREDIT_CAP", "800000")),
                min_remaining=int(os.environ.get("BT_MIN_REMAINING", "3600000"))))
        finally:
            conn.close()
        print(f"calls: {s.calls:,}  credits: {s.credits:,}  errors: {s.errors}  remaining: {s.remaining}")
        print(f"unmatched games: {len(s.unmatched)} {s.unmatched[:10]}")
        if s.stopped:
            print(f"stopped: {s.stopped}")
            return 1
        return 0

    conn = store.connect()
    try:
        if a.cmd == "replay":
            games = {g.game_id: g for g in nflverse.load_schedule(seasons)}
            run_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            n = replay.run_replay(conn, run_id, games, nflverse.load_stats(seasons), nflverse.load_snaps(seasons),
                                  settings.edge_min_ev, settings.edge_verify_ev)
            print(f"run {run_id}: {n:,} picks")
        else:
            run_id = report.run_report(conn, a.run, a.out)
            print(f"report for run {run_id} written to {a.out}")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    install_redaction()  # process-wide; only for real runs, never inside pytest
    sys.exit(main())
