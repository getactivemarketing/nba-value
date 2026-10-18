"""Buys historical snapshots once and stores them raw (spec §5, §6, §9).

Never touches live worker state: no store.log_api_call, no credits_* keys, so
the worker's credit guard projects live spend only. The budget is the sum of
edge_bt_snapshots.credits.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial

from psycopg2.extras import Json, execute_values

from src.edge.backtest import plan
from src.edge.backtest.nflverse import Game
from src.edge.client import OddsApiError
from src.edge.config import NFL, TD_MARKET
from src.edge.ingest import _ts

log = logging.getLogger("edge.backtest.fetch")
COLS = ("season", "week", "game_id", "event_id", "kind", "label", "kickoff", "requested_at", "snapshot_at",
        "home", "away", "credits", "payload")


class Stop(Exception):
    pass


@dataclass
class Summary:
    calls: int = 0
    credits: int = 0
    errors: int = 0
    stopped: str | None = None
    unmatched: list[str] = field(default_factory=list)
    remaining: int | None = None


class Repo:
    def __init__(self, conn):
        self.conn = conn

    def done_keys(self) -> set[tuple[str, str, str]]:
        with self.conn.cursor() as cur:
            cur.execute("SELECT event_id, kind, label FROM edge_bt_snapshots")
            return set(cur.fetchall())

    def spent(self) -> int:
        with self.conn.cursor() as cur:
            cur.execute("SELECT COALESCE(SUM(credits), 0) FROM edge_bt_snapshots")
            return int(cur.fetchone()[0])

    def events_payload(self, day_key: str) -> dict | None:
        with self.conn.cursor() as cur:
            cur.execute("SELECT payload FROM edge_bt_snapshots WHERE event_id = %s AND kind = 'events'", (day_key,))
            row = cur.fetchone()
        return row[0] if row else None

    def insert(self, rows: list[dict]) -> None:
        try:
            with self.conn.cursor() as cur:
                execute_values(cur, f"INSERT INTO edge_bt_snapshots ({', '.join(COLS)}) VALUES %s "
                                    "ON CONFLICT (event_id, kind, label) DO NOTHING",
                               [tuple(Json(r[c]) if c == "payload" else r[c] for c in COLS) for r in rows])
        except Exception:
            self.conn.rollback()
            raise
        self.conn.commit()


def ensure_tables(url: str) -> None:
    from sqlalchemy import create_engine

    from src.database import Base
    from src.models.edge import EdgeBtPick, EdgeBtSnapshot
    engine = create_engine(url)
    Base.metadata.create_all(engine, tables=[EdgeBtSnapshot.__table__, EdgeBtPick.__table__])
    engine.dispose()


def _row(job: plan.Job, game: Game | None, event_id: str, payload: dict, snapshot_at: datetime | None,
         credits: int) -> dict:
    first = job.games[0]
    return {"season": first.season, "week": first.week, "game_id": game.game_id if game else None,
            "event_id": event_id, "kind": job.kind, "label": job.label,
            "kickoff": game.kickoff if game else job.requested_at, "requested_at": job.requested_at,
            "snapshot_at": snapshot_at, "home": game.home if game else "", "away": game.away if game else "",
            "credits": credits, "payload": payload}


def _stub(event_id: str, status: int | None = None) -> dict:
    return {"id": event_id, "bookmakers": [], "missing": status or True}


async def run_fetch(repo, client, games: list[Game], *, cap: int, min_remaining: int,
                    sleep=asyncio.sleep, min_interval: float = 0.5,
                    kinds: tuple[str, ...] = ("props", "lines")) -> Summary:
    s = Summary()
    done = repo.done_keys()
    spent = repo.spent()

    async def call(job: plan.Job, request):
        nonlocal spent
        if spent + job.cost > cap:
            raise Stop(f"credit cap {cap:,} reached ({spent:,} spent)")
        if s.remaining is not None and s.remaining < min_remaining:
            raise Stop(f"credits remaining {s.remaining:,} below floor {min_remaining:,}")
        await sleep(min_interval)
        data, usage = await request()
        cost = usage.last if usage.last is not None else job.cost
        spent += cost
        s.calls += 1
        s.credits += cost
        if usage.remaining is not None:
            s.remaining = usage.remaining
        if s.calls % 100 == 0:
            log.info("backtest fetch: %d calls, %d credits this run, %s remaining", s.calls, s.credits, s.remaining)
        return data, cost

    try:
        events: list[dict] = []
        day_lists: list[tuple[plan.Job, list[dict]]] = []
        for job in plan.day_jobs(games):
            day_key = job.event_ids[0]
            payload = repo.events_payload(day_key) if job.keys[0] in done else None
            if payload is None:
                data, cost = await call(job, partial(client.get_historical_events, NFL, plan.iso(job.requested_at)))
                payload = data
                repo.insert([_row(job, None, day_key, data, _ts(data["timestamp"]), cost)])
            day_lists.append((job, payload.get("data") or []))
            events += day_lists[-1][1]

        # A game's own ET-day list carries its latest kickoff; an earlier day's list may predate a move.
        pairs, leftover = [], []
        for job, day_events in day_lists:
            got, missed = plan.match_events(list(job.games), day_events)
            pairs += got
            leftover += [gm for gm in job.games if gm.game_id in set(missed)]
        got, s.unmatched = plan.match_events(leftover, events)
        pairs += got
        for gid in s.unmatched:
            log.warning("backtest: no Odds API event matched %s", gid)

        for job in plan.snapshot_jobs(pairs, kinds):
            if all(k in done for k in job.keys):
                continue
            date = plan.iso(job.requested_at)
            try:
                if job.kind in ("props", "td"):
                    markets = (TD_MARKET,) if job.kind == "td" else plan.BT_PROP_MARKETS
                    request = partial(client.get_historical_event_odds, job.event_ids[0], NFL, date,
                                      markets=markets)
                else:
                    request = partial(client.get_historical_game_odds, NFL, date)
                data, cost = await call(job, request)
            except OddsApiError as e:
                if e.status in (401, 429):
                    s.stopped = f"odds_api {e.status}"
                    return s
                if e.status == 404:
                    repo.insert([_row(job, gm, eid, _stub(eid, 404), None, 0)
                                 for gm, eid in zip(job.games, job.event_ids)])
                    continue
                s.errors += 1
                log.warning("backtest %s %s %s skipped: %s", job.kind, job.label, job.event_ids, e)
                continue
            try:
                at = _ts(data["timestamp"])
                if job.kind in ("props", "td"):
                    if not isinstance(data["data"], dict):
                        raise TypeError(f"{job.kind} data is {type(data['data']).__name__}")
                    rows = [_row(job, job.games[0], job.event_ids[0], data["data"], at, cost)]
                else:
                    by_id = {e["id"]: e for e in data.get("data") or []}
                    rows = [_row(job, gm, eid, by_id.get(eid) or _stub(eid), at, cost if i == 0 else 0)
                            for i, (gm, eid) in enumerate(zip(job.games, job.event_ids))]
            except (KeyError, TypeError, ValueError, AttributeError) as e:
                # Paid for but unusable: keep the spend on record and never buy it again.
                s.errors += 1
                log.warning("backtest %s %s %s malformed response stored as stub: %s",
                            job.kind, job.label, job.event_ids, e)
                rows = [_row(job, gm, eid, {**_stub(eid), "malformed": str(e)[:200]}, None, cost if i == 0 else 0)
                        for i, (gm, eid) in enumerate(zip(job.games, job.event_ids))]
            repo.insert(rows)
    except Stop as e:
        s.stopped = str(e)
    return s
