# backend/src/edge/worker.py
"""Edge ingest / pricing / grading loop. Runs as its own Railway service:

    python -m src.edge.worker

It never stops silently: every failure lands in edge_worker_state.last_error
(redacted) and the heartbeat keeps ticking so /edge/status and the frontend
banner can tell "delayed" from "dead".
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import date, datetime, timedelta, timezone

from src.config import settings
from src.edge import grading, opportunities, players, schedule, store
from src.edge.client import EdgeOddsClient, OddsApiError, Usage
from src.edge.config import LINE_MARKETS, PROP_MARKETS
from src.edge.engine import price_quotes
from src.edge.ingest import event_row, parse_event
from src.utils.redact import redact
from src.utils.season import current_nba_season

log = logging.getLogger("edge.worker")
TICK_SECONDS = 30


class Backoff:
    def __init__(self) -> None:
        self.seconds = 0

    def failure(self, status: int, now: datetime) -> datetime:
        if status == 401:
            return now + timedelta(minutes=15)
        self.seconds = 30 if self.seconds == 0 else min(self.seconds * 2, 600)
        return now + timedelta(seconds=self.seconds)

    def success(self) -> None:
        self.seconds = 0


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _parse(v: str | None) -> datetime | None:
    return datetime.fromisoformat(v) if v else None


def _record_usage(conn, endpoint: str, usage: Usage, now: datetime) -> None:
    store.log_api_call(conn, endpoint, usage.last or 0, usage.remaining, now)
    if usage.remaining is not None:
        store.set_state(conn, "credits_remaining", str(usage.remaining), now)
    if usage.used is not None:
        store.set_state(conn, "credits_used", str(usage.used), now)


def _reprice(conn, event_ids: list[str], events: list[dict], now: datetime, factor: int) -> int:
    if not event_ids:
        return 0
    commence = {e["event_id"]: e["commence_time"] for e in events}
    quotes = schedule.fresh(store.load_quotes(conn, event_ids), commence, now, factor)
    priced = price_quotes(quotes, settings.edge_min_ev, settings.edge_verify_ev)
    return opportunities.record(conn, priced, now)


async def tick(conn, client: EdgeOddsClient, bdl, now: datetime) -> dict:
    state = store.get_state(conn)
    out: dict = {}

    # Events: free call, hourly.
    last_ev = _parse(state.get("last_events_refresh"))
    if last_ev is None or now - last_ev >= timedelta(minutes=60):
        events_json, _ = await client.get_events()
        store.upsert_events(conn, [event_row(e) for e in events_json])
        store.drop_missing_events(conn, [e["id"] for e in events_json], now)
        store.set_state(conn, "last_events_refresh", _iso(now), now)

    factor = schedule.guard_factor(store.cost_last_24h(conn, now), int(state.get("guard_factor", "1")),
                                   settings.edge_credit_cap, int(state["credits_remaining"])
                                   if state.get("credits_remaining") else None)
    if factor != int(state.get("guard_factor", "1")):
        log.warning("edge credit guard factor -> %s", factor)
    store.set_state(conn, "guard_factor", str(factor), now)

    events = store.upcoming_events(conn, now)
    plan = schedule.plan_polls(events, now, factor)
    touched: list[str] = []

    if plan.lines:
        data, usage = await client.get_game_odds()
        _record_usage(conn, "odds", usage, now)
        upcoming = {e["event_id"] for e in events}
        quotes = [q for ev in data if ev["id"] in upcoming for q in parse_event(ev, now)]
        out["line_history"] = store.write_quotes(conn, quotes)
        store.retire_missing(conn, sorted(upcoming), list(LINE_MARKETS), now)
        store.mark_polled(conn, sorted(upcoming), "last_lines_poll", now)
        touched += sorted(upcoming)

    by_id = {e["event_id"]: e for e in events}
    for eid in plan.prop_events:
        try:
            data, usage = await client.get_event_odds(eid)
        except OddsApiError as e:
            if e.status in (401, 429):
                raise  # account-wide: global backoff
            if e.status in (404, 422):
                log.warning("edge event %s gone upstream (%s); dropping", eid, e.status)
                store.mark_unavailable(conn, [eid])
            else:
                log.error("edge props poll failed for %s: %s", eid, e)
                store.set_state(conn, "last_event_error", f"{eid}: {e}"[:300], now)
            continue  # other events still poll; reprice and grading still run
        _record_usage(conn, "event_odds", usage, now)
        quotes = parse_event(data, now)
        store.write_quotes(conn, quotes)
        store.retire_missing(conn, [eid], list(PROP_MARKETS), now)
        store.mark_polled(conn, [eid], "last_props_poll", now)
        await players.resolve_new_players(conn, bdl, quotes, by_id[eid], now)
        touched.append(eid)

    out["opportunities"] = _reprice(conn, sorted(set(touched)), events, now, factor)

    # Scores while games are live; grading every 10 minutes.
    last_scores = _parse(state.get("last_scores_poll"))
    if grading.games_in_progress(conn, now) and (last_scores is None or now - last_scores >= timedelta(minutes=30)):
        data, usage = await client.get_scores(days_from=1)
        _record_usage(conn, "scores", usage, now)
        grading.update_scores(conn, data)
        store.set_state(conn, "last_scores_poll", _iso(now), now)

    last_grade = _parse(state.get("last_grading"))
    if last_grade is None or now - last_grade >= timedelta(minutes=10):
        out["graded"] = grading.run_grading(conn, now)
        store.set_state(conn, "last_grading", _iso(now), now)

    # Box scores once a day after 09:00 UTC, for yesterday (ET). A BallDontLie
    # failure is recorded under its own key (the loop clears last_error on every
    # good tick) and the day is marked done so we retry tomorrow, not every 30s.
    yday = players.et_date(now) - timedelta(days=1)
    if now.hour >= 9 and state.get("last_box_scores_date") != yday.isoformat():
        try:
            out["box_rows"] = await players.ingest_box_scores(conn, bdl, yday)
            store.set_state(conn, "bdl_error", "", now)
        except players.BdlError as e:
            store.set_state(conn, "bdl_error", f"balldontlie {e.status}: {e}"[:300], now)
        store.set_state(conn, "last_box_scores_date", yday.isoformat(), now)

    return out


async def ensure_backfill(conn, bdl, now: datetime) -> None:
    """One-time load of last season's box scores so hit rates work on night one.

    Runs inside the worker because only Railway can reach the private DB host.
    """
    prev = current_nba_season(now.date()) - 1
    key = f"backfill_{prev}"
    if store.get_state(conn).get(key) == "done":
        return
    try:
        total = await players.backfill(conn, bdl, date(prev, 10, 15), date(prev + 1, 6, 30))
        store.set_state(conn, key, "done", now)
        log.info("edge backfill %s: %s rows", prev, total)
    except players.BdlError as e:
        store.set_state(conn, "bdl_error", f"backfill: balldontlie {e.status}: {e}"[:300], now)


async def _run() -> None:
    client = EdgeOddsClient()
    bdl = players.BdlClient()
    backoff = Backoff()
    blocked_until: datetime | None = None
    conn = store.connect()
    await ensure_backfill(conn, bdl, datetime.now(timezone.utc))
    while True:
        now = datetime.now(timezone.utc)
        try:
            if conn.closed:
                conn = store.connect()
            store.set_state(conn, "heartbeat_at", _iso(now), now)
            if blocked_until is None or now >= blocked_until:
                result = await tick(conn, client, bdl, now)
                backoff.success()
                blocked_until = None
                store.set_state(conn, "last_error", "", now)
                log.info("edge tick %s", result)
        except OddsApiError as e:
            blocked_until = backoff.failure(e.status, now)
            log.error("edge odds api error %s: %s", e.status, e)
            store.set_state(conn, "last_error", f"odds_api {e.status}: {e}"[:500], now)
        except Exception as e:  # noqa: BLE001 - the loop must survive anything
            log.exception("edge tick failed")
            try:
                conn.rollback()
                store.set_state(conn, "last_error", redact(f"{type(e).__name__}: {e}")[:500], now)
            except Exception:  # noqa: BLE001
                conn = store.connect()
        await asyncio.sleep(TICK_SECONDS)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    while True:
        try:
            asyncio.run(_run())
        except Exception:  # noqa: BLE001
            log.exception("edge worker crashed; restarting in 30s")
            time.sleep(30)


if __name__ == "__main__":
    main()
