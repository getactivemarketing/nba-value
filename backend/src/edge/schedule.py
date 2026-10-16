# backend/src/edge/schedule.py
"""When to poll, when a quote is stale, and how hard to back off on credits.

Pure functions. Spec §5 (cadence table, credit guard, staleness).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable

from src.edge.quotes import Quote

FACTORS = (1, 2, 4, 8)
LOW_CREDITS = 50_000


def base_intervals(minutes_to_tip: float) -> tuple[int, int] | None:
    if minutes_to_tip <= 0 or minutes_to_tip > 36 * 60:
        return None
    if minutes_to_tip < 60:
        return (2, 2)
    if minutes_to_tip <= 6 * 60:
        return (3, 5)
    return (10, 15)


def interval_for(market: str, minutes_to_tip: float, factor: int) -> timedelta | None:
    base = base_intervals(minutes_to_tip)
    if base is None:
        return None
    minutes = base[1] if market.startswith("player_") else base[0]
    return timedelta(minutes=minutes * factor)


def is_due(last_poll: datetime | None, interval: timedelta, now: datetime) -> bool:
    return last_poll is None or now - last_poll >= interval


def is_stale(observed_at: datetime, interval: timedelta | None, now: datetime) -> bool:
    return interval is None or now - observed_at > 2 * interval


def _minutes_to(commence: datetime, now: datetime) -> float:
    return (commence - now).total_seconds() / 60


def fresh(quotes: Iterable[Quote], commence_by_event: dict[str, datetime], now: datetime,
          factor: int) -> list[Quote]:
    out = []
    for q in quotes:
        commence = commence_by_event.get(q.event_id)
        if commence is None:
            continue
        if not is_stale(q.observed_at, interval_for(q.market, _minutes_to(commence, now), factor), now):
            out.append(q)
    return out


def guard_factor(cost_last_24h: int, current_factor: int, cap: int, remaining: int | None) -> int:
    """Smallest factor whose projected monthly use fits the cap (max 8).

    cost_last_24h was spent at `current_factor`, so the unthrottled daily cost
    is roughly cost_last_24h * current_factor.
    """
    if remaining is not None and remaining < LOW_CREDITS:
        return FACTORS[-1]
    base_monthly = cost_last_24h * current_factor * 30
    for f in FACTORS:
        if base_monthly / f <= cap:
            return f
    return FACTORS[-1]


@dataclass(frozen=True)
class PollPlan:
    lines: bool
    prop_events: list[str]


def plan_polls(events: list[dict], now: datetime, factor: int) -> PollPlan:
    lines, props = False, []
    for e in events:
        mins = _minutes_to(e["commence_time"], now)
        li = interval_for("h2h", mins, factor)
        pi = interval_for("player_points", mins, factor)
        if li is not None and is_due(e.get("last_lines_poll"), li, now):
            lines = True
        if pi is not None and is_due(e.get("last_props_poll"), pi, now):
            props.append(e["event_id"])
    return PollPlan(lines, props)
