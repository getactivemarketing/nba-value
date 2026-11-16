# backend/src/edge/social/safety.py
"""Pre-post gates. Every post passes one; each fails closed (unknown or missing data means no post).

The owner's hard rule: never post a wrong score or a pick whose game has started.
"""
from __future__ import annotations

from datetime import datetime, timedelta

MIN_BEFORE_TIP = timedelta(minutes=60)
_BAD_STATUS = ("postponed", "cancelled", "canceled", "suspended")


def tip_of(r: dict) -> datetime:
    t = r["commence_time"]
    return datetime.fromisoformat(t) if isinstance(t, str) else t


def _aware(*dts: datetime) -> bool:
    return all(d.tzinfo is not None for d in dts)


def pick_ok(row: dict, now: datetime, current_ev: float | None) -> bool:
    """Blind play: at least 60 minutes to the tip and still +EV on the live board."""
    tip = tip_of(row)
    return (current_ev is not None and current_ev > 0 and _aware(tip, now)
            and tip - now >= MIN_BEFORE_TIP)


def value_ok(legs: list[dict], now: datetime) -> bool:
    """Top-5 post: no leg's game has started."""
    tips = [tip_of(l) for l in legs]
    return bool(legs) and _aware(now, *tips) and all(t > now for t in tips)


def final_ok(event: dict, second_source: tuple[int, int] | None) -> bool:
    """Recap or receipt: the game is final, not postponed/cancelled/suspended, and the odds
    feed's score equals an independent box-score total. No second source means no post."""
    if second_source is None or not event.get("completed"):
        return False
    if any(w in str(event.get("status") or "").lower() for w in _BAD_STATUS):
        return False
    home, away = event.get("home_score"), event.get("away_score")
    if home is None or away is None:
        return False
    try:
        return (int(home), int(away)) == (int(second_source[0]), int(second_source[1]))
    except (TypeError, ValueError):
        return False
