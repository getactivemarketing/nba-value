# backend/src/edge/ingest.py
"""Odds API JSON -> Quote rows. Malformed outcomes are skipped, never guessed."""
from __future__ import annotations

from datetime import datetime

from src.edge.quotes import Quote


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def event_row(event: dict) -> dict:
    return {"event_id": event["id"], "sport": event["sport_key"], "home_team": event["home_team"],
            "away_team": event["away_team"], "commence_time": _ts(event["commence_time"])}


def _outcome_quote(event: dict, book: str, market: str, o: dict, observed_at: datetime) -> Quote | None:
    price = o.get("price")
    if not isinstance(price, (int, float)) or price == 0:
        return None
    price = int(price)
    link = o.get("link")
    eid, home, away = event["id"], event["home_team"], event["away_team"]

    if market == "h2h":
        side = "home" if o.get("name") == home else "away" if o.get("name") == away else None
        return Quote(eid, market, "", side, 0.0, book, price, link, observed_at) if side else None

    point = o.get("point")
    if not isinstance(point, (int, float)):
        return None

    if market == "spreads":
        if o.get("name") == home:
            return Quote(eid, market, "", "home", float(point), book, price, link, observed_at)
        if o.get("name") == away:
            return Quote(eid, market, "", "away", -float(point), book, price, link, observed_at)
        return None

    side = {"Over": "over", "Under": "under"}.get(o.get("name"))
    if side is None:
        return None
    if market == "totals":
        return Quote(eid, market, "", side, float(point), book, price, link, observed_at)
    player = (o.get("description") or "").strip()
    if not player:
        return None
    return Quote(eid, market, player, side, float(point), book, price, link, observed_at)


def parse_event(event: dict, observed_at: datetime) -> list[Quote]:
    out: list[Quote] = []
    for bm in event.get("bookmakers") or []:
        for m in bm.get("markets") or []:
            for o in m.get("outcomes") or []:
                q = _outcome_quote(event, bm["key"], m["key"], o, observed_at)
                if q is not None:
                    out.append(q)
    return out
