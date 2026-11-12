# backend/src/edge/nightly.py
"""Tonight's top 5: the night's best NBA props, plus a parlay at one book, locked an hour before
the first game and graded like every other pick.

Picks come from the same board rows (views.ev_rows). A pick qualifies when 5+ books set its fair
price, it isn't marked "verify", and recent form isn't cold. Results come from the matching
edge_opportunities row: a prop's result is the same at every book and price.
"""
from __future__ import annotations

import json
import logging
import math
from datetime import date, datetime, timedelta, timezone

from psycopg2.extras import RealDictCursor

from src.edge import views
from src.edge.config import NBA, SportConfig
from src.edge.engine import american_to_decimal, prob_to_american
from src.edge.grading import profit_units
from src.edge.players import ET, et_date

log = logging.getLogger(__name__)

LOCK_BEFORE_FIRST_TIP = timedelta(minutes=60)
MIN_BOOKS = 5
TOP_N = 5
MIN_PARLAY_LEGS = 3
LEG_FIELDS = ("event_id", "matchup", "commence_time", "market", "market_label", "player", "side", "line",
              "bet", "book", "price", "fair_prob", "ev", "n_books", "form")


# ---- pure selection -------------------------------------------------------------------------

def eligible(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r["player"] and r["n_books"] >= MIN_BOOKS and not r["verify"]
            and not r.get("form_flag")]


def _one_per_game(rows: list[dict], n: int) -> list[dict]:
    games, players, out = set(), set(), []
    for r in sorted(rows, key=lambda r: r["ev"], reverse=True):
        if r["event_id"] in games or r["player"] in players:
            continue
        out.append(r)
        games.add(r["event_id"])
        players.add(r["player"])
        if len(out) == n:
            break
    return out


def top_picks(rows: list[dict], n: int = TOP_N) -> list[dict]:
    """Best edge first; one pick per game and per player, so the picks don't ride on one game."""
    return _one_per_game(rows, n)


def _parlay_price(legs: list[dict]) -> dict:
    dec = math.prod(american_to_decimal(int(l["price"])) for l in legs)
    prob = math.prod(float(l["fair_prob"]) for l in legs)
    return {"decimal": round(dec, 4), "american": prob_to_american(1 / dec), "hit_prob": round(prob, 4),
            "ev": round(dec * prob - 1, 4)}


def best_parlay(rows: list[dict], n: int = TOP_N) -> dict | None:
    """A parlay has to be bet at one book. Take each book's best leg per game (up to n) and pick
    the book with the most legs, then the most value. Legs are from different games, so the
    chances multiply."""
    best = None
    for book in sorted({r["book"] for r in rows}):
        legs = _one_per_game([r for r in rows if r["book"] == book], n)
        if len(legs) < MIN_PARLAY_LEGS:
            continue
        cand = {"book": book, "legs": legs, **_parlay_price(legs)}
        if best is None or (len(legs), cand["ev"]) > (len(best["legs"]), best["ev"]):
            best = cand
    return best


def parlay_result(legs: list[dict]) -> tuple[str | None, float | None]:
    """Any loss loses; any ungraded leg leaves it open; pushes and voids drop out of the price."""
    results = [l["result"] for l in legs]
    if "loss" in results:
        return "loss", -1.0
    if any(r is None for r in results):
        return None, None
    winners = [l for l in legs if l["result"] == "win"]
    if not winners:
        return "push", 0.0
    return "win", round(math.prod(american_to_decimal(int(l["price"])) for l in winners) - 1, 4)


def lock_at(tips: list[datetime]) -> datetime | None:
    return (min(tips) - LOCK_BEFORE_FIRST_TIP).astimezone(timezone.utc) if tips else None


def slate_date(now: datetime) -> date:
    return et_date(now)


# ---- DB ---------------------------------------------------------------------------------------

def _day_bounds(day: date) -> tuple[datetime, datetime]:
    start = datetime(day.year, day.month, day.day, tzinfo=ET)
    return start.astimezone(timezone.utc), (start + timedelta(days=1)).astimezone(timezone.utc)


def _tips(conn, sport: SportConfig, day: date) -> list[datetime]:
    lo, hi = _day_bounds(day)
    with conn.cursor() as cur:
        cur.execute("""SELECT commence_time FROM edge_events WHERE sport = %s AND status <> 'unavailable'
                       AND commence_time >= %s AND commence_time < %s""", (sport.key, lo, hi))
        return [r[0] for r in cur.fetchall()]


def _locked(conn, sport: SportConfig, day: date) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM edge_nightly_legs WHERE sport = %s AND slate_date = %s LIMIT 1",
                    (sport.slug, day))
        return cur.fetchone() is not None


def maybe_lock(conn, now: datetime, sport: SportConfig = NBA) -> int:
    """Lock tonight's picks once, from an hour before the first game. If the worker was down at
    lock time it locks late from the games not yet started. Returns legs written."""
    day = slate_date(now)
    when = lock_at(_tips(conn, sport, day))
    if when is None or now < when or _locked(conn, sport, day):
        return 0
    lo, hi = _day_bounds(day)
    rows = [r for r in views.ev_rows(conn, now, sport=sport)
            if lo <= datetime.fromisoformat(r["commence_time"]) < hi]
    ok = eligible(rows)
    top, parlay = top_picks(ok), best_parlay(ok)
    if not top:
        return 0  # nothing qualifies yet; try again next tick
    legs = [("top", i, r) for i, r in enumerate(top)]
    legs += [("parlay", i, r) for i, r in enumerate(parlay["legs"])] if parlay else []
    with conn.cursor() as cur:
        for kind, rank, r in legs:
            cur.execute(
                """INSERT INTO edge_nightly_legs (sport, slate_date, kind, rank, locked_at, event_id, matchup,
                       commence_time, market, market_label, player, side, line, bet, book, price, fair_prob,
                       ev, n_books, form)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (sport, slate_date, kind, rank) DO NOTHING""",
                (sport.slug, day, kind, rank, now, r["event_id"], r["matchup"], r["commence_time"],
                 r["market"], r["market_label"], r["player"], r["side"], r["line"], r["bet"], r["book"],
                 r["price"], r["fair_prob"], r["ev"], r["n_books"],
                 json.dumps(r["form"]) if r.get("form") else None))
    conn.commit()
    log.info("nightly %s %s locked: %d top, parlay %s", sport.slug, day, len(top),
             parlay["book"] if parlay else None)
    return len(legs)


def _leg_out(r: dict) -> dict:
    result = r["result"]
    return {**{k: r[k] for k in LEG_FIELDS},
            "commence_time": r["commence_time"].astimezone(timezone.utc).isoformat(), "line": float(r["line"]),
            "fair_prob": float(r["fair_prob"]), "ev": float(r["ev"]), "result": result,
            "profit_units": round(profit_units(result, int(r["price"])), 4) if result else None}


def _card(day: date, legs: list[dict]) -> dict:
    top = [l for l in legs if l["kind"] == "top"]
    par = [l for l in legs if l["kind"] == "parlay"]
    card = {"slate_date": day.isoformat(), "locked_at": legs[0]["locked_at"].astimezone(timezone.utc).isoformat(),
            "top": [_leg_out(l) for l in top], "parlay": None}
    if par:
        out = [_leg_out(l) for l in par]
        result, profit = parlay_result(out)
        card["parlay"] = {"book": par[0]["book"], "legs": out, **_parlay_price(out), "result": result,
                          "profit_units": profit}
    return card


def _record(cards: list[dict]) -> dict:
    graded = [l for c in cards for l in c["top"] if l["result"] in ("win", "loss", "push", "void")]
    parlays = [c["parlay"] for c in cards if c["parlay"] and c["parlay"]["result"]]
    return {
        "top": {"n": len(graded), "wins": sum(l["result"] == "win" for l in graded),
                "losses": sum(l["result"] == "loss" for l in graded),
                "units": round(sum(l["profit_units"] for l in graded), 2)},
        "parlay": {"n": len(parlays), "wins": sum(p["result"] == "win" for p in parlays),
                   "units": round(sum(p["profit_units"] for p in parlays), 2)},
    }


def nightly(conn, now: datetime, sport: SportConfig = NBA, days: int = 30) -> dict:
    """Tonight's card (or when it locks), recent cards, and the running record."""
    today = slate_date(now)
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        # A prop's result is the same at every book and price, so any graded logged row will do.
        cur.execute("""SELECT n.*, res.result FROM edge_nightly_legs n
                       LEFT JOIN LATERAL (
                         SELECT o.result FROM edge_opportunities o
                         WHERE o.event_id = n.event_id AND o.market = n.market AND o.player = n.player
                           AND o.side = n.side AND o.line = n.line AND o.result IS NOT NULL LIMIT 1) res ON true
                       WHERE n.sport = %s AND n.slate_date > %s
                       ORDER BY n.slate_date DESC, n.kind, n.rank""", (sport.slug, today - timedelta(days=days)))
        rows = [dict(r) for r in cur.fetchall()]
    by_day: dict[date, list[dict]] = {}
    for r in rows:
        by_day.setdefault(r["slate_date"], []).append(r)
    cards = [_card(d, legs) for d, legs in by_day.items()]
    tonight = next((c for c in cards if c["slate_date"] == today.isoformat()), None)
    when = lock_at(_tips(conn, sport, today))
    return {"slate_date": today.isoformat(), "locks_at": when.isoformat() if when else None,
            "tonight": tonight, "history": [c for c in cards if c is not tonight],
            "record": _record([c for c in cards if c is not tonight])}
