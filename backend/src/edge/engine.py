# backend/src/edge/engine.py
"""Pure pricing functions: odds math, power devig, consensus fair price, EV, CLV.

No I/O. Spec §6.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, Mapping

from src.edge.config import BOOK_WEIGHTS, DEFAULT_RULE, DFS_BOOKS, MARKET_RULES, MAX_HOLD, SHARP_BOOKS
from src.edge.quotes import Quote, sides_for

MarketKey = tuple[str, str, str, float]
BookSides = Mapping[str, Mapping[str, Quote]]


def american_to_decimal(price: int) -> float:
    if price == 0:
        raise ValueError("American price cannot be 0")
    return 1 + price / 100 if price > 0 else 1 + 100 / abs(price)


def implied_prob(price: int) -> float:
    return 1 / american_to_decimal(price)


def prob_to_american(p: float) -> int:
    if not 0 < p < 1:
        raise ValueError(f"probability out of range: {p}")
    return round(-100 * p / (1 - p)) if p >= 0.5 else round(100 * (1 - p) / p)


def devig_power(p1: float, p2: float) -> tuple[float, float]:
    """Find k with p1**k + p2**k == 1 (bisection), return the two fair probs.

    Unlike multiplicative devig, the power method takes more margin off the
    longshot than the favourite, which is where books actually put it.
    """
    lo, hi = 0.01, 100.0
    for _ in range(200):
        k = (lo + hi) / 2
        if p1 ** k + p2 ** k > 1:
            lo = k
        else:
            hi = k
    k = (lo + hi) / 2
    a, b = p1 ** k, p2 ** k
    return a / (a + b), b / (a + b)


@dataclass(frozen=True)
class Fair:
    prob: float | None   # None => thin market, no EV
    n_books: int
    has_sharp: bool

    @property
    def thin(self) -> bool:
        return self.prob is None


def group(quotes: Iterable[Quote]) -> dict[MarketKey, dict[str, dict[str, Quote]]]:
    out: dict[MarketKey, dict[str, dict[str, Quote]]] = defaultdict(lambda: defaultdict(dict))
    for q in quotes:
        cur = out[q.market_key][q.book].get(q.side)
        # Exchanges post a ladder per side: keep the price a bettor can actually take.
        if cur is None or american_to_decimal(q.price) > american_to_decimal(cur.price):
            out[q.market_key][q.book][q.side] = q
    return {k: {b: dict(s) for b, s in v.items()} for k, v in out.items()}


def fair_prob(book_sides: BookSides, market: str, side: str, exclude_book: str | None) -> Fair:
    a, b = sides_for(market)
    rule = MARKET_RULES.get(market, DEFAULT_RULE)
    weighted, weights, n, sharp = 0.0, 0.0, 0, False
    for book, sides in book_sides.items():
        if book == exclude_book or book in DFS_BOOKS or a not in sides or b not in sides:
            continue
        if rule.inputs is not None and book not in rule.inputs:
            continue
        ia, ib = implied_prob(sides[a].price), implied_prob(sides[b].price)
        if ia + ib - 1 > MAX_HOLD:
            continue
        pa, pb = devig_power(ia, ib)
        w = BOOK_WEIGHTS.get(book, 1.0)
        weighted += (pa if side == a else pb) * w
        weights += w
        n += 1
        sharp = sharp or book in SHARP_BOOKS
    if n < rule.min_books or (rule.require_sharp and not sharp):
        return Fair(None, n, sharp)
    return Fair(weighted / weights, n, sharp)


@dataclass(frozen=True)
class Priced:
    quote: Quote
    fair: Fair
    ev: float | None
    is_ev: bool
    verify: bool


def price_quotes(quotes: Iterable[Quote], min_ev: float, verify_ev: float) -> list[Priced]:
    out: list[Priced] = []
    for (_, market, _, _), books in group(quotes).items():
        for book, sides in books.items():
            if book in DFS_BOOKS:
                continue
            for side, q in sides.items():
                f = fair_prob(books, market, side, exclude_book=book)
                ev = None if f.thin else f.prob * american_to_decimal(q.price) - 1
                out.append(Priced(q, f, ev,
                                  is_ev=ev is not None and ev >= min_ev,
                                  verify=ev is not None and ev > verify_ev))
    return out


def latest_before(quotes: Iterable[Quote], cutoff: datetime, window: timedelta) -> list[Quote]:
    """Last observation per price key at or before `cutoff`, no older than `window`."""
    best: dict[tuple, Quote] = {}
    for q in quotes:
        if q.observed_at > cutoff or q.observed_at < cutoff - window:
            continue
        cur = best.get(q.key)
        if cur is None or q.observed_at > cur.observed_at:
            best[q.key] = q
    return list(best.values())


def clv(closing_prob: float, price: int) -> float:
    """Same definition as src/services/mlb/clv.py: closing fair prob minus break-even prob."""
    return closing_prob - 1 / american_to_decimal(price)
