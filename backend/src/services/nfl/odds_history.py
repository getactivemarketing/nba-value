"""Append-only NFL odds history — the record `nfl_markets` does not keep.

`nfl_markets` is overwritten in place, exactly as `mlb_markets` was. That
already cost MLB every historical candidate set: no backtest can ask "what else
was available", and closing-line value was uncomputable until an append-only
table was added on 2026-07-31 — months of the season already lost.

NFL has not opened yet. This is the one time the mistake can be prevented
instead of repaired, which is why it goes in before the first preseason slate
rather than after the model is finished.

Stricter than the MLB equivalent because NFL markets are line-AND-price:
a book can hold -3.5 and move the juice from -110 to -120. That is a real
market signal and must count as movement, not be filtered away.

Pure functions; no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import datetime
from decimal import Decimal
from typing import Iterable, Sequence, TypeVar

Number = float | Decimal | None


@dataclass(frozen=True)
class NFLOddsQuote:
    """One book's prices for one game at one moment, across all three markets."""

    ml_home: Number = None
    ml_away: Number = None

    # SIGNED home spread: -3.5 means the home team is laying 3.5.
    spread_line: Number = None
    spread_home_odds: Number = None
    spread_away_odds: Number = None

    total_line: Number = None
    over_odds: Number = None
    under_odds: Number = None


def _norm(value: Number) -> float | None:
    return None if value is None else float(value)


def nfl_prices_changed(previous: NFLOddsQuote | None, current: NFLOddsQuote) -> bool:
    """True when any priced field differs. A missing previous counts as changed.

    Line and price are compared independently: holding the spread while shading
    the juice is exactly the kind of move the market-intelligence layer needs,
    and treating it as "unchanged" would discard it permanently.
    """
    if previous is None:
        return True
    return any(
        _norm(getattr(previous, f.name)) != _norm(getattr(current, f.name))
        for f in fields(NFLOddsQuote)
    )


def minutes_to_kickoff(observed_at: datetime, kickoff: datetime) -> int:
    """Whole minutes until kickoff; negative once the game has started.

    The sign is load-bearing — it is what keeps an in-game price from being
    mistaken for a closing line.
    """
    return round((kickoff - observed_at).total_seconds() / 60)


T = TypeVar("T")


def select_closing_quote(
    quotes: Iterable[tuple[T, datetime]],
    kickoff: datetime,
) -> T | None:
    """The last quote observed at or before kickoff, or None if there is none.

    None rather than a fallback: an honest gap beats a live in-game price
    posing as the close, which would flatter CLV precisely when capture broke.
    """
    eligible: Sequence[tuple[T, datetime]] = [
        (ident, seen) for ident, seen in quotes if seen <= kickoff
    ]
    if not eligible:
        return None
    return max(eligible, key=lambda pair: pair[1])[0]


def quote_from_bookmaker(
    bookmaker: dict,
    home_team: str,
    name_to_abbr: dict[str, str],
) -> NFLOddsQuote:
    """Collapse one bookmaker's Odds API payload into a single quote.

    Sides are resolved by team NAME, never outcome order. The MLB runline
    sign-pairing incident — favourite -1.5 prices labelled +1.5, inflating a
    losing market's record for months — came from exactly this, and an NFL
    spread has the same failure mode with more games riding on it.
    """
    ml_home = ml_away = None
    spread_line = spread_home_odds = spread_away_odds = None
    total_line = over_odds = under_odds = None

    for market in bookmaker.get("markets", []):
        key = market.get("key")
        outcomes = market.get("outcomes", [])

        if key == "h2h":
            for o in outcomes:
                if name_to_abbr.get(o.get("name")) == home_team:
                    ml_home = o.get("price")
                else:
                    ml_away = o.get("price")

        elif key == "spreads":
            for o in outcomes:
                if name_to_abbr.get(o.get("name")) == home_team:
                    spread_line = o.get("point")
                    spread_home_odds = o.get("price")
                else:
                    spread_away_odds = o.get("price")

        elif key == "totals":
            for o in outcomes:
                if o.get("name") == "Over":
                    total_line = o.get("point")
                    over_odds = o.get("price")
                else:
                    under_odds = o.get("price")

    return NFLOddsQuote(
        ml_home=ml_home, ml_away=ml_away,
        spread_line=spread_line,
        spread_home_odds=spread_home_odds,
        spread_away_odds=spread_away_odds,
        total_line=total_line, over_odds=over_odds, under_odds=under_odds,
    )
