"""Append-only odds history for MLB — the record `mlb_markets` never kept.

`mlb_markets` is overwritten in place on every 30-minute ingest, so line
movement and closing prices are destroyed as they arrive. That is why
historical candidate sets are unrecoverable, why no backtest can ask "what
else was available that night", and why closing-line value has never been
computable for any pick.

This module supplies the pure logic behind the fix:

- `prices_changed` decides whether a fresh quote is worth a new row. Writing
  every book every 30 minutes would be ~16k rows/day of mostly duplicates;
  writing only on change stores exactly the line movement we want to analyse.
- `select_closing_quote` picks the last price observed before first pitch,
  which is what CLV is measured against.

Deliberately free of database and network dependencies so the rules are
testable in isolation. See `docs/engineering/06-roadmap-2026H2.md` Phase 1.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import datetime
from decimal import Decimal
from typing import Iterable, Sequence, TypeVar

Number = float | Decimal | None


@dataclass(frozen=True)
class OddsQuote:
    """One book's prices for one game at one moment, across all three markets.

    Packed into a single record (rather than one per market) because a book
    publishes them together; splitting them triples the row count and makes
    "this book moved" harder to reconstruct.
    """

    ml_home: Number = None
    ml_away: Number = None

    # Runline. `rl_line` is the SIGNED home line (-1.5 = home favoured).
    rl_line: Number = None
    rl_home_odds: Number = None
    rl_away_odds: Number = None

    total_line: Number = None
    over_odds: Number = None
    under_odds: Number = None


def quote_from_bookmaker(
    bookmaker: dict,
    home_team: str,
    name_to_abbr: dict[str, str],
) -> OddsQuote:
    """Collapse one bookmaker's Odds API payload into a single quote.

    Sides are resolved by team NAME, never by outcome order — the API makes no
    ordering guarantee, and mis-pairing sides is precisely the failure that
    produced the 2026-07-21 runline sign incident (favourite -1.5 prices
    labelled +1.5, inflating the tracked record for months).

    `rl_line` is stored signed from the home perspective, matching the scorer.
    An unrecognised team name is treated as the away side rather than dropped,
    so a roster/relocation change degrades to a usable quote instead of a gap.
    """
    ml_home = ml_away = None
    rl_line = rl_home_odds = rl_away_odds = None
    total_line = over_odds = under_odds = None

    for market in bookmaker.get("markets", []):
        key = market.get("key")
        outcomes = market.get("outcomes", [])

        if key == "h2h":
            for outcome in outcomes:
                if name_to_abbr.get(outcome.get("name")) == home_team:
                    ml_home = outcome.get("price")
                else:
                    ml_away = outcome.get("price")

        elif key == "spreads":
            for outcome in outcomes:
                if name_to_abbr.get(outcome.get("name")) == home_team:
                    rl_line = outcome.get("point")
                    rl_home_odds = outcome.get("price")
                else:
                    rl_away_odds = outcome.get("price")

        elif key == "totals":
            for outcome in outcomes:
                if outcome.get("name") == "Over":
                    total_line = outcome.get("point")
                    over_odds = outcome.get("price")
                else:
                    under_odds = outcome.get("price")

    return OddsQuote(
        ml_home=ml_home,
        ml_away=ml_away,
        rl_line=rl_line,
        rl_home_odds=rl_home_odds,
        rl_away_odds=rl_away_odds,
        total_line=total_line,
        over_odds=over_odds,
        under_odds=under_odds,
    )


def _norm(value: Number) -> float | None:
    """Normalise for comparison — values return from Numeric columns as Decimal."""
    if value is None:
        return None
    return float(value)


def prices_changed(previous: OddsQuote | None, current: OddsQuote) -> bool:
    """True when `current` differs from `previous` on any priced field.

    A missing previous quote counts as a change: the first observation of a
    game/book must always be recorded.

    Juice-only moves count. A book shading the vig while holding the line is
    precisely the market-dynamics signal Phase 4 needs, so it must not be
    filtered out as "no movement".
    """
    if previous is None:
        return True
    return any(
        _norm(getattr(previous, f.name)) != _norm(getattr(current, f.name))
        for f in fields(OddsQuote)
    )


def minutes_to_first_pitch(observed_at: datetime, first_pitch: datetime) -> int:
    """Whole minutes from `observed_at` until `first_pitch`.

    Positive before the game, negative after. The sign is kept meaningful so
    post-start rows can be excluded from closing-line selection rather than
    silently polluting it.
    """
    return round((first_pitch - observed_at).total_seconds() / 60)


T = TypeVar("T")


def select_closing_quote(
    quotes: Iterable[tuple[T, datetime]],
    first_pitch: datetime,
) -> T | None:
    """Identify the closing quote: the latest one observed at or before first pitch.

    Accepts `(identifier, observed_at)` pairs and returns the identifier, so
    callers can pass primary keys without loading whole rows.

    Returns None when nothing was observed pre-game — an honest "no closing
    line" beats defaulting to a live in-game price, which would make CLV look
    good precisely when the pipeline was broken.
    """
    eligible: Sequence[tuple[T, datetime]] = [
        (identifier, seen_at) for identifier, seen_at in quotes if seen_at <= first_pitch
    ]
    if not eligible:
        return None
    return max(eligible, key=lambda pair: pair[1])[0]
