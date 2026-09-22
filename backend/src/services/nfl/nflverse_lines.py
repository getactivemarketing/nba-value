"""Fallback betting lines from nflverse schedules, for when The Odds API is out.

The Odds API is one paid key, and when it lapsed on 2026-09-07 nothing noticed
for two weeks: Weeks 1 and 2 produced zero snapshots because every game was
(correctly) skipped on stale lines. nflverse's schedule file carries a spread,
a total and both moneylines per game, posted days ahead, for free. It is a
worse instrument -- one consensus line, no per-book prices, no intra-day
movement -- but it is always available, and collecting a weaker number beats
collecting nothing.

Sign conventions, which differ by destination and are easy to invert:
  - nflverse `spread_line`  : POSITIVE when the home team is favoured (+6 = home -6)
  - `nfl_markets.line`      : same convention, so it copies across unchanged
  - `nfl_odds_snapshots.spread_line`: NEGATIVE when the home team is favoured
                              (see settlement.py), so it is the negation

Prices: nflverse posts no juice on the spread or total, only the game's
moneylines. Those rows are written at a standard -110 so the scorer (which
requires both sides priced) can evaluate them. That price is an assumption, not
an observation, which is exactly why every row is labelled `book="nflverse"`:
line CLV against these rows is meaningful, price CLV is not, and the label is
what lets the gate's analysis drop them.
"""
from datetime import datetime, timezone

import pandas as pd

from src.services.nfl.odds_history import NFLOddsQuote

BOOK = "nflverse"

# -110 in decimal, the standard price for a spread or total.
STANDARD_PRICE_DECIMAL = 1.9091


def american_to_decimal(odds) -> float | None:
    """American odds -> decimal, rounded to 4dp (matches STANDARD_PRICE_DECIMAL)."""
    if odds is None or (isinstance(odds, float) and pd.isna(odds)):
        return None
    odds = float(odds)
    if odds == 0:
        return None
    dec = 1 + (odds / 100.0 if odds > 0 else 100.0 / abs(odds))
    return round(dec, 4)


def _num(value) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return float(value)


def lines_from_schedule(sched: pd.DataFrame) -> list[dict]:
    """Regular-season games that already have a spread and total posted."""
    if "game_type" in sched:
        sched = sched[sched["game_type"] == "REG"]

    lines = []
    for row in sched.to_dict("records"):
        spread, total = _num(row.get("spread_line")), _num(row.get("total_line"))
        if spread is None or total is None:
            continue   # not on the board yet
        lines.append({
            "game_id": row["game_id"],
            "spread_line": spread,          # home expected margin
            "total_line": total,
            "ml_home": american_to_decimal(row.get("home_moneyline")),
            "ml_away": american_to_decimal(row.get("away_moneyline")),
        })
    return lines


def upcoming_only(lines: list[dict], kickoffs: dict, now: datetime | None = None) -> list[dict]:
    """Drop games that have kicked off.

    After kickoff the schedule file holds the CLOSING line. Writing it as a
    fresh capture would invent pre-game history that was never observed, and
    the snapshot store is append-only.
    """
    now = now or datetime.now(timezone.utc)
    kept = []
    for line in lines:
        kickoff = kickoffs.get(line["game_id"])
        if kickoff is None:
            continue
        if kickoff.tzinfo is None:
            kickoff = kickoff.replace(tzinfo=timezone.utc)
        if kickoff > now:
            kept.append(line)
    return kept


def market_rows(line: dict) -> list[dict]:
    """One nfl_markets-shaped dict per market for a single game."""
    return [
        {
            "game_id": line["game_id"], "market_type": "spread",
            "line": line["spread_line"],
            "home_odds": STANDARD_PRICE_DECIMAL, "away_odds": STANDARD_PRICE_DECIMAL,
            "over_odds": None, "under_odds": None, "book": BOOK,
        },
        {
            "game_id": line["game_id"], "market_type": "total",
            "line": line["total_line"],
            "home_odds": None, "away_odds": None,
            "over_odds": STANDARD_PRICE_DECIMAL, "under_odds": STANDARD_PRICE_DECIMAL,
            "book": BOOK,
        },
        {
            "game_id": line["game_id"], "market_type": "moneyline",
            "line": None,
            "home_odds": line["ml_home"], "away_odds": line["ml_away"],
            "over_odds": None, "under_odds": None, "book": BOOK,
        },
    ]


def quote_from_line(line: dict) -> NFLOddsQuote:
    """nfl_odds_snapshots quote — note the spread sign flip (see module docstring)."""
    return NFLOddsQuote(
        ml_home=line["ml_home"], ml_away=line["ml_away"],
        spread_line=-line["spread_line"],
        spread_home_odds=STANDARD_PRICE_DECIMAL, spread_away_odds=STANDARD_PRICE_DECIMAL,
        total_line=line["total_line"],
        over_odds=STANDARD_PRICE_DECIMAL, under_odds=STANDARD_PRICE_DECIMAL,
    )
