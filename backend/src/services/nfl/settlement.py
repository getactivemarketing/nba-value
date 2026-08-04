"""Closing-line attachment and CLV settlement for shadow predictions.

TWO CLOSES ARE ATTACHED TO EVERY ROW, never one:

  same-book   what the bettor could actually have obtained at that book. The
              honest bettor-oriented number, and the one a real bet settles
              against.
  consensus   MEDIAN across books. Robust to one book going stale or wide —
              a mean would let a single broken feed drag the whole number.

Storing both means a disagreement between them is visible rather than
averaged away, and a book dropping out degrades to a labelled fallback instead
of a silent gap.

Frozen candidate and model fields are never rewritten. Settlement writes only
settlement columns; the forecast is a historical fact.

Pure functions; no I/O.
"""

from __future__ import annotations

import statistics
from datetime import datetime
from decimal import Decimal
from typing import Iterable, Sequence

from src.services.nfl.clv import crossed_key_number, line_clv, novig_two_way, price_clv


class CloseSource:
    """Where the attached close came from. Stored, not inferred later."""

    BOOK = "book"                          # same book the candidate came from
    CONSENSUS_FALLBACK = "consensus_fallback"   # that book had no pre-kickoff quote
    NONE = "none"                          # nothing usable — CLV stays NULL


def canonical_line(value) -> str:
    """Canonical string for a line, used in candidate identity.

    -3.50 and -3.5 are the same line; -0.0 and 0.0 are the same pick'em. Float
    formatting differences must never split one candidate into two rows, or
    idempotency silently stops working and the store fills with duplicates.
    """
    if value is None:
        return "NONE"
    v = float(Decimal(str(value)))
    if v == 0:
        v = 0.0          # collapse -0.0
    # Lines move in half points; normalise then strip trailing zeros.
    text = f"{v:.2f}".rstrip("0").rstrip(".")
    return text or "0"


def canonical_price(value) -> str:
    """Canonical string for a decimal price.

    Books quote two decimals. Float noise (1.9100000001) is not a price move
    and must not create a new candidate.
    """
    if value is None:
        return "NONE"
    v = float(Decimal(str(value)))
    text = f"{v:.2f}".rstrip("0").rstrip(".")
    return text or "0"


def _market_fields(market: str, side: str) -> tuple[str, str, str]:
    """(line_key, own_odds_key, opposing_odds_key) on an odds-history row."""
    if market == "spread":
        return ("spread_line",
                "spread_home_odds" if side == "home" else "spread_away_odds",
                "spread_away_odds" if side == "home" else "spread_home_odds")
    if market == "total":
        return ("total_line",
                "over_odds" if side == "over" else "under_odds",
                "under_odds" if side == "over" else "over_odds")
    return ("", "ml_home" if side == "home" else "ml_away",
            "ml_away" if side == "home" else "ml_home")


def select_book_close(
    quotes: Iterable[dict],
    kickoff: datetime,
) -> dict | None:
    """Latest quote at or before kickoff, with its staleness recorded.

    Staleness is reported even when large. A three-day-old quote is still the
    closing price we observed, but calling it a "close" without saying how old
    it is would overstate what was captured.
    """
    eligible = [q for q in quotes if q.get("snapshot_time") and q["snapshot_time"] <= kickoff]
    if not eligible:
        return None
    latest = max(eligible, key=lambda q: q["snapshot_time"])
    return {
        **latest,
        "staleness_min": int(round((kickoff - latest["snapshot_time"]).total_seconds() / 60)),
    }


def consensus_close(
    quotes: Iterable[dict],
    kickoff: datetime,
    market: str,
    side: str = "home",
    min_books: int = 3,
) -> dict | None:
    """Median close across books, each contributing its own latest pre-kickoff quote.

    Median rather than mean on purpose: one stale or wide book should not move
    the consensus, and in a thin overnight market it otherwise would.
    """
    line_key, own_key, opp_key = _market_fields(market, side)

    per_book: dict[str, dict] = {}
    for q in quotes:
        book, seen = q.get("book"), q.get("snapshot_time")
        if not book or not seen or seen > kickoff:
            continue
        if book not in per_book or seen > per_book[book]["snapshot_time"]:
            per_book[book] = q

    lines, probs = [], []
    for q in per_book.values():
        if line_key and q.get(line_key) is not None:
            lines.append(float(q[line_key]))
        own, opp = q.get(own_key), q.get(opp_key)
        if own and opp:
            p_own, _ = novig_two_way(float(own), float(opp))
            if p_own is not None:
                probs.append(p_own)

    usable = max(len(lines), len(probs))
    if usable < min_books:
        return None

    return {
        "line": statistics.median(lines) if lines else None,
        "novig_prob": statistics.median(probs) if probs else None,
        "book_count": len(per_book),
    }


def settle_candidate(
    candidate: dict,
    book_close: dict | None,
    consensus: dict | None,
    actual_value: float | None,
) -> dict:
    """Attach closes, compute CLV both ways, and grade the side.

    The outcome is recorded even when no close is available — knowing the bet
    won is independent of knowing whether the price was good. CLV stays NULL
    rather than 0, so an unmeasured row can never be mistaken for a neutral one.
    """
    market, side = candidate["market_type"], candidate["side"]
    bet_line = candidate.get("line")
    bet_odds = candidate.get("odds_decimal")

    out: dict = {
        "closing_source": CloseSource.NONE,
        "closing_line": None, "closing_odds": None, "closing_novig_prob": None,
        "closing_staleness_min": None,
        "consensus_line": None, "consensus_novig_prob": None, "consensus_book_count": None,
        "line_clv": None, "price_clv": None,
        "line_clv_consensus": None, "price_clv_consensus": None,
        "crossed_key": None,
        "actual_value": actual_value,
        "side_won": None,
    }

    # --- consensus (always recorded when available) ----------------------
    if consensus:
        out["consensus_line"] = consensus.get("line")
        out["consensus_novig_prob"] = consensus.get("novig_prob")
        out["consensus_book_count"] = consensus.get("book_count")
        out["line_clv_consensus"] = line_clv(bet_line, consensus.get("line"), side)
        out["price_clv_consensus"] = price_clv(bet_odds, consensus.get("novig_prob"))

    # --- primary close: the same book, else consensus, else nothing ------
    if book_close:
        _, own_key, opp_key = _market_fields(market, side)
        line_key, _, _ = _market_fields(market, side)
        close_line = book_close.get(line_key) if line_key else None
        own, opp = book_close.get("odds") or book_close.get(own_key), \
                   book_close.get("opposing_odds") or book_close.get(opp_key)
        if book_close.get("line") is not None:
            close_line = book_close["line"]

        novig, _ = novig_two_way(own, opp) if own and opp else (None, None)

        out.update({
            "closing_source": CloseSource.BOOK,
            "closing_line": None if close_line is None else float(close_line),
            "closing_odds": None if own is None else float(own),
            "closing_novig_prob": novig,
            "closing_staleness_min": book_close.get("staleness_min"),
            "line_clv": line_clv(bet_line, close_line, side),
            "price_clv": price_clv(bet_odds, novig),
            "crossed_key": crossed_key_number(bet_line, close_line),
        })
    elif consensus:
        out.update({
            "closing_source": CloseSource.CONSENSUS_FALLBACK,
            "closing_line": consensus.get("line"),
            "closing_novig_prob": consensus.get("novig_prob"),
            "line_clv": out["line_clv_consensus"],
            "price_clv": out["price_clv_consensus"],
            "crossed_key": crossed_key_number(bet_line, consensus.get("line")),
        })

    # --- grade the side (independent of whether a close was found) -------
    if actual_value is not None:
        graded = _grade(market, side, bet_line, float(actual_value))
        out["side_won"] = graded

    return out


def _grade(market: str, side: str, bet_line, actual: float) -> bool | None:
    """Did this side win? None on an exact push.

    TWO OPPOSITE SPREAD CONVENTIONS LIVE IN THIS SYSTEM — get this wrong and
    every graded spread flips:

      Odds API / nfl_odds_snapshots : NEGATIVE when the home team is favoured
                                      (home -3.5 means home lays 3.5)
      nflverse / training data      : POSITIVE when the home team is favoured
                                      (verified: corr(spread_line, margin)
                                      +0.452, MAE 9.806 using +)

    Candidates come from the ODDS API side, so a home line of -3.5 means the
    home team must win by more than 3.5. `actual` is always the home margin
    (home - away), or the total for totals markets.
    """
    if market == "moneyline":
        if actual == 0:
            return None
        return actual > 0 if side == "home" else actual < 0

    if bet_line is None:
        return None
    line = float(bet_line)

    if market == "spread":
        # home  line -3.5 -> must beat +3.5   (threshold = -line, wants over)
        # away  line +3.5 -> must stay under 3.5 (threshold =  line, wants under)
        threshold = -line if side == "home" else line
        if actual == threshold:
            return None
        return actual > threshold if side == "home" else actual < threshold

    # totals
    if actual == line:
        return None
    return actual > line if side == "over" else actual < line
