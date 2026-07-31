"""Closing-line value for MLB picks.

Realized P&L over a few hundred bets cannot separate edge from luck — the
moneyline record is +1.18 standard errors from zero. CLV can, in roughly a
tenth the sample, because it strips out game-outcome randomness entirely and
asks only whether we bought a price the market later disagreed with.

    clv = closing_novig_prob - (1 / bet_decimal_odds)

The market's vig-free consensus probability for the side we took, minus the
probability we needed to break even at the price we got. Positive means the
closing market says our price was profitable, whether or not the bet won.

Devigging uses the same multiplicative method the scorer uses at bet time, so
the two probabilities are directly comparable rather than merely similar.

Pure functions only — no database, no network. See
`docs/engineering/06-roadmap-2026H2.md` Phase 1.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from decimal import Decimal
from typing import Iterable, Sequence

from src.services.mlb.value_calculator import MLBValueCalculator

Number = float | Decimal | None

# A single book is a quote, not a consensus. Three keeps one stale or wide
# book from dominating while staying achievable on thin overnight markets.
DEFAULT_MIN_BOOKS = 3


@dataclass(frozen=True)
class ClosingQuote:
    """One book's closing prices, as stored on `mlb_odds_snapshots`."""

    ml_home: Number = None
    ml_away: Number = None
    rl_line: Number = None       # SIGNED home line
    rl_home_odds: Number = None
    rl_away_odds: Number = None
    total_line: Number = None
    over_odds: Number = None
    under_odds: Number = None


def _f(value: Number) -> float | None:
    return None if value is None else float(value)


def novig_prob_for_side(
    quote: ClosingQuote,
    market: str,
    is_home: bool,
    line: Number = None,
    direction: str | None = None,
) -> float | None:
    """Vig-free probability this book assigned to the side we bet.

    Returns None when the quote cannot price that exact bet — a missing side,
    or a line that has moved. A moved line is a *different bet*, and silently
    comparing across it would manufacture CLV that was never available.
    """
    if market == "moneyline":
        home, away = _f(quote.ml_home), _f(quote.ml_away)
        if not home or not away:
            return None
        p_home, p_away = MLBValueCalculator.devig_odds(home, away)
        return p_home if is_home else p_away

    if market == "runline":
        rl_line, home, away = _f(quote.rl_line), _f(quote.rl_home_odds), _f(quote.rl_away_odds)
        if rl_line is None or not home or not away or line is None:
            return None
        # rl_line is stored from the home perspective; the away side is its mirror.
        expected = rl_line if is_home else -rl_line
        if float(line) != expected:
            return None
        p_home, p_away = MLBValueCalculator.devig_odds(home, away)
        return p_home if is_home else p_away

    if market == "total":
        total_line, over, under = _f(quote.total_line), _f(quote.over_odds), _f(quote.under_odds)
        if total_line is None or not over or not under or line is None:
            return None
        if float(line) != total_line:
            return None
        p_over, p_under = MLBValueCalculator.devig_odds(over, under)
        return p_over if (direction or "over") == "over" else p_under

    return None


def consensus_novig_prob(
    probs: Iterable[float | None],
    min_books: int = DEFAULT_MIN_BOOKS,
) -> float | None:
    """Mean vig-free probability across books that could price the exact bet.

    A plain mean, not a best-price pick: we want the market's final collective
    assessment, not the most favourable corner of it. Books that could not
    price the bet are dropped rather than counted as neutral.
    """
    usable: Sequence[float] = [p for p in probs if p is not None]
    if len(usable) < max(1, min_books):
        return None
    return sum(usable) / len(usable)


# -- sample-size thresholds for the reported verdict --------------------------
# CLV converges far faster than win rate, but not instantly. 30 is the point
# a mean stops being anecdote; 100 is where it is worth acting on. Chosen
# before looking at any CLV data — see docs/engineering/04 §4 on
# pre-registration.
VERDICT_PROVISIONAL_AT = 30
VERDICT_MEANINGFUL_AT = 100


def summarize_clv(values: Iterable[float | None]) -> dict:
    """Aggregate per-pick CLV into a reportable summary.

    Unmeasured picks (None) are excluded rather than counted as zero: treating
    "no closing line" as neutral would pull the mean toward zero and let a
    broken capture pipeline read as an absence of edge.

    `verdict` deliberately depends only on sample size, never on how good the
    number looks — a +25pt mean over five picks is still not evidence.
    """
    measured: Sequence[float] = [float(v) for v in values if v is not None]
    n = len(measured)

    if n == 0:
        return {
            "measured": 0,
            "mean_clv": None,
            "median_clv": None,
            "beat_close_rate": None,
            "std_error": None,
            "verdict": "insufficient",
        }

    mean = statistics.fmean(measured)
    std_error = (
        round(statistics.stdev(measured) / (n ** 0.5), 5) if n > 1 else None
    )

    if n >= VERDICT_MEANINGFUL_AT:
        verdict = "meaningful"
    elif n >= VERDICT_PROVISIONAL_AT:
        verdict = "provisional"
    else:
        verdict = "insufficient"

    return {
        "measured": n,
        "mean_clv": round(mean, 5),
        "median_clv": round(statistics.median(measured), 5),
        "beat_close_rate": round(sum(1 for v in measured if v > 0) / n, 4),
        "std_error": std_error,
        "verdict": verdict,
    }


def clv_from_closing(
    bet_odds: Number,
    closing_novig_prob: float | None,
) -> float | None:
    """Closing-line value in probability points.

    None when either input is missing — an unmeasured pick must never read as
    neutral CLV, which would quietly dilute the average toward zero and make a
    broken capture pipeline look like an absence of edge.
    """
    odds = _f(bet_odds)
    if odds is None or odds <= 1.0 or closing_novig_prob is None:
        return None
    return closing_novig_prob - (1.0 / odds)
