"""Separate "no edge tonight" from "the data never arrived".

Every external dependency degrades into the same observable outcome: no picks.
A broken odds feed, an unmapped team name, and a genuinely quiet market look
identical from outside — which is how 111 Athletics games were skipped all
season because the Odds API started sending bare "Athletics" while the mapping
still expected "Oakland Athletics". The scheduler logged success every 30
minutes throughout.

That gap is wider now than it was. With runline and totals both disabled,
moneyline is the only market that can produce a pick, so silence is the
EXPECTED output on many nights and a broken feed hides inside it.

This assigns each game a reason and the slate a verdict, so "quiet" and
"broken" stop sharing a signature.

Pure functions; no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

# Books quoting a moneyline on a healthy game. The live slate carries 11
# consistently; anything far below means we are pricing against a fragment of
# the market, not that the market is quiet.
MIN_HEALTHY_BOOKS = 3

# One missing game on a full slate is ordinary (a postponement, a late
# listing). A meaningful share missing is a pipeline problem.
GAP_ALARM_FRACTION = 0.2


@dataclass(frozen=True)
class GameSlot:
    """One game's data availability and outcome.

    `decided` is False until the game has been through the snapshot window.
    Without it a pre-game slate reads as fifteen no-edge verdicts when in fact
    the model has not been asked yet — overstating "we looked and found
    nothing" for most of every day.
    """

    matchup: str
    ml_books: int
    has_pick: bool
    decided: bool = True


def diagnose_game(slot: GameSlot) -> dict:
    """Why this game produced no pick — or that it did.

    A pick always wins the classification: if we bet it, the data was
    sufficient by definition, however thin the market looked.
    """
    if slot.has_pick:
        return {"matchup": slot.matchup, "reason": "pick", "is_data_gap": False}

    if not slot.decided:
        # Not yet snapshotted. Absence of a pick says nothing yet, but a
        # missing market already does — so fall through when there are no
        # books, and only claim "pending" when the data is actually there.
        if slot.ml_books >= MIN_HEALTHY_BOOKS:
            return {"matchup": slot.matchup, "reason": "pending", "is_data_gap": False}

    if slot.ml_books <= 0:
        # The game exists but no book ever quoted it to us. The model never
        # got to have an opinion.
        return {"matchup": slot.matchup, "reason": "no_market", "is_data_gap": True}

    if slot.ml_books < MIN_HEALTHY_BOOKS:
        return {"matchup": slot.matchup, "reason": "thin_market", "is_data_gap": True}

    # Full market, no qualifying value. This is the legitimate quiet night.
    return {"matchup": slot.matchup, "reason": "no_edge", "is_data_gap": False}


def diagnose_slate(slots: Sequence[GameSlot]) -> dict:
    """Slate-level verdict, so a quiet night and a broken feed read differently."""
    if not slots:
        # Distinct from "no odds": nothing was ingested at all, which points at
        # the schedule feed rather than the odds feed.
        return {
            "verdict": "no_games",
            "games": 0, "picks": 0, "no_edge": 0, "pending": 0, "data_gaps": 0,
            "explanation": "no games ingested for this slate",
            "games_detail": [],
        }

    detail = [diagnose_game(s) for s in slots]
    picks = sum(1 for d in detail if d["reason"] == "pick")
    no_edge = sum(1 for d in detail if d["reason"] == "no_edge")
    pending = sum(1 for d in detail if d["reason"] == "pending")
    gaps = [d for d in detail if d["is_data_gap"]]

    gap_fraction = len(gaps) / len(slots)
    if gap_fraction >= GAP_ALARM_FRACTION:
        verdict = "data_gap"
    elif gaps:
        verdict = "warn"
    else:
        verdict = "ok"

    if gaps:
        named = ", ".join(d["matchup"] for d in gaps[:5])
        more = f" (+{len(gaps) - 5} more)" if len(gaps) > 5 else ""
        explanation = (
            f"{len(gaps)}/{len(slots)} games had no usable market: {named}{more}. "
            f"{picks} picks, {no_edge} genuinely no-edge, {pending} not yet scored."
        )
    else:
        explanation = (
            f"{picks} picks, {no_edge} no_edge, {pending} pending across "
            f"{len(slots)} games — full market coverage, silence is real."
        )

    return {
        "verdict": verdict,
        "games": len(slots),
        "picks": picks,
        "no_edge": no_edge,
        "pending": pending,
        "data_gaps": len(gaps),
        "explanation": explanation,
        "games_detail": detail,
    }
