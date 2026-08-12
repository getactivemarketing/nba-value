"""Narration and overlay copy for the pick-preview video.

Deterministic templates, never an LLM: a model inventing a stat that ships
under TruLine's name with odds attached is a liability, and the variation the
feed wants comes from the data changing daily rather than the phrasing.

Structure — the video argues AGAINST itself before making its case, which is
what makes it read as analysis rather than a pitch:

    hook          the strongest reason not to take it
    pick          team, market, price
    case_against  form and starter ERA
    turn          the first-inning split — the retention anchor
    numbers       model projection vs breakeven
    close         CTA + disclaimer

Pure functions only — no database, no network.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.services.mlb.first_inning import FirstInningSplit
from src.services.mlb.team_form import Streak
from src.services.mlb.value_calculator import MLBValueCalculator

DISCLAIMER = "Not betting advice. 21+."

# Minimum starts before the first-inning split may be narrated as evidence.
#
# The turn beat is the video's centrepiece argument, and FirstInningSplit(1, 1)
# renders as "held opponents scoreless in the first in 1 of 1 starts" — a coin
# flip presented as a reason to bet, published with odds attached. Below this
# floor the beat is dropped entirely rather than softened: absent is not
# neutral, which is the same rule every other beat in this module follows for
# data it cannot stand behind.
MIN_TURN_BEAT_STARTS = 5


class NarrationContractError(RuntimeError):
    """Published narration violates a hard constraint.

    Raised loudly rather than published: a video narration claiming an unpaid
    result (like 'edge') or containing a banned word from database fields is
    a public factual claim with odds attached and must never ship.
    """


@dataclass(frozen=True)
class Beat:
    """One narrated segment and the graphic shown over it."""

    key: str
    narration: str
    overlay: dict[str, str]


@dataclass(frozen=True)
class PickPayload:
    team_abbr: str
    team_name: str
    odds_american: int
    model_prob: float
    last_10_record: str | None = None
    streak: Streak | None = None
    starter_name: str | None = None
    starter_era: float | None = None
    first_inning: FirstInningSplit | None = None


def breakeven_prob(odds_american: int) -> float:
    """Probability the offered price must beat to break even.

    Deliberately the RAW breakeven, not the devigged consensus: at +155 that is
    39.2% where the devigged fair price is nearer 38%, so quoting breakeven
    makes the model-vs-market gap look SMALLER. Between two defensible numbers,
    publish the conservative one.
    """
    return 1.0 / MLBValueCalculator.american_to_decimal(odds_american)


def _price(odds_american: int) -> str:
    return f"+{odds_american}" if odds_american > 0 else str(odds_american)


def build_beats(payload: PickPayload) -> list[Beat]:
    """Beats for one pick, in order. Unmeasurable beats are omitted entirely."""
    price = _price(payload.odds_american)
    market = breakeven_prob(payload.odds_american)
    beats: list[Beat] = []

    # -- hook: lead with the reason to walk away ------------------------------
    if payload.streak is not None and payload.streak.direction == "lost":
        hook = (
            f"Backing a team that's lost {payload.streak.length} straight."
            if payload.streak.length > 1
            else "Backing a team coming off a loss."
        )
    else:
        hook = f"The model's taking a {price} underdog tonight."
    beats.append(Beat("hook", hook, {"line": hook}))

    # -- pick ----------------------------------------------------------------
    beats.append(Beat(
        "pick",
        f"{payload.team_name}, moneyline, {price}.",
        {"team": payload.team_name.upper(), "market": "MONEYLINE", "price": price},
    ))

    # -- case against --------------------------------------------------------
    clauses: list[str] = []
    chips: list[str] = []

    # LOSING streaks only. This beat's entire job is the strongest reason NOT
    # to take the pick, and the turn beat that follows opens with "But" —
    # which only parses after a negative. A winning streak argues FOR the
    # pick, so admitting it here produced routine output like "They're 6-4 in
    # their last ten, on a 4-game winning streak. But Castillo has held..."
    # — a case against that makes the case for, then rebuts itself.
    losing_streak = (
        payload.streak
        if payload.streak is not None
        and payload.streak.direction == "lost"
        and payload.streak.length > 1
        else None
    )

    if payload.last_10_record:
        clauses.append(f"They're {payload.last_10_record} in their last ten")
        chips.append(f"{payload.last_10_record} L10")
        # If form exists, other clauses can be lowercase continuations
        if losing_streak is not None:
            clauses.append(f"on a {losing_streak.length}-game losing streak")
            chips.append(f"{losing_streak.length}-game losing streak")
    else:
        # Form not present, so streak needs to be its own complete sentence
        if losing_streak is not None:
            clauses.append(f"They're on a {losing_streak.length}-game losing streak")
            chips.append(f"{losing_streak.length}-game losing streak")

    # Starter always stands on its own with subject
    if payload.starter_name and payload.starter_era is not None:
        clauses.append(
            f"{payload.starter_name} carries a {payload.starter_era:.2f} ERA"
        )
        chips.append(f"{payload.starter_name} {payload.starter_era:.2f} ERA")

    if clauses:
        beats.append(Beat(
            "case_against",
            ", ".join(clauses) + ".",
            {"chips": " · ".join(chips)},
        ))

    # -- turn: the retention anchor ------------------------------------------
    if (
        payload.first_inning is not None
        and payload.first_inning.starts >= MIN_TURN_BEAT_STARTS
        and payload.starter_name
    ):
        split = payload.first_inning
        beats.append(Beat(
            "turn",
            f"But {payload.starter_name} has held opponents scoreless in the "
            f"first in {split.scoreless} of {split.starts} starts.",
            {
                "stat": f"{split.scoreless} of {split.starts}",
                "label": "SCORELESS 1ST",
            },
        ))

    # -- numbers -------------------------------------------------------------
    beats.append(Beat(
        "numbers",
        f"Model projection: {payload.model_prob:.0%}. "
        f"The price needs {market:.0%}.",
        {
            "model": f"{payload.model_prob:.0%}",
            "model_label": "MODEL PROJECTION",
            "market": f"{market:.0%}",
            "market_label": "BREAKEVEN",
        },
    ))

    # -- close ---------------------------------------------------------------
    beats.append(Beat(
        "close",
        "Full model at truline dot app.",
        {"cta": "truline.app", "disclaimer": DISCLAIMER},
    ))

    # -- contract: no banned words in opaque fields ---------------------------
    for beat in beats:
        if "edge" in beat.narration.lower():
            raise NarrationContractError(
                f"Narration contains 'edge': {beat.narration}"
            )
        for key, value in beat.overlay.items():
            if "edge" in value.lower():
                raise NarrationContractError(
                    f"Overlay[{key}] contains 'edge': {value}"
                )

    return beats
