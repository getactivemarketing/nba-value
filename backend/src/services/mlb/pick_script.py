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
    if payload.last_10_record:
        clauses.append(f"They're {payload.last_10_record} in their last ten")
        chips.append(f"{payload.last_10_record} L10")
    if payload.streak is not None and payload.streak.length > 1:
        verb = "losing" if payload.streak.direction == "lost" else "winning"
        clauses.append(f"on a {payload.streak.length}-game {verb} streak")
        chips.append(f"{payload.streak.length}-game {verb} streak")
    if payload.starter_name and payload.starter_era is not None:
        clauses.append(
            f"and {payload.starter_name} carries a {payload.starter_era:.2f} ERA"
        )
        chips.append(f"{payload.starter_name} {payload.starter_era:.2f} ERA")

    if clauses:
        beats.append(Beat(
            "case_against",
            ", ".join(clauses) + ".",
            {"chips": " · ".join(chips)},
        ))

    # -- turn: the retention anchor ------------------------------------------
    if payload.first_inning is not None and payload.starter_name:
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

    return beats
