"""Starter first-inning run prevention.

'Held opponents scoreless in the 1st in N of M starts' is the turn beat of the
pick-preview video — the moment it stops being a pick and becomes an argument.

The stat is always the OPPOSING side's first-inning runs. Callers are
responsible for selecting the correct column per appearance:

    home start -> mlb_games.away_first_inning_runs
    away start -> mlb_games.home_first_inning_runs

Pure functions only — no database, no network.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class StarterAppearance:
    """One start, with the runs the OPPOSING side scored in the 1st."""

    date: str  # ISO, YYYY-MM-DD
    opponent_first_inning_runs: int | None


@dataclass(frozen=True)
class FirstInningSplit:
    scoreless: int
    starts: int


def starter_first_inning_split(
    appearances: Iterable[StarterAppearance],
    as_of: str,
) -> FirstInningSplit | None:
    """Scoreless-first-inning count over starts strictly before `as_of`.

    Appearances with unrecorded first-inning runs are dropped from BOTH the
    numerator and the denominator. Counting them as scoreless would inflate the
    stat; counting them as scoring would deflate it. Neither is measured.

    Returns None when nothing is measurable — the beat is dropped rather than
    narrated as '0 of 0'.
    """
    usable = [
        a for a in appearances
        if a.date < as_of and a.opponent_first_inning_runs is not None
    ]
    if not usable:
        return None

    return FirstInningSplit(
        scoreless=sum(1 for a in usable if a.opponent_first_inning_runs == 0),
        starts=len(usable),
    )
