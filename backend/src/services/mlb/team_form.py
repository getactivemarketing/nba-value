"""Current win/loss streak for a team.

Used by the 'case against' beat of the pick-preview video, which states the
strongest reason NOT to take the pick before making its case.

Pure functions only — no database, no network. See
`docs/superpowers/specs/2026-08-10-tiktok-pick-preview-videos-design.md`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class GameResult:
    """One completed game from a single team's perspective."""

    date: str  # ISO, YYYY-MM-DD
    won: bool


@dataclass(frozen=True)
class Streak:
    direction: str  # "won" | "lost"
    length: int


def current_streak(results: Iterable[GameResult], as_of: str) -> Streak | None:
    """Consecutive same-result games ending most recently before `as_of`.

    Sorting happens here rather than being required of the caller: a caller
    that passes rows in query order would silently get a wrong streak, and a
    wrong streak is narrated aloud in a published video.

    Returns None when there are no prior games — an unmeasured streak drops
    its clause rather than being narrated as zero.
    """
    prior = sorted((r for r in results if r.date < as_of), key=lambda r: r.date)
    if not prior:
        return None

    latest = prior[-1].won
    length = 0
    for result in reversed(prior):
        if result.won != latest:
            break
        length += 1

    return Streak("won" if latest else "lost", length)
