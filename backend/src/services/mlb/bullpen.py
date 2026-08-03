"""Bullpen workload and FIP — the two feature families the model cannot see.

Relievers throw roughly 40% of modern innings and the run-diff model has ZERO
visibility into them: no reliever quality, no recent usage, no closer
availability. Separately, the model judges pitchers by ERA, an outcome stat
carrying large defensive and sequencing luck, while FIP uses only the outcomes
a pitcher controls almost entirely — home runs, walks, hit batters, strikeouts.

Neither needs a new data source. A team's pitching game log covers every
pitcher who appeared; subtracting that game's starter leaves the bullpen. FIP
components already sit in the same per-game logs.

STATUS: acquisition, not promotion. The 2026 retrain established that 1,405
games cannot validate 33 features, let alone more. These exist so history
accumulates now — the alternative is reaching next April with none. Nothing
here is wired into MODEL_FEATURE_NAMES; that is a separate, gated decision
(docs/engineering/03 §4).

Pure functions; no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterable, Sequence

# FIP is scaled to look like ERA by adding a league constant, roughly
# league_ERA - league_raw_FIP. It drifts by season and run environment;
# 3.15 is a standard modern default. Callers should pass a fitted value once
# one is computed from league totals — the constant shifts every FIP equally,
# so it never changes rankings, only the scale.
FIP_CONSTANT_DEFAULT = 3.15


@dataclass(frozen=True)
class BullpenLine:
    """One game's relief work, derived as team minus starter."""

    innings: float
    earned_runs: int
    hits: int
    walks: int


def fip(
    home_runs: int,
    walks: int,
    hit_by_pitch: int,
    strikeouts: int,
    innings: float,
    constant: float = FIP_CONSTANT_DEFAULT,
) -> float | None:
    """Fielding Independent Pitching.

    Strips out everything that depends on the defence behind the pitcher and
    on sequencing luck, both of which make ERA a noisy estimate of true
    ability. Returns None on zero innings rather than dividing.
    """
    if not innings or innings <= 0:
        return None
    raw = (13 * home_runs + 3 * (walks + hit_by_pitch) - 2 * strikeouts) / innings
    return raw + constant


def bullpen_from_team_and_starter(
    team_ip: float | None,
    team_er: int | None,
    team_h: int | None,
    team_bb: int | None,
    starter_ip: float | None,
    starter_er: int | None,
    starter_h: int | None,
    starter_bb: int | None,
) -> BullpenLine | None:
    """Relief line for one game: the team's pitching minus its starter's.

    Returns None when either side is missing — without both, the split is
    unknowable and a guess would silently invent bullpen performance.

    Counting residuals are clamped at zero. Small inconsistencies between the
    team and player feeds could otherwise produce a bullpen that un-allowed
    runs, which would quietly flatter every downstream rate.
    """
    if team_ip is None or starter_ip is None:
        return None
    if any(v is None for v in (team_er, team_h, team_bb, starter_er, starter_h, starter_bb)):
        return None

    return BullpenLine(
        innings=max(0.0, team_ip - starter_ip),
        earned_runs=max(0, int(team_er) - int(starter_er)),
        hits=max(0, int(team_h) - int(starter_h)),
        walks=max(0, int(team_bb) - int(starter_bb)),
    )


def cumulative_bullpen(
    lines: Iterable[tuple[str, BullpenLine]],
    as_of: str,
) -> dict[str, float] | None:
    """Season-to-date bullpen rates from games STRICTLY BEFORE `as_of`.

    Rates are recomputed from cumulative components, never averaged from
    per-game rates — a one-inning blow-up and a four-inning shutout do not
    average.
    """
    prior: Sequence[BullpenLine] = [ln for d, ln in lines if d < as_of]
    if not prior:
        return None

    innings = sum(ln.innings for ln in prior)
    if innings <= 0:
        return None

    er = sum(ln.earned_runs for ln in prior)
    hits = sum(ln.hits for ln in prior)
    walks = sum(ln.walks for ln in prior)

    return {
        "bullpen_ip": round(innings, 2),
        "bullpen_era": round(9 * er / innings, 3),
        "bullpen_whip": round((hits + walks) / innings, 3),
        "bullpen_games": len(prior),
    }


def recent_workload(
    lines: Iterable[tuple[str, BullpenLine]],
    as_of: str,
    days: int = 3,
) -> float:
    """Relief innings thrown in the `days` before `as_of`. Fatigue, not quality.

    Excludes the target day itself: tonight's usage cannot inform tonight's
    prediction. Returns 0.0 rather than None for an idle pen — a rested
    bullpen is a real, meaningful state, not missing data.
    """
    target = date.fromisoformat(as_of)
    window_start = (target - timedelta(days=days)).isoformat()
    return round(
        sum(ln.innings for d, ln in lines if window_start <= d < as_of), 2
    )
