"""Point-in-time team rate stats reconstructed from MLB game logs.

The same gap the pitcher backfill closed, one level up. `update_team_stats`
only began writing OPS/AVG/OBP/SLG/ERA/WHIP on 2026-07-31, so all 111 days of
prior history carried NULLs — 0% coverage in the 2026 training set for six
features the run-diff model was trained on with real variance.

Team game logs give per-game component counts, so cumulative rates can be
rebuilt as they stood before any game.

CRITICAL: rates are recomputed from cumulative COMPONENTS, never averaged from
per-game rates. A 1-for-4 game and a 3-for-3 game make a .571 hitter, not a
.625 one. Averaging rates over-weights low-volume games and would bias every
row in the training set.

Pure functions; no database, no network.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from src.services.mlb.pitcher_history import parse_innings


@dataclass(frozen=True)
class TeamGameLog:
    """One game's component counts. Hitting and pitching arrive separately."""

    date: str

    # Hitting
    at_bats: int = 0
    hits: int = 0
    total_bases: int = 0
    walks: int = 0
    hit_by_pitch: int = 0
    sac_flies: int = 0

    # Pitching
    innings_pitched: Any = None
    earned_runs: int = 0
    hits_allowed: int = 0
    walks_allowed: int = 0


def cumulative_team_stats(
    hitting: Iterable[TeamGameLog],
    pitching: Iterable[TeamGameLog],
    as_of: str,
) -> dict[str, float | None] | None:
    """Season-to-date rates from every game STRICTLY BEFORE `as_of`.

    Returns None when there is no prior hitting or pitching to summarise —
    inventing an opening-day rate would hand the model a fabricated team.
    Individual rates are None when their denominator is empty.
    """
    prior_hit: Sequence[TeamGameLog] = [g for g in hitting if g.date < as_of]
    prior_pit: Sequence[TeamGameLog] = [g for g in pitching if g.date < as_of]

    if not prior_hit and not prior_pit:
        return None

    ab = sum(g.at_bats or 0 for g in prior_hit)
    h = sum(g.hits or 0 for g in prior_hit)
    tb = sum(g.total_bases or 0 for g in prior_hit)
    bb = sum(g.walks or 0 for g in prior_hit)
    hbp = sum(g.hit_by_pitch or 0 for g in prior_hit)
    sf = sum(g.sac_flies or 0 for g in prior_hit)

    innings = 0.0
    er = ha = bba = 0
    for game in prior_pit:
        ip = parse_innings(game.innings_pitched)
        if ip is None:
            continue
        innings += ip
        er += game.earned_runs or 0
        ha += game.hits_allowed or 0
        bba += game.walks_allowed or 0

    avg = round(h / ab, 4) if ab else None
    slg = round(tb / ab, 4) if ab else None
    obp_denom = ab + bb + hbp + sf
    obp = round((h + bb + hbp) / obp_denom, 4) if obp_denom else None
    ops = round(obp + slg, 4) if (obp is not None and slg is not None) else None

    era = round(9 * er / innings, 3) if innings > 0 else None
    whip = round((ha + bba) / innings, 3) if innings > 0 else None

    if avg is None and era is None:
        return None

    return {
        "batting_avg": avg,
        "obp": obp,
        "slg": slg,
        "ops": ops,
        "era": era,
        "team_whip": whip,
    }


def _next_day(iso_date: str) -> str:
    from datetime import date, timedelta

    y, m, d = (int(p) for p in iso_date.split("-"))
    return (date(y, m, d) + timedelta(days=1)).isoformat()


def build_team_stat_history(
    hitting: Iterable[TeamGameLog],
    pitching: Iterable[TeamGameLog],
) -> list[dict]:
    """One cumulative row per game day, dated the day AFTER it.

    Dating rows to the following day encodes when the numbers became knowable,
    which is what makes the existing lookup (`stat_date < game_date`, most
    recent first) correct without changing it.
    """
    hitting = list(hitting)
    pitching = list(pitching)
    game_days = sorted({g.date for g in hitting} | {g.date for g in pitching})

    history: list[dict] = []
    for day in game_days:
        as_of = _next_day(day)
        stats = cumulative_team_stats(hitting, pitching, as_of)
        if stats is None:
            continue
        history.append({"stat_date": as_of, **stats})

    return history
