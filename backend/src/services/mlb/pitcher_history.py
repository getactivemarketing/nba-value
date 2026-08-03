"""Point-in-time pitcher stats reconstructed from MLB game logs.

`mlb_pitcher_stats` only ever held one season-to-date snapshot, taken whenever
the ingest last ran. Training a model on an April game with July's numbers is
leakage, and it is the reason a 2026 retrain was not previously possible even
though 1,405 completed games were sitting in the database.

MLB's gameLog endpoint returns per-start lines with dates, so cumulative stats
can be rebuilt exactly as they stood before any given game. That turns the
starting-pitcher features — the widest-varying inputs the model has, with
starter_era_diff std 2.665 in the original training set — into something a
2026 retrain can legitimately use.

THE TRAP: MLB writes innings pitched in baseball notation. "6.1" is six and
one THIRD innings, not 6.1. Treating it as a decimal quietly corrupts ERA,
WHIP, K/9 and BB/9 — every rate stat divides by innings.

Pure functions only; no database, no network.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class GameLogEntry:
    """One appearance, as returned by the gameLog endpoint."""

    date: str                    # ISO 'YYYY-MM-DD'
    innings_pitched: Any         # baseball notation, e.g. '6.1'
    earned_runs: int = 0
    hits: int = 0
    walks: int = 0
    strikeouts: int = 0
    # FIP inputs. ERA carries defensive and sequencing luck; FIP uses only
    # what the pitcher controls almost entirely.
    home_runs: int = 0
    hit_by_pitch: int = 0


def parse_innings(raw: Any) -> float | None:
    """Convert baseball innings notation to true innings.

    '6.0' -> 6.0, '6.1' -> 6.333, '6.2' -> 6.667.

    Returns None for missing or unparseable input rather than 0.0, which would
    quietly deflate every rate stat that divides by innings.
    """
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None

    if "." not in text:
        try:
            return float(int(text))
        except ValueError:
            return None

    whole, _, frac = text.partition(".")
    try:
        innings = float(int(whole))
    except ValueError:
        return None

    # The fractional digit counts outs, not tenths.
    outs = frac[:1]
    if outs not in {"0", "1", "2"}:
        return None
    return innings + int(outs) / 3.0


def cumulative_through(
    logs: Iterable[GameLogEntry],
    as_of: str,
) -> dict[str, float] | None:
    """Season-to-date rate stats from every appearance STRICTLY BEFORE `as_of`.

    Strictly before matters: a start on the target date must never inform the
    prediction for that same date.

    Returns None when there is no prior work to summarise — a pitcher's first
    start has no history, and inventing a 0.00 ERA for it would hand the model
    a phantom ace.
    """
    prior: Sequence[GameLogEntry] = [g for g in logs if g.date < as_of]
    if not prior:
        return None

    innings = 0.0
    earned = hits = walks = strikeouts = homers = hbp = 0
    for game in prior:
        ip = parse_innings(game.innings_pitched)
        if ip is None:
            continue
        innings += ip
        earned += game.earned_runs or 0
        hits += game.hits or 0
        walks += game.walks or 0
        strikeouts += game.strikeouts or 0
        homers += game.home_runs or 0
        hbp += game.hit_by_pitch or 0

    if innings <= 0:
        return None

    from src.services.mlb.bullpen import fip as _fip

    return {
        "innings_pitched": round(innings, 2),
        "era": round(9 * earned / innings, 3),
        "whip": round((hits + walks) / innings, 3),
        "k_per_9": round(9 * strikeouts / innings, 3),
        "bb_per_9": round(9 * walks / innings, 3),
        "fip": round(_fip(homers, walks, hbp, strikeouts, innings), 3),
        "games_started": len(prior),
    }


def _next_day(iso_date: str) -> str:
    from datetime import date, timedelta

    y, m, d = (int(p) for p in iso_date.split("-"))
    return (date(y, m, d) + timedelta(days=1)).isoformat()


def build_stat_history(logs: Iterable[GameLogEntry]) -> list[dict]:
    """One cumulative stat row per appearance, dated the day AFTER it.

    Dating each row to the following day encodes when the numbers actually
    became knowable, which is what makes the existing point-in-time lookup
    (`stat_date < game_date`, most recent first) correct without change.
    """
    ordered = sorted(logs, key=lambda g: g.date)
    history: list[dict] = []

    for index, game in enumerate(ordered):
        through = ordered[: index + 1]
        # cumulative_through excludes games on the boundary date, so ask as of
        # the day after this appearance to include it.
        stats = cumulative_through(through, _next_day(game.date))
        if stats is None:
            continue
        history.append({"stat_date": _next_day(game.date), **stats})

    return history
