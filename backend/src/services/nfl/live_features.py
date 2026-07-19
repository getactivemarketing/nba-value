"""Assemble the live feature row `scorer.score_game` needs.

Pure function: no I/O, no DB, no side effects. `home_stats`/`away_stats` are
the nfl_team_stats rows at through_week = game.week - 1; `spread_line`/
`total_line` are the CURRENT market lines (from nfl_markets).

Reuses `training_data._feature_diffs` for the diff/pace_sum/off_epa_sum
computation so live scoring can never drift from what the model was trained
on (same _DIFF_MAP, same arithmetic).
"""
from src.services.nfl.training_data import _feature_diffs


def build_live_feature_row(game, home_stats, away_stats, context,
                            spread_line, total_line) -> dict | None:
    if home_stats is None or away_stats is None:
        return None  # prior-week team stats not available yet -> can't score
    if spread_line is None or total_line is None:
        return None  # no current market line -> can't score

    context = context or {}
    home_rest = context.get("home_rest_days")
    away_rest = context.get("away_rest_days")
    wind_mph = context.get("wind_mph")
    temp_f = context.get("temp_f")

    row = {
        "is_divisional": int(bool(game["is_divisional"])),
        "is_primetime": int(bool(game["is_primetime"])),
        "spread_line": float(spread_line),
        "total_line": float(total_line),
        "rest_diff": (0 if home_rest is None else int(home_rest))
                     - (0 if away_rest is None else int(away_rest)),
        "is_dome": int(bool(context.get("is_dome"))),
        "wind_mph": None if wind_mph is None else float(wind_mph),
        "temp_f": None if temp_f is None else float(temp_f),
    }
    row.update(_feature_diffs(home_stats, away_stats))
    return row
