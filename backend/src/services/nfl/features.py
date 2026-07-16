"""Pure NFL feature engineering over nflverse play-by-play.

All functions here are deterministic transforms of DataFrames — no DB, no I/O —
so they are cheaply unit-testable and hold the point-in-time invariant explicitly.
"""
import pandas as pd


def team_game_epa(pbp: pd.DataFrame) -> pd.DataFrame:
    """Aggregate play-by-play to one row per (season, week, team).

    Offense metrics come from plays where the team has possession (posteam);
    defense EPA is the EPA the team *allowed* (mean epa of plays where it is
    defteam). Plays with no posteam (kickoffs, etc.) are ignored for offense.
    """
    valid = pbp[pbp["posteam"].notna() & pbp["defteam"].notna()].copy()

    off = valid.groupby(["season", "week", "posteam"]).agg(
        off_epa_play=("epa", "mean"),
        pass_epa=("epa", lambda s: s[valid.loc[s.index, "pass"] == 1].mean()),
        rush_epa=("epa", lambda s: s[valid.loc[s.index, "rush"] == 1].mean()),
        success_rate=("success", "mean"),
        plays=("epa", "size"),
    ).reset_index().rename(columns={"posteam": "team"})

    deff = valid.groupby(["season", "week", "defteam"]).agg(
        def_epa_play=("epa", "mean"),
    ).reset_index().rename(columns={"defteam": "team"})

    out = off.merge(deff, on=["season", "week", "team"], how="outer")
    return out
