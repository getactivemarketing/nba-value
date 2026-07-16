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
    valid = pbp[pbp["posteam"].notna() & pbp["defteam"].notna()].copy().reset_index(drop=True)

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


_EPA_COLS = ["off_epa_play", "def_epa_play", "pass_epa", "rush_epa", "success_rate"]


def rolling_team_stats(team_game: pd.DataFrame, window: int = 8) -> pd.DataFrame:
    """Point-in-time trailing team form.

    For each (season, team) and each played week w, emit a row keyed
    through_week=w that aggregates ONLY that team's games with week <= w
    (trailing `window` games). This row represents the team's form used to
    predict week w+1 — it must never reference week > w.
    """
    out_rows: list[dict] = []
    for (season, team), grp in team_game.groupby(["season", "team"]):
        grp = grp.sort_values("week")
        weeks = grp["week"].tolist()
        for w in weeks:
            hist = grp[grp["week"] <= w].tail(window)   # <= w, never > w
            row = {"season": int(season), "team": team, "through_week": int(w),
                   "pace": float(hist["plays"].mean())}
            for col in _EPA_COLS:
                row[col] = float(hist[col].mean())
            row["power_rating"] = row["off_epa_play"] - row["def_epa_play"]
            out_rows.append(row)
    return pd.DataFrame(out_rows)
