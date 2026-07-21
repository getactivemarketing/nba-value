"""Per-QB career dropback-EPA ratings, empirical-Bayes shrunk to replacement.

Point-in-time: a QB's rating AS OF (season, week) uses only his dropbacks
strictly BEFORE that game (crosses seasons). Feeds the qb_delta MOV feature
(see training_data). Self-contained from nflverse pbp.
"""
import bisect

import pandas as pd

REPLACEMENT_EPA: float = -0.10   # replacement-level QB dropback EPA/play
PRIOR_DROPBACKS: int = 200       # empirical-Bayes prior strength (K)


def shrink(epa_sum: float, dropbacks: float,
           replacement: float = REPLACEMENT_EPA, k: float = PRIOR_DROPBACKS) -> float:
    """(epa_sum + k*replacement) / (dropbacks + k). 0 dropbacks -> replacement."""
    return (epa_sum + k * replacement) / (dropbacks + k)


def qb_game_dropback_epa(pbp: pd.DataFrame) -> pd.DataFrame:
    """Aggregate pbp to per-(passer, season, week, team) dropback count + EPA sum.

    Dropback = nflverse `qb_dropback == 1` (pass attempts + sacks + scrambles);
    EPA from `qb_epa`. Rows with no passer id or NaN epa are dropped.
    """
    db = pbp[(pbp["qb_dropback"] == 1) & pbp["passer_player_id"].notna()].copy()
    db["qb_epa"] = pd.to_numeric(db["qb_epa"], errors="coerce")
    db = db.dropna(subset=["qb_epa"])
    g = db.groupby(["passer_player_id", "season", "week", "posteam"], as_index=False).agg(
        dropbacks=("qb_epa", "size"), epa_sum=("qb_epa", "sum"))
    return g.rename(columns={"posteam": "team"})


def build_qb_timelines(qb_game_epa: pd.DataFrame) -> dict:
    """Per QB: sorted game ordinals + cumulative-INCLUSIVE dropbacks/EPA.

    ord = season*100 + week. Used by rating_as_of for a strictly-before lookup.
    """
    tl: dict = {}
    if qb_game_epa.empty:
        return tl
    df = qb_game_epa.copy()
    df["ord"] = df["season"].astype(int) * 100 + df["week"].astype(int)
    for qb, sub in df.sort_values("ord").groupby("passer_player_id"):
        tl[qb] = {"ord": sub["ord"].tolist(),
                  "cdb": sub["dropbacks"].cumsum().tolist(),
                  "cepa": sub["epa_sum"].cumsum().tolist()}
    return tl


def rating_as_of(timelines: dict, qb, season: int, week: int,
                 replacement: float = REPLACEMENT_EPA, k: float = PRIOR_DROPBACKS) -> float:
    """Shrunk career rating for `qb` using only games STRICTLY before (season, week)."""
    t = timelines.get(qb)
    if not t:
        return replacement
    q = int(season) * 100 + int(week)
    i = bisect.bisect_left(t["ord"], q)   # first game with ord >= q; strictly-before = index i-1
    if i == 0:
        return replacement
    return shrink(t["cepa"][i - 1], t["cdb"][i - 1], replacement, k)


def _form_qb(team_games: pd.DataFrame, season: int, week: int, form_window: int):
    """Passer with the most dropbacks for a team over weeks [w-window, w-1] of the season."""
    win = team_games[(team_games["season"] == season)
                     & (team_games["week"] >= week - form_window)
                     & (team_games["week"] <= week - 1)]
    if win.empty:
        return None
    return win.groupby("passer_player_id")["dropbacks"].sum().idxmax()


def compute_qb_deltas(pbp: pd.DataFrame, games: pd.DataFrame, form_window: int = 8,
                      replacement: float = REPLACEMENT_EPA,
                      k: float = PRIOR_DROPBACKS) -> pd.DataFrame:
    """Per-game qb_delta = (home starter - home form_qb) - (away starter - away form_qb).

    Both the starter and the plurality trailing-form QB are rated through the same
    rating_as_of lookup, so a stable QB (starter == form_qb) yields exactly 0.
    """
    if games.empty:
        return pd.DataFrame({"game_id": [], "qb_delta": []})
    qge = qb_game_dropback_epa(pbp)
    tl = build_qb_timelines(qge)

    def component(qb_id, team, s, w):
        starter = rating_as_of(tl, qb_id, s, w, replacement, k)
        fq = _form_qb(qge[qge["team"] == team], s, w, form_window)
        form = replacement if fq is None else rating_as_of(tl, fq, s, w, replacement, k)
        return starter - form

    rows = []
    for _, g in games.iterrows():
        s, w = int(g["season"]), int(g["week"])
        h = component(g["home_qb_id"], g["home_team"], s, w)
        a = component(g["away_qb_id"], g["away_team"], s, w)
        rows.append({"game_id": g["game_id"], "qb_delta": h - a})
    return pd.DataFrame(rows)
