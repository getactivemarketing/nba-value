"""Per-game defense and player aggregates from nflverse play-by-play.

Pure: no DB and no network except `load_pbp` / `load_rosters`. Teams use
BallDontLie abbreviations, players use `players.normalize_name` keys.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.edge.players import normalize_name
from src.edge.teams import NFLVERSE_TO_BDL

POSITIONS = ("QB", "RB", "WR", "TE")
PBP_COLS = ("season", "week", "season_type", "game_id", "posteam", "defteam", "play_type",
            "two_point_attempt", "yardline_100", "rusher_player_id", "receiver_player_id",
            "passer_player_id", "rushing_yards", "receiving_yards", "passing_yards",
            "rush_touchdown", "pass_touchdown", "td_player_id")
NFLVERSE = "https://github.com/nflverse/nflverse-data/releases/download/"
ROSTER_COLS = ["gsis_id", "full_name", "position", "team", "season"]
RED_ZONE = 20


@dataclass(frozen=True)
class TeamGame:
    """Yards, red-zone opportunities and TDs `team` allowed to one position in one game."""
    season: int
    week: int
    team: str
    pos: str
    yds: float
    rz_opps: int
    tds: int


@dataclass(frozen=True)
class PlayerGame:
    """One row per (season, week, team, player) for QB/RB/WR/TE players with at
    least one carry or target in that game; the team_* and pos_* fields are the
    denominators for his usage shares."""
    season: int
    week: int
    team: str
    player_key: str
    player_name: str
    pos: str
    carries: int
    targets: int
    rz_carries: int
    rz_targets: int
    team_carries: int
    team_targets: int
    team_rz_opps: int
    pos_carries: int
    pos_targets: int


def _bdl(code: str) -> str:
    return NFLVERSE_TO_BDL.get(code, code)


def load_pbp(seasons) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(f"{NFLVERSE}pbp/play_by_play_{s}.parquet", columns=list(PBP_COLS))
                      for s in seasons], ignore_index=True)


def load_rosters(seasons) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(f"{NFLVERSE}rosters/roster_{s}.parquet", columns=ROSTER_COLS)
                      for s in seasons], ignore_index=True)


def aggregate(pbp: pd.DataFrame, rosters: pd.DataFrame) -> tuple[list[TeamGame], list[PlayerGame]]:
    reg = pbp[(pbp["season_type"] == "REG") & pbp["defteam"].notna() & pbp["posteam"].notna()]
    games = reg[["season", "week", "game_id", "defteam"]].drop_duplicates()
    plays = reg[reg["play_type"].isin(["run", "pass"]) & (reg["two_point_attempt"].fillna(0) != 1)].copy()

    ros = rosters.drop_duplicates(["season", "gsis_id"], keep="last").copy()
    ros["pos"] = ros["position"].replace({"FB": "RB"})
    ros = ros[["season", "gsis_id", "full_name", "pos"]]

    for col in ("rushing_yards", "receiving_yards", "passing_yards"):
        plays[col] = plays[col].fillna(0)
    plays["rz"] = plays["yardline_100"].le(RED_ZONE)
    plays["rush_td"] = plays["rush_touchdown"].fillna(0).eq(1)
    plays["rec_td"] = plays["pass_touchdown"].fillna(0).eq(1)
    for who, col in (("rusher", "rusher_player_id"), ("receiver", "receiver_player_id"),
                     ("passer", "passer_player_id")):
        plays = (plays.merge(ros[["season", "gsis_id", "pos"]].rename(
            columns={"gsis_id": col, "pos": f"{who}_pos"}), on=["season", col], how="left"))

    is_run, is_pass = plays["play_type"] == "run", plays["play_type"] == "pass"
    has_rusher = is_run & plays["rusher_player_id"].notna()
    has_recv = is_pass & plays["receiver_player_id"].notna()

    # Defense allowed, per position: yards, red-zone opportunities, TDs.
    keys = ["season", "week", "defteam"]
    qb_pass = plays[is_pass & (plays["passer_pos"] == "QB")]
    ev = []
    for pos in POSITIONS:
        rush = plays[has_rusher & (plays["rusher_pos"] == pos)]
        recv = plays[has_recv & (plays["receiver_pos"] == pos)] if pos != "QB" else plays.iloc[0:0]
        if pos == "QB":
            yds = qb_pass.groupby(keys)["passing_yards"].sum()
        else:
            yds = recv.groupby(keys)["receiving_yards"].sum()
            if pos == "RB":
                yds = yds.add(rush.groupby(keys)["rushing_yards"].sum(), fill_value=0)
        rz = rush.groupby(keys)["rz"].sum().add(recv.groupby(keys)["rz"].sum(), fill_value=0)
        tds = rush.groupby(keys)["rush_td"].sum().add(recv.groupby(keys)["rec_td"].sum(), fill_value=0)
        ev.append((pos, yds, rz, tds))

    tg = []
    for g in games.itertuples(index=False):
        k = (g.season, g.week, g.defteam)
        for pos, y, r, t in ev:
            tg.append(TeamGame(int(g.season), int(g.week), _bdl(g.defteam), pos,
                               float(y.get(k, 0.0)), int(r.get(k, 0)), int(t.get(k, 0))))

    # Player rows.
    gk = ["season", "week", "posteam"]
    carries = plays[has_rusher]
    targets = plays[has_recv]
    team_c = carries.groupby(gk).size()
    team_t = targets.groupby(gk).size()
    team_rz = (carries["rz"].groupby([carries[c] for c in gk]).sum()
               .add(targets["rz"].groupby([targets[c] for c in gk]).sum(), fill_value=0))
    pos_c = carries.groupby(gk + ["rusher_pos"]).size()
    pos_t = targets.groupby(gk + ["receiver_pos"]).size()

    c = carries.groupby(gk + ["rusher_player_id"]).agg(carries=("rz", "size"), rz_carries=("rz", "sum"))
    c.index.names = gk + ["gsis_id"]
    t = targets.groupby(gk + ["receiver_player_id"]).agg(targets=("rz", "size"), rz_targets=("rz", "sum"))
    t.index.names = gk + ["gsis_id"]
    usage = c.join(t, how="outer").fillna(0).reset_index().merge(ros, on=["season", "gsis_id"], how="inner")
    usage = usage[usage["pos"].isin(POSITIONS)]

    pg = []
    for u in usage.itertuples(index=False):
        k = (u.season, u.week, u.posteam)
        pg.append(PlayerGame(
            int(u.season), int(u.week), _bdl(u.posteam), normalize_name(u.full_name), u.full_name, u.pos,
            int(u.carries), int(u.targets), int(u.rz_carries), int(u.rz_targets),
            int(team_c.get(k, 0)), int(team_t.get(k, 0)), int(team_rz.get(k, 0)),
            int(pos_c.get(k + (u.pos,), 0)), int(pos_t.get(k + (u.pos,), 0))))
    return tg, pg


def window_games(rows: list, season: int, week: int, n: int, key) -> list:
    """Last `n` rows matching `key` strictly before (season, week), oldest first.

    Counts games, not weeks (byes leave no row). Fills from season - 1 only when
    fewer than 4 current-season rows exist.
    """
    def pick(s, before):
        return sorted((r for r in rows if r.season == s and key(r) and (before is None or r.week < before)),
                      key=lambda r: r.week)

    cur = pick(season, week)[-n:]
    if len(cur) >= 4 or len(cur) >= n:
        return cur
    prev = pick(season - 1, None)[-(n - len(cur)):]
    return prev + cur
