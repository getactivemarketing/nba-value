"""Per-game defense and player aggregates from nflverse play-by-play.

Pure: no DB and no network except `load_pbp` / `load_rosters`. Teams use
BallDontLie abbreviations, players use `players.normalize_name` keys.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import pandas as pd

from src.edge.config import TD_MARKET
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

    # Null ids would left-merge onto each other; a nameless player can't get a key.
    ros = rosters.dropna(subset=["gsis_id", "full_name"]).drop_duplicates(["season", "gsis_id"], keep="last").copy()
    ros["pos"] = ros["position"].replace({"FB": "RB"})
    ros = ros[["season", "gsis_id", "full_name", "pos"]]

    for col in ("rushing_yards", "receiving_yards", "passing_yards"):
        plays[col] = plays[col].fillna(0)
    plays["rz"] = plays["yardline_100"].le(RED_ZONE)
    # TDs use the rush_touchdown/pass_touchdown flags; td_player_id is loaded but deliberately unused.
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
    qb_pass = plays[is_pass & plays["passer_player_id"].notna() & (plays["passer_pos"] == "QB")]
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


# ---- Tags ------------------------------------------------------------------

TAG_KEYS = ("def_soft", "def_tough", "def_rz", "def_td", "usage_up", "rz_role", "teammate_out", "questionable")
POLARITY: dict[str, int] = {k: (-1 if k in ("def_tough", "questionable") else 1) for k in TAG_KEYS}
POSITION_MARKETS = {
    "QB": ("player_pass_yds", "player_pass_completions", "player_pass_attempts"),
    "RB": ("player_rush_yds", "player_rush_attempts", "player_rush_reception_yds"),
    "WR": ("player_reception_yds", "player_receptions"),
    "TE": ("player_reception_yds", "player_receptions"),
}
TD_POSITIONS = ("RB", "WR", "TE")
DEF_WINDOW, MIN_GAMES, TOP = 8, 4, 8
USAGE_RECENT, USAGE_MIN_EARLIER, USAGE_UP = 3, 2, 0.05
RZ_WINDOW, RZ_ROLE = 4, 0.25
MATE_WINDOW, MATE_SHARE, MATE_STATUSES = 4, 0.30, ("out", "doubtful")
EPS = 1e-9  # so a share exactly at a cutoff fires despite float error
STAT_WORD = {"yds": "yds", "rz": "red-zone looks", "td": "TDs"}


@dataclass(frozen=True)
class Tag:
    key: str
    direction: str  # "tailwind" | "headwind", relative to the row's side
    detail: str


def applies(key: str, pos: str, market: str) -> bool:
    own = POSITION_MARKETS.get(pos, ())
    if key in ("def_soft", "def_tough"):
        return market in own
    if key == "def_rz":
        return pos in TD_POSITIONS and (market == TD_MARKET or market == own[0])
    if key in ("def_td", "rz_role"):
        return pos in TD_POSITIONS and market == TD_MARKET
    if key in ("usage_up", "teammate_out"):
        return pos in TD_POSITIONS and (market in own or market == TD_MARKET)
    if key == "questionable":
        return market.startswith("player_")
    return False


_ALL_MARKETS = {m for ms in POSITION_MARKETS.values() for m in ms} | {TD_MARKET}
MARKETS_FOR: dict[str, frozenset[str]] = {
    k: frozenset(m for m in _ALL_MARKETS for p in POSITIONS if applies(k, p, m)) for k in TAG_KEYS}


@dataclass
class TagContext:
    def_rank: dict[tuple[str, str], dict[str, tuple[float, int]]]
    players: dict[str, list[tuple[str, str]]]
    player_rows: dict[tuple[str, str], list[PlayerGame]]
    injuries: dict[str, dict[str, str]]
    season: int = 0
    week: int = 0
    n_ranked: dict[str, int] = field(default_factory=dict)  # pos -> number of ranked teams


def ordinal(n: int) -> str:
    suffix = "th" if 11 <= n % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def build_context(team_games, player_games, injuries, season: int, week: int) -> TagContext:
    """Everything a tag needs for a game in (season, week), from rows strictly before it."""
    cut = (season, week)
    tgs = [r for r in team_games if (r.season, r.week) < cut]
    pgs = sorted((r for r in player_games if (r.season, r.week) < cut), key=lambda r: (r.season, r.week))

    by_team_pos = defaultdict(list)
    for r in tgs:
        by_team_pos[(r.team, r.pos)].append(r)
    per_game: dict[str, dict[tuple[str, str], dict[str, float]]] = defaultdict(dict)
    for (team, pos), rows in by_team_pos.items():
        win = window_games(rows, season, week, DEF_WINDOW, lambda r: True)
        if len(win) >= MIN_GAMES:
            n = len(win)
            per_game[pos][(team, pos)] = {"yds": sum(r.yds for r in win) / n,
                                          "rz": sum(r.rz_opps for r in win) / n,
                                          "td": sum(r.tds for r in win) / n}
    def_rank: dict = {}
    n_ranked: dict[str, int] = {}
    for pos, teams in per_game.items():
        n_ranked[pos] = len(teams)
        for stat in STAT_WORD:
            order = sorted(teams, key=lambda tp: (-teams[tp][stat], tp[0]))
            for rank, tp in enumerate(order, start=1):
                def_rank.setdefault(tp, {})[stat] = (teams[tp][stat], rank)

    player_rows: dict = defaultdict(list)
    by_key: dict = defaultdict(list)
    for r in pgs:
        player_rows[(r.team, r.player_key)].append(r)
        by_key[r.player_key].append(r)
    players = {key: _identities(rows) for key, rows in by_key.items()}
    return TagContext(def_rank, players, dict(player_rows), injuries, season, week, n_ranked)


def _identities(rows: list[PlayerGame]) -> list[tuple[str, str]]:
    """[(team, pos)] from the key's latest row. A traded player has one row per game, so
    two rows in one game of his latest season mean two players share the key: then
    every row of the latest such game is an entry, so the key stays ambiguous."""
    latest = rows[-1]
    games = defaultdict(list)
    for r in rows:
        if r.season == latest.season:
            games[r.week].append(r)
    shared = [w for w, rs in games.items() if len(rs) > 1]
    if not shared:
        return [(latest.team, latest.pos)]
    return [(r.team, r.pos) for r in games[max(shared)]]


def _share(rows, num, den) -> float | None:
    d = sum(den(r) for r in rows)
    return sum(num(r) for r in rows) / d if d else None


def tags_for(ctx: TagContext, player: str, event_teams: tuple[str, str], market: str, side: str) -> list[Tag]:
    key = normalize_name(player)
    ids = [tp for tp in ctx.players.get(key, []) if tp[0] in event_teams]
    if len(ids) != 1:
        return []  # unknown, traded off the event's teams, or an ambiguous name key
    team, pos = ids[0]
    opp = event_teams[1] if event_teams[0] == team else event_teams[0]
    sign = 1 if side == "over" else -1
    out: list[Tag] = []

    def add(k: str, detail: str) -> None:
        if applies(k, pos, market):
            out.append(Tag(k, "tailwind" if POLARITY[k] * sign > 0 else "headwind", detail))

    ranks = ctx.def_rank.get((opp, pos))
    if ranks:
        n = ctx.n_ranked.get(pos, 0)
        value, rank = ranks["yds"]
        if rank <= TOP:
            add("def_soft", f"{opp} allows {value:.0f} {pos} yds per game ({ordinal(rank)} most)")
        if rank > TOP and rank >= n - (TOP - 1):  # never both soft and tough when few are ranked
            # "Fewest" counts from the bottom: 33 - rank with 32 ranked teams.
            add("def_tough", f"{opp} allows {value:.0f} {pos} yds per game ({ordinal(n + 1 - rank)} fewest)")
        for k, stat in (("def_rz", "rz"), ("def_td", "td")):
            value, rank = ranks[stat]
            if rank <= TOP:
                add(k, f"{opp} allows {value:.1f} {pos} {STAT_WORD[stat]} per game ({ordinal(rank)} most)")  # per-game counts are small

    rows = ctx.player_rows.get((team, key), [])
    if pos in TD_POSITIONS:
        num = (lambda r: r.carries) if pos == "RB" else (lambda r: r.targets)
        den = (lambda r: r.team_carries) if pos == "RB" else (lambda r: r.team_targets)
        cur = [r for r in rows if r.season == ctx.season]
        if len(cur) >= USAGE_RECENT + USAGE_MIN_EARLIER:
            recent = _share(cur[-USAGE_RECENT:], num, den)
            earlier = _share(cur[:-USAGE_RECENT], num, den)
            if recent is not None and earlier is not None and recent - earlier >= USAGE_UP - EPS:
                word = "Carry share" if pos == "RB" else "Target share"
                add("usage_up", f"{word} {recent:.0%} last 3 vs {earlier:.0%} before")

        rz = _share(window_games(rows, ctx.season, ctx.week, RZ_WINDOW, lambda r: True),
                    lambda r: r.rz_carries + r.rz_targets, lambda r: r.team_rz_opps)
        if rz is not None and rz >= RZ_ROLE - EPS:
            add("rz_role", f"{rz:.0%} of team red-zone looks, last 4")

        # One tag per qualifying injured teammate.
        pos_den = (lambda r: r.pos_carries) if pos == "RB" else (lambda r: r.pos_targets)
        unit = "carries" if pos == "RB" else "targets"
        for mate, status in sorted(ctx.injuries.get(team, {}).items()):
            if mate == key or status not in MATE_STATUSES:
                continue
            if [tp for tp in ctx.players.get(mate, []) if tp[0] == team] != [(team, pos)]:
                continue  # not (only) a same-position player on this team
            mrows = ctx.player_rows.get((team, mate), [])
            share = _share(window_games(mrows, ctx.season, ctx.week, MATE_WINDOW, lambda r: True), num, pos_den)
            if share is not None and share >= MATE_SHARE - EPS:
                add("teammate_out", f"{mrows[-1].player_name} ({status.capitalize()}) — {share:.0%} of {pos} {unit}, last 4")

    if ctx.injuries.get(team, {}).get(key) == "questionable":
        add("questionable", "Listed Questionable")
    return out
