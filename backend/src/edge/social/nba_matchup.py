# backend/src/edge/social/nba_matchup.py
"""NBA matchup context for Blind Plays and the "favorable" rule.

Windows: every number uses the whole current season (the spec leaves the window to
the builder). Usage and minutes additionally report a last-10 / last-5 view.

Definitions
- Opponent-vs-position: per defending team, per game, sum the market stat
  (players.stat_for) allowed to opposing players of one position group (G/F/C),
  then average over the defender's games (a game where that position produced
  nothing counts as 0). Rank N = allows the most (N is 30 with a full league).
  Players with a NULL position are left out of vs-position but still count in
  team totals.
- Team totals per (bdl_game_id, team) are sums of that team's player rows; the
  opponent's totals are the other team in the same bdl_game_id.
- Possessions = FGA + 0.44*FTA - OREB + TOV. DRtg = 100 * opp pts / opp
  possessions (season sums). Pace = own possessions per game. drtg_rank 1 = best
  defense; pace_rank 1 = fastest.
- Usage = 100*((FGA + 0.44*FTA + TOV)*(team_min/5)) / (min*(team FGA + 0.44*team
  FTA + team TOV)); team_min is the sum of the team's player minutes in that
  game (about 240). Averaged per game over games with minutes > 0.
- Insufficient data: a team with fewer than MIN_TEAM_GAMES games is returned with
  empty vs_pos and ranks of 0, is left out of the ranking, and panel() treats it
  (or any missing (position, market)) as NOT favorable instead of raising.

Only load_context touches the database.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

from src.edge import players

MIN_TEAM_GAMES = 5
POSSESSION_FTA = 0.44


@dataclass(frozen=True)
class TeamRanks:
    team: str
    vs_pos: dict[tuple[str, str], int] = field(default_factory=dict)
    drtg_rank: int = 0
    pace_rank: int = 0


@dataclass(frozen=True)
class PlayerUsage:
    usage_l10: float
    usage_season: float
    minutes_l5: float
    minutes_season: float
    games: int


@dataclass(frozen=True)
class Panel:
    favorable: bool
    opp_vs_pos_rank: int
    drtg_rank: int
    pace_rank: int
    usage_l10: float
    usage_change: float
    minutes_l5: float


def _possessions(t: dict) -> float:
    return t["fga"] + POSSESSION_FTA * t["fta"] - t["oreb"] + t["tov"]


def team_totals(games: list[dict]) -> dict[tuple[int, str], dict]:
    """Sum player rows into per-(bdl_game_id, team) totals."""
    out: dict[tuple[int, str], dict] = {}
    for g in games:
        t = out.setdefault((g["bdl_game_id"], g["team"]),
                           {"pts": 0, "fga": 0, "fta": 0, "oreb": 0, "tov": 0, "minutes": 0.0})
        for k in ("pts", "fga", "fta", "oreb", "tov"):
            t[k] += g[k] or 0
        t["minutes"] += g["minutes"] or 0
    return out


def _rank(values: dict[str, float], descending: bool) -> dict[str, int]:
    """1 = first after sorting; ties broken by team name so the result is deterministic."""
    order = sorted(values, key=lambda k: (-values[k] if descending else values[k], k))
    return {team: i + 1 for i, team in enumerate(order)}


def build_ranks(games: list[dict]) -> dict[str, TeamRanks]:
    totals = team_totals(games)
    team_games: dict[str, set[int]] = defaultdict(set)
    for gid, team in totals:
        team_games[team].add(gid)
    teams = set(team_games)
    ok = {t for t in teams if len(team_games[t]) >= MIN_TEAM_GAMES}

    # opponent-vs-position: allowed[(defender, market, position)][game_id] -> stat sum
    allowed: dict[tuple[str, str, str], dict[int, float]] = defaultdict(lambda: defaultdict(float))
    for g in games:
        pos = g.get("position")
        if not pos:
            continue
        for market in players.STAT_PARTS:
            allowed[(g["opponent"], market, pos)][g["bdl_game_id"]] += players.stat_for(market, g)
    avg: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)  # (pos, market) -> team -> avg allowed
    for (defender, market, pos), per_game in allowed.items():
        if defender in ok:
            avg[(pos, market)][defender] = sum(per_game.values()) / len(team_games[defender])
    vs_rank = {key: _rank(vals, descending=False) for key, vals in avg.items()}  # most allowed -> highest rank

    # DRtg and pace
    opp_pts: dict[str, float] = defaultdict(float)
    opp_poss: dict[str, float] = defaultdict(float)
    own_poss: dict[str, float] = defaultdict(float)
    for (gid, team), t in totals.items():
        own_poss[team] += _possessions(t)
    by_game: dict[int, dict[str, dict]] = defaultdict(dict)
    for (gid, team), t in totals.items():
        by_game[gid][team] = t
    for gid, sides in by_game.items():
        if len(sides) != 2:
            continue
        (a, ta), (b, tb) = sides.items()
        opp_pts[a] += tb["pts"]; opp_poss[a] += _possessions(tb)
        opp_pts[b] += ta["pts"]; opp_poss[b] += _possessions(ta)
    drtg = {t: 100 * opp_pts[t] / opp_poss[t] for t in ok if opp_poss[t] > 0}
    pace = {t: own_poss[t] / len(team_games[t]) for t in ok}
    drtg_rank = _rank(drtg, descending=False)    # lowest rating allowed = 1
    pace_rank = _rank(pace, descending=True)     # most possessions = 1

    result: dict[str, TeamRanks] = {}
    for team in teams:
        if team not in ok or team not in drtg_rank:
            result[team] = TeamRanks(team, {}, 0, 0)
            continue
        vs = {key: ranks[team] for key, ranks in vs_rank.items() if team in ranks}
        result[team] = TeamRanks(team, vs, drtg_rank[team], pace_rank[team])
    return result


def player_usage(rows: list[dict], team_totals: dict[tuple[int, str], dict]) -> PlayerUsage:
    """Season and last-10 usage plus last-5 / season minutes over games with minutes > 0."""
    played = sorted((r for r in rows if (r["minutes"] or 0) > 0), key=lambda r: r["game_date"], reverse=True)
    usages: list[float | None] = []  # newest first, aligned with `played`
    for r in played:
        t = team_totals.get((r["bdl_game_id"], r["team"]))
        if not t:
            usages.append(None)
            continue
        denom = r["minutes"] * (t["fga"] + POSSESSION_FTA * t["fta"] + t["tov"])
        if denom <= 0:
            usages.append(None)
            continue
        num = (r["fga"] + POSSESSION_FTA * r["fta"] + r["tov"]) * (t["minutes"] / 5)
        usages.append(100 * num / denom)

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else 0.0

    mins = [r["minutes"] for r in played]
    return PlayerUsage(usage_l10=mean(usages[:10]), usage_season=mean(usages),
                       minutes_l5=mean(mins[:5]), minutes_season=mean(mins), games=len(played))


def panel(row: dict, position: str, opp: TeamRanks, usage: PlayerUsage) -> Panel:
    vs = opp.vs_pos.get((position, row["market"]))
    insufficient = vs is None or not opp.drtg_rank
    vs_rank = vs or 0
    side = row["side"]
    if insufficient:
        matchup_ok = False
    elif side == "over":
        matchup_ok = vs_rank >= 21 and opp.drtg_rank > 10
    elif side == "under":
        matchup_ok = vs_rank <= 10 and opp.drtg_rank <= 20
    else:
        matchup_ok = False
    steady = (usage.minutes_l5 >= 0.85 * usage.minutes_season
              and usage.usage_l10 >= usage.usage_season - 3)
    return Panel(favorable=matchup_ok and steady, opp_vs_pos_rank=vs_rank, drtg_rank=opp.drtg_rank,
                 pace_rank=opp.pace_rank, usage_l10=usage.usage_l10,
                 usage_change=usage.usage_l10 - usage.usage_season, minutes_l5=usage.minutes_l5)


def load_context(conn, season: int) -> tuple[dict[str, TeamRanks], Callable[[int], PlayerUsage | None]]:
    """The only DB-touching function: ranks for the season plus a bdl_player_id -> usage lookup."""
    with conn.cursor() as cur:
        cur.execute("""SELECT g.bdl_player_id, g.bdl_game_id, g.game_date, g.team, g.opponent, g.minutes,
                              g.pts, g.reb, g.ast, g.fg3m, g.tov, g.fga, g.fta, g.oreb, p.position
                       FROM edge_player_games g
                       LEFT JOIN (SELECT bdl_player_id, MAX(position) AS position FROM edge_players
                                  WHERE bdl_player_id IS NOT NULL GROUP BY bdl_player_id) p
                              ON p.bdl_player_id = g.bdl_player_id
                       WHERE g.season = %s""", (season,))
        cols = [c[0] for c in cur.description]
        games = [dict(zip(cols, r)) for r in cur.fetchall()]
    conn.commit()
    totals = team_totals(games)
    by_player: dict[int, list[dict]] = defaultdict(list)
    for g in games:
        by_player[g["bdl_player_id"]].append(g)

    def usage_for(bdl_player_id: int) -> PlayerUsage | None:
        rows = by_player.get(bdl_player_id)
        return player_usage(rows, totals) if rows else None

    return build_ranks(games), usage_for
