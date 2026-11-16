# backend/src/edge/social/nfl_matchup.py
"""NFL "favorable matchup" rule for Blind Plays.

A bet is favorable iff it carries the proven `def_soft` tailwind tag and no tag at
all points the other way (headwind). Tags come from `tagging.row_tags`, stored on
the opportunity row as `tags` (list of {"key","direction","detail"}).
"""
from __future__ import annotations

from dataclasses import dataclass

from src.edge import matchups, tagging


def favorable(row: dict) -> bool:
    tags = row.get("tags") or []
    if any(t.get("direction") == "headwind" for t in tags):
        return False
    return any(t.get("key") == "def_soft" and t.get("direction") == "tailwind" for t in tags)


@dataclass(frozen=True)
class NflPanel:
    favorable: bool
    opp_rank: int | None  # opponent's rank allowing yards to the player's position (1 = most)
    share_l4: float | None  # carry share (RB) or target share (others) of the team, last 4 games
    snap_share: float | None  # not available: the matchup tables hold no snap counts
    tags: list[str]


def panel(row: dict, ctx: matchups.TagContext) -> NflPanel:
    """Context numbers for one row. Uses the context's current week (live use), not the row's game week."""
    tags = [t["key"] for t in row.get("tags") or []]
    opp_rank = share_l4 = None
    teams = tagging.event_teams(row.get("home_team", ""), row.get("away_team", ""))
    who = matchups.resolve_player(ctx, row.get("player", ""), teams) if teams else None
    if who:
        key, team, pos, opp = who
        ranks = ctx.def_rank.get((opp, pos))
        if ranks:
            opp_rank = ranks["yds"][1]
        num = (lambda r: r.carries) if pos == "RB" else (lambda r: r.targets)
        den = (lambda r: r.team_carries) if pos == "RB" else (lambda r: r.team_targets)
        rows = ctx.player_rows.get((team, key), [])
        win = matchups.window_games(rows, ctx.season, ctx.week, matchups.RZ_WINDOW, lambda r: True)
        share_l4 = matchups._share(win, num, den)
    return NflPanel(favorable(row), opp_rank, share_l4, None, tags)
