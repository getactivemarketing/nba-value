# backend/src/edge/tagging.py
"""Live NFL matchup tags for the board, player pages and the opportunity log (spec §7-8).

Tags are context, never price: they never touch fair_prob, ev or is_ev. They only
reorder rows, and only by the signed CLV lift of tags the backtest proved or condemned.
No tags never means no board: every failure here degrades to `tags: []`.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Callable

from src.edge import matchups, matchups_store, nfl_injuries, store
from src.edge.config import TD_MARKET
from src.edge.teams import NFL_ABBR_BY_NAME
from src.utils.season import current_nfl_season

log = logging.getLogger(__name__)

TAG_LABEL = {
    "def_soft": "Soft matchup", "def_tough": "Tough matchup", "def_rz": "Red-zone-friendly D",
    "def_td": "TD-friendly D", "usage_up": "Usage rising", "rz_role": "Red-zone role",
    "teammate_out": "Teammate out", "questionable": "Questionable",
}
STALE_AFTER = timedelta(days=8)
ALL_STORED_WEEKS = 99  # every stored row of the current season counts

Tagger = Callable[[str, tuple[str, str], str, str], list[dict]]


def _statuses(conn) -> dict[tuple[str, str], tuple[str, float]]:
    with conn.cursor() as cur:
        cur.execute("SELECT tag, direction, verdict, clv_lift FROM edge_tag_status")
        rows = cur.fetchall()
    conn.commit()
    return {(t, d): (v, float(lift or 0.0)) for t, d, v, lift in rows}


def live_tagger(conn, now: datetime) -> Tagger | None:
    try:
        ok = store.get_state(conn).get("nfl:matchups_ok_at")
        if not ok or now - datetime.fromisoformat(ok) > STALE_AFTER:
            return None
        season = current_nfl_season(now.date())
        team_games, player_games = matchups_store.load(conn, [season - 1, season])
        ctx = matchups.build_context(team_games, player_games, nfl_injuries.live_index(conn),
                                     season, ALL_STORED_WEEKS)
        status = _statuses(conn)
    except Exception as e:  # noqa: BLE001 - the board must serve without tags
        log.error("edge live tags unavailable: %s", e)
        return None

    def tagger(player: str, event_teams: tuple[str, str], market: str, side: str) -> list[dict]:
        try:
            tags = matchups.tags_for(ctx, player, event_teams, market, side)
        except Exception as e:  # noqa: BLE001
            log.error("edge tags failed for %s: %s", player, e)
            return []
        out = []
        for t in tags:
            verdict, lift = status.get((t.key, t.direction), ("neutral", 0.0))
            out.append({"key": t.key, "label": TAG_LABEL[t.key], "direction": t.direction,
                        "verdict": verdict, "lift": lift, "detail": t.detail})
        return out

    return tagger


def sort_key(row: dict) -> float:
    return row["ev"] + sum(t["lift"] for t in row["tags"] if t["verdict"] in ("proven", "negative"))


def event_teams(home_team: str, away_team: str) -> tuple[str, str] | None:
    home, away = NFL_ABBR_BY_NAME.get(home_team), NFL_ABBR_BY_NAME.get(away_team)
    return (home, away) if home and away else None


def row_tags(tagger: Tagger | None, player: str, market: str, side: str,
             home_team: str, away_team: str) -> list[dict]:
    """Player markets only; game lines and unmappable teams get none."""
    if tagger is None or not player or not (market.startswith("player_") or market == TD_MARKET):
        return []
    teams = event_teams(home_team, away_team)
    return tagger(player, teams, market, side) if teams else []
