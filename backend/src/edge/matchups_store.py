"""Live storage for NFL matchup aggregates (team and player weeks)."""
from __future__ import annotations

from dataclasses import astuple, fields
from datetime import datetime

from psycopg2.extras import execute_values

from src.edge.matchups import PlayerGame, TeamGame, drop_ambiguous

_TEAM = [f.name for f in fields(TeamGame)]
_PLAYER = [f.name for f in fields(PlayerGame)]
_TEAM_KEY = ["season", "week", "team", "pos"]
_PLAYER_KEY = ["season", "week", "team", "player_key"]


def _sql(table: str, cols: list[str], key: list[str]) -> str:
    sets = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols if c not in key)
    return (f"INSERT INTO {table} ({', '.join(cols)}, loaded_at) VALUES %s "
            f"ON CONFLICT ({', '.join(key)}) DO UPDATE SET {sets}, loaded_at = EXCLUDED.loaded_at")


def upsert(conn, team_games: list[TeamGame], player_games: list[PlayerGame], now: datetime) -> int:
    # Same-team namesakes would repeat the unique key in one batch (CardinalityViolation)
    # and fail the whole refresh; their earlier stored rows would let a guess through.
    seasons = sorted({p.season for p in player_games})
    player_games, dropped = drop_ambiguous(player_games)
    with conn.cursor() as cur:
        if dropped:
            for team, key in sorted(dropped):
                cur.execute("DELETE FROM edge_nfl_player_weeks WHERE team = %s AND player_key = %s "
                            "AND season = ANY(%s)", (team, key, seasons))
        if team_games:
            execute_values(cur, _sql("edge_nfl_team_weeks", _TEAM, _TEAM_KEY),
                           [astuple(t) + (now,) for t in team_games])
        if player_games:
            execute_values(cur, _sql("edge_nfl_player_weeks", _PLAYER, _PLAYER_KEY),
                           [astuple(p) + (now,) for p in player_games])
    conn.commit()
    return len(team_games) + len(player_games)


def load(conn, seasons: list[int]) -> tuple[list[TeamGame], list[PlayerGame]]:
    with conn.cursor() as cur:
        cur.execute(f"SELECT {', '.join(_TEAM)} FROM edge_nfl_team_weeks WHERE season = ANY(%s) "
                    "ORDER BY season, week, team, pos", (list(seasons),))
        teams = [TeamGame(*r) for r in cur.fetchall()]
        cur.execute(f"SELECT {', '.join(_PLAYER)} FROM edge_nfl_player_weeks WHERE season = ANY(%s) "
                    "ORDER BY season, week, team, player_key", (list(seasons),))
        players = [PlayerGame(*r) for r in cur.fetchall()]
    conn.commit()
    return teams, players
