"""nflverse data for the backtest: schedules, weekly player stats and snap counts.

Weekly stats come from the `stats_player` release file for every season: the
nfl_data_py weekly loader 404s for 2025 (checked 2026-10-16), and one source
for all three seasons keeps grading uniform.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from src.edge.players import normalize_name
from src.edge.teams import NFLVERSE_TO_BDL

ET = ZoneInfo("America/New_York")
STATS_URL = ("https://github.com/nflverse/nflverse-data/releases/download/"
             "stats_player/stats_player_week_{season}.parquet")
STAT_MAP = {"attempts": "pass_att", "completions": "pass_cmp", "passing_yards": "pass_yds",
            "passing_tds": "pass_tds", "passing_interceptions": "pass_int", "carries": "rush_att",
            "rushing_yards": "rush_yds", "receptions": "receptions", "receiving_yards": "rec_yds",
            "rushing_tds": "rush_tds", "receiving_tds": "rec_tds", "special_teams_tds": "ret_tds"}


@dataclass(frozen=True)
class Game:
    game_id: str
    season: int
    week: int
    home: str              # BallDontLie abbreviation, as teams.NFL_ABBR_BY_NAME
    away: str
    kickoff: datetime      # UTC
    home_score: int | None
    away_score: int | None


def _team(abbr: str) -> str:
    return NFLVERSE_TO_BDL.get(abbr, abbr)


def _int(v) -> int:
    return 0 if pd.isna(v) else int(v)


def _score(v) -> int | None:
    return None if pd.isna(v) else int(v)


def schedule_games(frame) -> list[Game]:
    out = []
    for r in frame.itertuples(index=False):
        if r.game_type != "REG":
            continue
        kickoff = (datetime.strptime(f"{r.gameday} {r.gametime}", "%Y-%m-%d %H:%M")
                   .replace(tzinfo=ET).astimezone(timezone.utc))
        out.append(Game(r.game_id, int(r.season), int(r.week), _team(r.home_team), _team(r.away_team),
                        kickoff, _score(r.home_score), _score(r.away_score)))
    return out


def stats_index(frame) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in frame.to_dict("records"):
        if r["season_type"] != "REG" or not r["player_display_name"]:
            continue  # nameless team-level rows (~22 a season) grade nothing
        row = {"name": normalize_name(r["player_display_name"]), "team": _team(r["team"])}
        row.update({col: _int(r[src]) for src, col in STAT_MAP.items()})
        out.setdefault(r["game_id"], []).append(row)
    return out


def snaps_index(frame) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for r in frame.itertuples(index=False):
        if r.game_type != "REG":
            continue
        if _int(r.offense_snaps) + _int(r.defense_snaps) + _int(r.st_snaps) > 0:
            out.setdefault(r.game_id, set()).add(normalize_name(r.player))
    return out


def load_schedule(seasons) -> list[Game]:
    import nfl_data_py
    return schedule_games(nfl_data_py.import_schedules(list(seasons)))


def load_stats(seasons) -> dict[str, list[dict]]:
    return stats_index(pd.concat([pd.read_parquet(STATS_URL.format(season=s)) for s in seasons]))


def load_snaps(seasons) -> dict[str, set[str]]:
    import nfl_data_py
    return snaps_index(nfl_data_py.import_snap_counts(list(seasons)))
