"""Thin nflverse (nfl_data_py) I/O wrapper returning pandas DataFrames."""
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd

# nflverse gametime is US/Eastern wall-clock; localize with this zone so DST
# (EDT Sep-early Nov, EST Nov-Feb) is handled automatically when converting to UTC.
_ET = ZoneInfo("America/New_York")

from src.services.nfl.constants import (
    is_divisional, is_primetime, normalize_team, primetime_slot,
)


def load_schedules(seasons: list[int]) -> pd.DataFrame:
    import nfl_data_py as nfl
    return nfl.import_schedules(seasons)


def load_pbp(seasons: list[int]) -> pd.DataFrame:
    import nfl_data_py as nfl
    pbp = nfl.import_pbp_data(seasons, downcast=True)
    return pbp[pbp["season_type"] == "REG"].copy()


def _kickoff_utc(row) -> datetime | None:
    """Combine nflverse 'gameday' (YYYY-MM-DD) + 'gametime' (HH:MM ET) into true UTC.

    nflverse gametime is US/Eastern wall-clock. We localize to America/New_York
    (handling EDT/EST DST across the Sep-Feb season) and convert to UTC, so
    kickoff_utc is a real instant. This matters for P4: the scheduler windows
    snapshots off kickoff_utc, and odds matching derives kickoff_date from its
    UTC .date() -- evening primetime kickoffs (TNF/SNF/MNF) cross midnight UTC,
    so their true date is one day past the ET gameday and must match the Odds
    API's UTC commence_date. (Earlier P1 stored ET-wall-clock tagged UTC, which
    silently dropped every primetime game's odds; fixed here.)
    """
    gameday = row.get("gameday")
    gametime = row.get("gametime")
    if not gameday or not gametime:
        return None
    try:
        naive_et = datetime.strptime(f"{gameday} {gametime}", "%Y-%m-%d %H:%M")
    except (ValueError, TypeError):
        return None
    return naive_et.replace(tzinfo=_ET).astimezone(timezone.utc)


def schedule_to_game_rows(sched: pd.DataFrame) -> list[dict]:
    """Pure map of nflverse schedule rows -> NFLGame kwargs dicts."""
    rows: list[dict] = []
    for _, g in sched.iterrows():
        wd = g.get("weekday")
        weekday = "" if pd.isna(wd) else wd
        gt = g.get("gametime")
        gametime = "" if pd.isna(gt) else gt
        home = normalize_team(g["home_team"])
        away = normalize_team(g["away_team"])
        rows.append({
            "game_id": g["game_id"],
            "season": int(g["season"]),
            "week": int(g["week"]),
            "season_type": g.get("game_type", "REG"),
            "home_team": home,
            "away_team": away,
            "kickoff_utc": _kickoff_utc(g),
            "home_score": None if pd.isna(g.get("home_score")) else int(g["home_score"]),
            "away_score": None if pd.isna(g.get("away_score")) else int(g["away_score"]),
            "roof": g.get("roof"),
            "surface": g.get("surface"),
            "neutral_site": bool(g.get("location") == "Neutral"),
            "home_qb": g.get("home_qb_name"),
            "home_qb_id": g.get("home_qb_id"),
            "away_qb": g.get("away_qb_name"),
            "away_qb_id": g.get("away_qb_id"),
            "is_divisional": is_divisional(home, away),
            "is_primetime": is_primetime(weekday, gametime),
            "primetime_slot": primetime_slot(weekday, gametime),
            "status": "final" if not pd.isna(g.get("home_score")) else "scheduled",
        })
    return rows
