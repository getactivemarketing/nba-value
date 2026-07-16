import pandas as pd
from src.services.nfl.nfl_data import schedule_to_game_rows


def test_schedule_to_game_rows_maps_flags_and_keys():
    sched = pd.read_parquet("tests/fixtures/nfl_schedule_sample.parquet")
    rows = schedule_to_game_rows(sched)
    assert len(rows) == len(sched)
    sample = rows[0]
    # required NFLGame kwargs present
    for key in ("game_id", "season", "week", "home_team", "away_team",
                "is_divisional", "is_primetime", "primetime_slot"):
        assert key in sample
    # divisional flag is a bool derived from constants, not copied blindly
    assert isinstance(sample["is_divisional"], bool)


def test_schedule_to_game_rows_divisional_matches_constants():
    from src.services.nfl.constants import is_divisional
    sched = pd.read_parquet("tests/fixtures/nfl_schedule_sample.parquet")
    rows = {r["game_id"]: r for r in schedule_to_game_rows(sched)}
    for _, g in sched.iterrows():
        expected = is_divisional(g["home_team"], g["away_team"])
        assert rows[g["game_id"]]["is_divisional"] is expected


def test_schedule_to_game_rows_normalizes_historical_team_codes():
    # OAK (Raiders pre-2020) vs KC — both AFC West once OAK is normalized to LV.
    sched = pd.DataFrame([{
        "game_id": "2015_01_OAK_KC",
        "season": 2015,
        "week": 1,
        "game_type": "REG",
        "gameday": "2015-09-13",
        "weekday": "Sunday",
        "gametime": "13:00",
        "home_team": "OAK",
        "away_team": "KC",
        "home_score": 21,
        "away_score": 20,
        "roof": "outdoors",
        "surface": "grass",
        "location": "Home",
        "home_qb_name": "Derek Carr",
        "home_qb_id": "00-0031280",
        "away_qb_name": "Alex Smith",
        "away_qb_id": "00-0026143",
        "home_rest": 7,
        "away_rest": 7,
        "div_game": 1,
    }])
    rows = schedule_to_game_rows(sched)
    assert len(rows) == 1
    row = rows[0]
    assert row["home_team"] == "LV"
    assert row["is_divisional"] is True
