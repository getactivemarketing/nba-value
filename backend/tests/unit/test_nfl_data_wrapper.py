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
