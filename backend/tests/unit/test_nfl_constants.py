from src.services.nfl.constants import (
    NFL_DIVISIONS, is_divisional, is_primetime, normalize_team, primetime_slot,
)


def test_all_32_teams_have_a_division():
    assert len(NFL_DIVISIONS) == 32
    assert NFL_DIVISIONS["KC"] == "AFC West"
    assert NFL_DIVISIONS["PHI"] == "NFC East"


def test_divisional_game_detection():
    assert is_divisional("KC", "DEN") is True      # both AFC West
    assert is_divisional("KC", "PHI") is False


def test_primetime_slots():
    # Thursday night
    assert primetime_slot("Thursday", "20:15") == "TNF"
    # Sunday night (>= 19:00)
    assert primetime_slot("Sunday", "20:20") == "SNF"
    # Sunday afternoon is NOT primetime
    assert primetime_slot("Sunday", "13:00") is None
    # Monday night
    assert primetime_slot("Monday", "20:15") == "MNF"


def test_is_primetime_matches_slot():
    assert is_primetime("Sunday", "20:20") is True
    assert is_primetime("Sunday", "13:00") is False


def test_normalize_team_maps_historical_abbreviations():
    assert normalize_team("OAK") == "LV"
    assert normalize_team("SD") == "LAC"
    assert normalize_team("STL") == "LA"


def test_normalize_team_passes_through_unknown_and_current_codes():
    assert normalize_team("KC") == "KC"
    assert normalize_team("XYZ") == "XYZ"


def test_normalize_team_result_is_divisional():
    # SD (pre-2017 Chargers) and KC are both AFC West once normalized.
    assert is_divisional(normalize_team("SD"), normalize_team("KC")) is True
