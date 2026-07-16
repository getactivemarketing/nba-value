import pandas as pd
from src.services.nfl.features import team_game_epa


def test_team_game_epa_offense_and_defense_split():
    # 2 plays: KC on offense vs DEN; then DEN on offense vs KC.
    pbp = pd.DataFrame([
        {"season": 2023, "week": 1, "posteam": "KC", "defteam": "DEN",
         "epa": 1.0, "success": 1, "pass": 1, "rush": 0, "play_type": "pass"},
        {"season": 2023, "week": 1, "posteam": "DEN", "defteam": "KC",
         "epa": -0.5, "success": 0, "pass": 0, "rush": 1, "play_type": "run"},
    ])
    out = team_game_epa(pbp).set_index("team")
    # KC offense = +1.0 over 1 play; KC defense = DEN's -0.5 EPA allowed
    assert round(out.loc["KC", "off_epa_play"], 3) == 1.0
    assert round(out.loc["KC", "def_epa_play"], 3) == -0.5
    assert round(out.loc["KC", "success_rate"], 3) == 1.0
    # DEN mirror
    assert round(out.loc["DEN", "off_epa_play"], 3) == -0.5
    assert round(out.loc["DEN", "def_epa_play"], 3) == 1.0


def test_team_game_epa_ignores_plays_without_posteam():
    pbp = pd.DataFrame([
        {"season": 2023, "week": 1, "posteam": None, "defteam": None,
         "epa": 5.0, "success": 1, "pass": 0, "rush": 0, "play_type": "kickoff"},
        {"season": 2023, "week": 1, "posteam": "SF", "defteam": "SEA",
         "epa": 0.2, "success": 1, "pass": 1, "rush": 0, "play_type": "pass"},
    ])
    out = team_game_epa(pbp).set_index("team")
    # The posteam=None play must not pollute SF's offense
    assert round(out.loc["SF", "off_epa_play"], 3) == 0.2


def test_pass_and_rush_epa_split():
    pbp = pd.DataFrame([
        {"season": 2023, "week": 1, "posteam": "KC", "defteam": "DEN",
         "epa": 2.0, "success": 1, "pass": 1, "rush": 0, "play_type": "pass"},
        {"season": 2023, "week": 1, "posteam": "KC", "defteam": "DEN",
         "epa": -1.0, "success": 0, "pass": 0, "rush": 1, "play_type": "run"},
    ])
    out = team_game_epa(pbp).set_index("team")
    assert round(out.loc["KC", "pass_epa"], 3) == 2.0
    assert round(out.loc["KC", "rush_epa"], 3) == -1.0
    assert round(out.loc["KC", "off_epa_play"], 3) == 0.5   # mean(2.0, -1.0)


def test_team_game_epa_normalizes_historical_team_codes():
    # OAK is the historical Raiders abbreviation; output team key must be
    # normalized to the current franchise code (LV).
    pbp = pd.DataFrame([
        {"season": 2015, "week": 1, "posteam": "OAK", "defteam": "KC",
         "epa": 1.0, "success": 1, "pass": 1, "rush": 0, "play_type": "pass"},
    ])
    out = team_game_epa(pbp)
    assert "LV" in out["team"].tolist()
    assert "OAK" not in out["team"].tolist()


def test_handles_non_unique_index():
    pbp = pd.DataFrame([
        {"season": 2023, "week": 1, "posteam": "KC", "defteam": "DEN",
         "epa": 2.0, "success": 1, "pass": 1, "rush": 0, "play_type": "pass"},
        {"season": 2023, "week": 1, "posteam": "KC", "defteam": "DEN",
         "epa": -1.0, "success": 0, "pass": 0, "rush": 1, "play_type": "run"},
    ], index=[0, 0])
    out = team_game_epa(pbp).set_index("team")
    assert round(out.loc["KC", "pass_epa"], 3) == 2.0
    assert round(out.loc["KC", "rush_epa"], 3) == -1.0
