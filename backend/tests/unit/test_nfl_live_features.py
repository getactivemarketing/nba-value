from src.services.nfl.live_features import build_live_feature_row
from src.services.nfl.training_data import MOV_FEATURES, TOTALS_FEATURES


def _stats(off, deff, pw, pace):
    return {"off_epa_play": off, "def_epa_play": deff, "pass_epa": 0.1, "rush_epa": 0.0,
            "success_rate": 0.47, "pace": pace, "power_rating": pw}


def test_live_row_matches_model_feature_columns():
    game = {"game_id": "2026_02_CIN_KC", "week": 2, "home_team": "KC", "away_team": "CIN",
            "is_divisional": False, "is_primetime": True}
    ctx = {"home_rest_days": 7, "away_rest_days": 7, "is_dome": False, "wind_mph": 6.0, "temp_f": 70.0}
    row = build_live_feature_row(game, _stats(0.15, -0.05, 0.20, 62.0),
                                 _stats(0.0, 0.05, -0.05, 64.0), ctx,
                                 spread_line=3.0, total_line=47.5)
    # every model feature is present (so scorer won't KeyError)
    for col in set(MOV_FEATURES) | set(TOTALS_FEATURES):
        assert col in row, col
    assert round(row["off_epa_diff"], 3) == 0.15
    assert round(row["power_diff"], 3) == 0.25
    assert row["spread_line"] == 3.0 and row["total_line"] == 47.5
    assert row["is_primetime"] == 1


def test_missing_stats_returns_none():
    game = {"game_id": "g", "week": 2, "home_team": "KC", "away_team": "CIN",
            "is_divisional": False, "is_primetime": False}
    assert build_live_feature_row(game, None, _stats(0, 0, 0, 60), {}, 3.0, 47.5) is None
    assert build_live_feature_row(game, _stats(0, 0, 0, 60), _stats(0, 0, 0, 60), {}, None, 47.5) is None
