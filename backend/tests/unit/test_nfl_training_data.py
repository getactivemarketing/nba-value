import pandas as pd
from src.services.nfl.training_data import build_feature_frame


def _frames():
    # Two teams, season 2023. Team stats exist through_week 1 (entering wk2).
    games = pd.DataFrame([
        {"game_id": "G2", "season": 2023, "week": 2, "home_team": "KC", "away_team": "DET",
         "home_score": 24, "away_score": 17, "is_divisional": False, "is_primetime": True},
        {"game_id": "G1", "season": 2023, "week": 1, "home_team": "KC", "away_team": "DET",
         "home_score": 20, "away_score": 21, "is_divisional": False, "is_primetime": True},
    ])
    team_stats = pd.DataFrame([
        {"team": "KC", "season": 2023, "through_week": 1, "off_epa_play": 0.1, "def_epa_play": -0.05,
         "pass_epa": 0.2, "rush_epa": 0.0, "success_rate": 0.48, "pace": 62.0, "power_rating": 0.15},
        {"team": "DET", "season": 2023, "through_week": 1, "off_epa_play": 0.0, "def_epa_play": 0.05,
         "pass_epa": 0.1, "rush_epa": -0.1, "success_rate": 0.45, "pace": 64.0, "power_rating": -0.05},
    ])
    context = pd.DataFrame([
        {"game_id": "G2", "home_rest_days": 7, "away_rest_days": 7, "is_dome": False,
         "wind_mph": 5.0, "temp_f": 60.0},
        {"game_id": "G1", "home_rest_days": 7, "away_rest_days": 7, "is_dome": False,
         "wind_mph": None, "temp_f": None},
    ])
    lines = pd.DataFrame([
        {"game_id": "G2", "spread_line": 3.0, "total_line": 47.0,
         "home_moneyline": -160, "away_moneyline": 140},
        {"game_id": "G1", "spread_line": 4.0, "total_line": 53.0,
         "home_moneyline": -198, "away_moneyline": 164},
    ])
    return games, team_stats, context, lines


def test_week1_excluded_and_features_point_in_time():
    frame = build_feature_frame(*_frames())
    # Only week-2 game G2 is modelable (week 1 has no through_week=0 stats)
    assert list(frame["game_id"]) == ["G2"]
    row = frame.iloc[0]
    # diffs = home(KC) - away(DET)
    assert round(row["off_epa_diff"], 3) == 0.1        # 0.1 - 0.0
    assert round(row["power_diff"], 3) == 0.2          # 0.15 - (-0.05)
    assert round(row["pace_sum"], 1) == 126.0          # 62 + 64
    assert row["is_primetime"] == 1
    # targets + lines
    assert row["margin"] == 7                           # 24 - 17
    assert row["total"] == 41                            # 24 + 17
    assert row["spread_line"] == 3.0
    assert row["total_line"] == 47.0


def test_missing_team_stats_row_drops_game():
    games, team_stats, context, lines = _frames()
    # Remove DET's stats -> G2 can't be built
    team_stats = team_stats[team_stats["team"] != "DET"]
    frame = build_feature_frame(games, team_stats, context, lines)
    assert len(frame) == 0


def test_null_spread_line_drops_game():
    games, team_stats, context, lines = _frames()
    lines = lines.copy()
    lines.loc[lines["game_id"] == "G2", "spread_line"] = float("nan")
    frame = build_feature_frame(games, team_stats, context, lines)
    # G2 is the only modelable game; a null spread_line must drop it, not
    # silently include a fabricated 0.0 anchor after downstream fillna(0).
    assert len(frame) == 0


def test_null_total_line_drops_game():
    games, team_stats, context, lines = _frames()
    lines = lines.copy()
    lines.loc[lines["game_id"] == "G2", "total_line"] = float("nan")
    frame = build_feature_frame(games, team_stats, context, lines)
    assert len(frame) == 0


def test_build_feature_frame_adds_qb_delta_column():
    import pandas as pd
    from src.services.nfl.training_data import build_feature_frame, MOV_FEATURES
    # qb_delta must be a declared MOV feature
    assert MOV_FEATURES[-1] == "qb_delta"
    # minimal 1-game frame (reuse the module's existing fixture builders if present;
    # otherwise construct inline as below)
    games = pd.DataFrame([{
        "game_id": "g", "season": 2022, "week": 2, "home_team": "KC", "away_team": "CIN",
        "home_score": 27, "away_score": 20, "is_divisional": False, "is_primetime": True,
    }])
    ts = pd.DataFrame([
        {"team": "KC", "season": 2022, "through_week": 1, "off_epa_play": 0.1, "def_epa_play": -0.05,
         "pass_epa": 0.1, "rush_epa": 0.0, "success_rate": 0.47, "pace": 62.0, "power_rating": 0.2},
        {"team": "CIN", "season": 2022, "through_week": 1, "off_epa_play": 0.0, "def_epa_play": 0.05,
         "pass_epa": 0.1, "rush_epa": 0.0, "success_rate": 0.47, "pace": 64.0, "power_rating": -0.05},
    ])
    ctx = pd.DataFrame([{"game_id": "g", "home_rest_days": 7, "away_rest_days": 7,
                         "is_dome": False, "wind_mph": 6.0, "temp_f": 70.0}])
    lines = pd.DataFrame([{"game_id": "g", "spread_line": 3.0, "total_line": 44.0,
                           "home_moneyline": -160, "away_moneyline": 140}])
    qbd = pd.DataFrame([{"game_id": "g", "qb_delta": 0.12}])
    out = build_feature_frame(games, ts, ctx, lines, qbd)
    assert "qb_delta" in out.columns and abs(out.iloc[0]["qb_delta"] - 0.12) < 1e-9
    # default (no qb_deltas) -> column present, 0.0 (keeps back-compat for callers that omit it)
    out0 = build_feature_frame(games, ts, ctx, lines)
    assert out0.iloc[0]["qb_delta"] == 0.0
