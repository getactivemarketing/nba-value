import pandas as pd
from src.services.nfl.features import rolling_team_stats


def _tg(rows):
    return pd.DataFrame(rows)


def test_through_week_never_includes_current_or_future_weeks():
    # KC posts off_epa 1.0 in wk1, 3.0 in wk2, 5.0 in wk3.
    tg = _tg([
        {"season": 2023, "week": 1, "team": "KC", "off_epa_play": 1.0,
         "def_epa_play": 0.0, "pass_epa": 1.0, "rush_epa": 1.0, "success_rate": 0.5, "plays": 60},
        {"season": 2023, "week": 2, "team": "KC", "off_epa_play": 3.0,
         "def_epa_play": 0.0, "pass_epa": 3.0, "rush_epa": 3.0, "success_rate": 0.5, "plays": 60},
        {"season": 2023, "week": 3, "team": "KC", "off_epa_play": 5.0,
         "def_epa_play": 0.0, "pass_epa": 5.0, "rush_epa": 5.0, "success_rate": 0.5, "plays": 60},
    ])
    out = rolling_team_stats(tg).set_index("through_week")
    # through_week=1 sees ONLY week 1 -> 1.0 (no leakage of wk2/wk3)
    assert round(out.loc[1, "off_epa_play"], 3) == 1.0
    # through_week=2 sees weeks 1-2 -> mean(1,3)=2.0
    assert round(out.loc[2, "off_epa_play"], 3) == 2.0
    # through_week=3 sees weeks 1-3 -> mean(1,3,5)=3.0
    assert round(out.loc[3, "off_epa_play"], 3) == 3.0


def test_power_rating_is_off_minus_def():
    tg = _tg([
        {"season": 2023, "week": 1, "team": "SF", "off_epa_play": 0.2,
         "def_epa_play": -0.1, "pass_epa": 0.2, "rush_epa": 0.2, "success_rate": 0.5, "plays": 60},
    ])
    out = rolling_team_stats(tg).set_index("through_week")
    assert round(out.loc[1, "power_rating"], 3) == 0.3


def test_window_limits_trailing_games():
    rows = [
        {"season": 2023, "week": w, "team": "BUF", "off_epa_play": float(w),
         "def_epa_play": 0.0, "pass_epa": float(w), "rush_epa": float(w),
         "success_rate": 0.5, "plays": 60}
        for w in range(1, 11)
    ]
    out = rolling_team_stats(_tg(rows), window=3).set_index("through_week")
    # through_week=10 with window 3 -> mean of weeks 8,9,10 = 9.0
    assert round(out.loc[10, "off_epa_play"], 3) == 9.0
