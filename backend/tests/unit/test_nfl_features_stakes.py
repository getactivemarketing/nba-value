import pandas as pd
from src.services.nfl.features import playoff_stakes, starters_out


def test_starters_out_counts_only_out_starters():
    injuries = pd.DataFrame([
        {"team": "KC", "gsis_id": "p1", "report_status": "Out"},
        {"team": "KC", "gsis_id": "p2", "report_status": "Questionable"},
        {"team": "KC", "gsis_id": "p3", "report_status": "Out"},
    ])
    depth = pd.DataFrame([
        {"team": "KC", "gsis_id": "p1", "depth_team": 1},   # starter, Out -> counts
        {"team": "KC", "gsis_id": "p2", "depth_team": 1},   # starter but Questionable
        {"team": "KC", "gsis_id": "p3", "depth_team": 2},   # Out but backup -> no count
    ])
    assert starters_out(injuries, depth, "KC") == 1


def test_starters_out_missing_data_returns_zero():
    empty = pd.DataFrame(columns=["team", "gsis_id", "report_status"])
    depth = pd.DataFrame(columns=["team", "gsis_id", "depth_team"])
    assert starters_out(empty, depth, "KC") == 0


def test_starters_out_injuries_present_but_depth_empty_returns_zero():
    injuries = pd.DataFrame([
        {"team": "KC", "gsis_id": "p1", "report_status": "Out"},
    ])
    empty_depth = pd.DataFrame(columns=["team", "gsis_id", "depth_team"])
    assert starters_out(injuries, empty_depth, "KC") == 0


def test_starters_out_depth_present_but_injuries_empty_returns_zero():
    empty_injuries = pd.DataFrame(columns=["team", "gsis_id", "report_status"])
    depth = pd.DataFrame([
        {"team": "KC", "gsis_id": "p1", "depth_team": 1},
    ])
    assert starters_out(empty_injuries, depth, "KC") == 0


def test_playoff_stakes_all_alive_before_week_15():
    standings = pd.DataFrame([{"team": t, "wins": w} for t, w in
                              [("KC", 10), ("DEN", 3), ("BUF", 8)]])
    out = playoff_stakes(standings, 2023, 10)
    assert set(out.values()) == {"alive"}
    assert out["KC"] == "alive"


def test_playoff_stakes_ranks_by_wins_week_15_plus():
    # 9 teams so thirds are clean: top 3 clinched, bottom 3 eliminated, mid 3 alive
    standings = pd.DataFrame([{"team": f"T{i}", "wins": 20 - i} for i in range(9)])
    out = playoff_stakes(standings, 2023, 16)
    # T0..T2 highest wins -> clinched; T6..T8 lowest -> eliminated
    assert out["T0"] == "clinched"
    assert out["T8"] == "eliminated"
    assert out["T4"] == "alive"
