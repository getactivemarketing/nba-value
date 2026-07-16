import pandas as pd
from src.services.nfl.features import starters_out


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
