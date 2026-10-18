from datetime import datetime, timezone

import pandas as pd

from src.edge import nfl_injuries as inj

NOW = datetime(2026, 10, 18, 12, 0, tzinfo=timezone.utc)


def row(status, date, team="KC", first="Travis", last="Kelce", pos="TE"):
    return {"player": {"first_name": first, "last_name": last, "position_abbreviation": pos,
                       "team": {"abbreviation": team} if team else None},
            "status": status, "date": date}


def test_ir_types_count_as_out():
    for s in ("Out", "IR", "IR-R", "PUP-R", "NFI-R", "Reserve-Sus"):
        assert inj.normalize_status(s) == "out"
    assert inj.normalize_status("Doubtful") == "doubtful"
    assert inj.normalize_status("Questionable") == "questionable"


def test_unknown_status_dropped():
    assert inj.normalize_status("Probable") is None
    assert inj.live_rows([row("Probable", "2026-10-15T00:00:00Z")], NOW) == []


def test_questionable_older_than_14_days_ignored_but_ir_kept():
    rows = inj.live_rows([row("Questionable", "2026-09-27T00:00:00Z", last="A"),
                          row("IR", "2026-09-01", last="B"),
                          row("Questionable", None, last="C"),
                          row("Questionable", "2026-10-15T08:00:00Z", last="D"),
                          row("Out", "2026-10-15", last="E", team=None)], NOW)
    assert {r["player_norm"]: r["status"] for r in rows} == {"travis b": "out", "travis d": "questionable"}


def test_live_rows_keeps_most_severe_duplicate():
    rows = inj.live_rows([row("Questionable", "2026-10-15"), row("Out", "2026-10-16")], NOW)
    assert len(rows) == 1 and rows[0]["status"] == "out" and rows[0]["team"] == "KC"


def test_history_index_maps_nflverse_team_codes_to_bdl():
    frame = pd.DataFrame([
        {"season": 2025, "game_type": "REG", "team": "LA", "week": 3, "full_name": "Puka Nacua", "report_status": "Out"},
        {"season": 2025, "game_type": "REG", "team": "WAS", "week": 3, "full_name": "Terry McLaurin", "report_status": "Questionable"},
        {"season": 2025, "game_type": "POST", "team": "KC", "week": 19, "full_name": "X Y", "report_status": "Out"},
        {"season": 2025, "game_type": "REG", "team": "KC", "week": 3, "full_name": "Z W", "report_status": None},
        {"season": 2025, "game_type": "REG", "team": "KC", "week": 3, "full_name": float("nan"), "report_status": "Out"},
    ])
    idx = inj.history_index(frame)
    assert idx == {(2025, 3): {"LAR": {"puka nacua": "out"}, "WSH": {"terry mclaurin": "questionable"}}}
