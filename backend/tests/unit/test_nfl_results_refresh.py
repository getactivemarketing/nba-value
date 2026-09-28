# tests/unit/test_nfl_results_refresh.py
"""Scores must reach nfl_games while the weekend is still the weekend.

Found 2026-09-28: Week 3 produced 15 snapshots and graded NOTHING. grade_finals
selects on `nfl_games.home_score IS NOT NULL`, and the only writer of that
column was refresh_schedule inside the Tuesday weekly_refresh -- so Sunday
results sat ungraded for ~40 hours while the hourly grading job logged
"graded=0" 40 times.
"""
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

from src.services.nfl import season_update
from src.tasks import nfl_scheduler as sched


def _sched_df():
    return pd.DataFrame([
        {"game_id": "2026_03_LV_NO", "game_type": "REG", "season": 2026, "week": 3,
         "home_team": "NO", "away_team": "LV", "gameday": "2026-09-27", "gametime": "16:25",
         "away_score": 35.0, "home_score": 27.0},
        {"game_id": "2026_04_SEA_ARI", "game_type": "REG", "season": 2026, "week": 4,
         "home_team": "ARI", "away_team": "SEA", "gameday": "2026-10-04", "gametime": "13:00",
         "away_score": None, "home_score": None},
    ])


async def test_refresh_results_upserts_scores_without_the_heavy_loads(monkeypatch):
    """The weekly refresh also pulls play-by-play, injuries and depth charts
    (~500k rows) to recompute team stats. That is far too heavy to run hourly,
    and results need none of it: the schedule file carries the scores."""
    monkeypatch.setattr(season_update, "load_schedules", lambda seasons: _sched_df())

    def _boom(*a, **k):
        raise AssertionError("results refresh must not pull pbp/injury/depth data")

    monkeypatch.setattr(season_update, "load_pbp", _boom)
    monkeypatch.setattr(season_update, "_load_injury_depth", _boom)

    upsert = AsyncMock(return_value=2)
    monkeypatch.setattr(season_update, "upsert_games", upsert)

    session = MagicMock()
    n = await season_update.refresh_results(session, 2026)

    assert n == 2
    rows = {r["game_id"]: r for r in upsert.await_args[0][1]}
    assert rows["2026_03_LV_NO"]["home_score"] == 27
    assert rows["2026_03_LV_NO"]["status"] == "final"
    assert rows["2026_04_SEA_ARI"]["home_score"] is None
    assert rows["2026_04_SEA_ARI"]["status"] == "scheduled"


async def test_run_grade_is_preceded_by_a_results_refresh(monkeypatch):
    """Grading an hour before the score lands just burns an hour. The results
    pull is registered on the same hourly cadence as grading."""
    call_log = []
    names = _enabled_names(monkeypatch, call_log)

    assert "run_refresh_results" in names
    assert "run_grade" in names


def _enabled_names(monkeypatch, call_log):
    monkeypatch.setattr(sched.settings, "nfl_scheduler_enabled", True)
    monkeypatch.setattr(sched.settings, "nfl_capture_enabled", True)
    monkeypatch.setattr(sched.time, "sleep", MagicMock())
    monkeypatch.setattr(sched, "_init_engine", MagicMock())
    for name in ("run_refresh_odds", "run_shadow_capture", "run_weekly_refresh",
                 "run_snapshot", "run_grade", "run_refresh_results"):
        stub = MagicMock(side_effect=lambda n=name: call_log.append(n))
        stub.__name__ = name
        monkeypatch.setattr(sched, name, stub)

    registered = []

    class _Rec:
        def every(self, n=1):
            class _E:
                def __getattr__(self, unit):
                    def do(fn, *a, **k):
                        registered.append(fn)
                        return fn
                    return type("J", (), {"do": staticmethod(do)})()
            return _E()

        def run_pending(self):
            raise SystemExit

    monkeypatch.setattr(sched.schedule, "Scheduler", _Rec)
    with pytest.raises(SystemExit):
        sched.start_scheduler()
    return {getattr(f, "__name__", str(f)) for f in registered}


async def test_boot_grades_after_pulling_results(monkeypatch):
    """A restart must not park graded-ready games for another hour."""
    call_log = []
    _enabled_names(monkeypatch, call_log)

    assert call_log.index("run_refresh_results") < call_log.index("run_grade")
