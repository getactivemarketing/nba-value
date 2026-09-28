# tests/unit/test_nfl_evaluation_tracked.py
"""The Results page in a tracking season.

Week 3 graded 15 games and the Results page still read 0-0: both evaluation
endpoints selected on `best_bet_result IS NOT NULL`, which stays null for
every snapshot while no market is in best_bet. Grading fills
best_total_result / best_spread_result / best_ml_result; nothing read them.
"""
from contextlib import asynccontextmanager
from datetime import date
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

from src.api import nfl as nfl_api
from src.config import settings
from src.main import app
from src.models import NFLPredictionSnapshot


def _tracked_snap(gid, total_result, spread_result, ml_result=None, gday=date(2026, 9, 27)):
    """A graded tracking snapshot: every best_bet_* column is null."""
    return NFLPredictionSnapshot(
        game_id=gid, home_team="GB", away_team="ATL", game_date=gday,
        actual_total=48, actual_margin=-21, home_score=13, away_score=34,
        best_total_direction="under", best_total_line=43.5,
        best_total_result=total_result,
        best_total_profit=None if total_result is None else (90.9 if total_result == "win" else -100.0),
        best_spread_team="away", best_spread_line=5.5,
        best_spread_result=spread_result,
        best_spread_profit=None if spread_result is None else (90.9 if spread_result == "win" else -100.0),
        best_ml_team="away" if ml_result else None,
        best_ml_result=ml_result, best_ml_profit=None,
        best_bet_type=None, best_bet_result=None, best_bet_profit=None,
    )


def _patch(monkeypatch, rows):
    res = MagicMock()
    res.scalars.return_value.all.return_value = list(rows)
    session = MagicMock()
    session.execute = AsyncMock(return_value=res)

    @asynccontextmanager
    async def _factory():
        yield session

    monkeypatch.setattr(nfl_api, "async_session", _factory)
    return session


def test_summary_counts_tracked_markets_when_best_bet_is_null(monkeypatch):
    for flag in ("nfl_totals_in_best_bet", "nfl_spread_in_best_bet", "nfl_ml_in_best_bet"):
        monkeypatch.setattr(settings, flag, False)
    _patch(monkeypatch, [
        _tracked_snap("g1", "win", "win"),
        _tracked_snap("g2", "loss", "win"),
        _tracked_snap("g3", None, "loss"),      # no total lean on this game
    ])

    body = TestClient(app).get(f"{settings.api_v1_prefix}/nfl/evaluation/summary").json()

    assert body["tracking_only"] is True
    assert body["graded"] == 3
    assert body["by_market"]["total"] == {
        "wins": 1, "losses": 1, "pushes": 0, "profit": -9.1, "win_rate": 0.5, "count": 2,
    }
    assert body["by_market"]["spread"]["wins"] == 2
    assert body["by_market"]["spread"]["losses"] == 1
    assert body["by_market"]["spread"]["win_rate"] == 0.667


def test_summary_selects_on_graded_not_on_best_bet_result(monkeypatch):
    """The bug itself: the WHERE clause. With a mocked session the rows come
    back regardless, so assert the SQL rather than the tally."""
    session = _patch(monkeypatch, [])

    TestClient(app).get(f"{settings.api_v1_prefix}/nfl/evaluation/summary")

    sql = str(session.execute.await_args[0][0].compile(compile_kwargs={"literal_binds": True}))
    where = sql.split("WHERE", 1)[1]   # SELECT lists every column, so check the clause
    assert "actual_total IS NOT NULL" in where
    assert "best_bet_result" not in where


def test_tracked_endpoint_lists_graded_leans_per_game(monkeypatch):
    _patch(monkeypatch, [_tracked_snap("2026_03_ATL_GB", "win", "loss", ml_result="win")])

    body = TestClient(app).get(f"{settings.api_v1_prefix}/nfl/evaluation/tracked").json()

    assert body["total"] == 1
    g = body["games"][0]
    assert g["game_id"] == "2026_03_ATL_GB"
    assert g["away_score"] == 34 and g["home_score"] == 13 and g["actual_total"] == 48
    assert g["total_direction"] == "under" and g["total_line"] == 43.5 and g["total_result"] == "win"
    # spread side stored as home/away is quoted from the team's own perspective
    assert g["spread_team"] == "ATL" and g["spread_line"] == 5.5 and g["spread_result"] == "loss"
    assert g["ml_team"] == "ATL" and g["ml_result"] == "win"
