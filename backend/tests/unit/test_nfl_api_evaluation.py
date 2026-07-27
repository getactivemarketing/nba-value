from contextlib import asynccontextmanager
from datetime import date
from unittest.mock import AsyncMock, MagicMock
from fastapi.testclient import TestClient
from src.api import nfl as nfl_api
from src.config import settings
from src.main import app
from src.models import NFLPredictionSnapshot


def _snap(gid, bb_result, bb_profit, vs, gday, sp_result=None, sp_profit=None):
    return NFLPredictionSnapshot(
        game_id=gid, home_team="KC", away_team="CIN", game_date=gday,
        best_bet_type="total", best_bet_value_score=vs,
        best_bet_result=bb_result, best_bet_profit=bb_profit,
        best_spread_result=sp_result, best_spread_profit=sp_profit,
        best_ml_result=None, best_ml_profit=None,
    )


def _scalars(items):
    res = MagicMock(); res.scalars.return_value.all.return_value = list(items); return res


def _patch(monkeypatch, rows):
    session = MagicMock(); session.execute = AsyncMock(return_value=_scalars(rows))
    @asynccontextmanager
    async def _factory():
        yield session
    monkeypatch.setattr(nfl_api, "async_session", _factory)


def test_evaluation_summary_aggregates_best_bet_and_shadow(monkeypatch):
    rows = [
        _snap("g1", "win", 90.9, 55.0, date(2026, 9, 13), sp_result="loss", sp_profit=-100.0),
        _snap("g2", "loss", -100.0, 48.0, date(2026, 9, 13), sp_result="win", sp_profit=90.9),
        _snap("g3", "push", 0.0, 44.0, date(2026, 9, 14)),
    ]
    _patch(monkeypatch, rows)
    body = TestClient(app).get(f"{settings.api_v1_prefix}/nfl/evaluation/summary").json()
    assert body["total_predictions"] == 3 and body["wins"] == 1 and body["losses"] == 1 and body["pushes"] == 1
    assert round(body["total_profit"], 1) == -9.1           # 90.9 - 100 + 0
    assert body["by_market"]["best_bet"]["wins"] == 1
    assert body["by_market"]["spread"]["wins"] == 1 and body["by_market"]["spread"]["losses"] == 1


def test_evaluation_daily_groups_by_date(monkeypatch):
    rows = [
        _snap("g1", "win", 90.9, 55.0, date(2026, 9, 13)),
        _snap("g2", "loss", -100.0, 48.0, date(2026, 9, 13)),
        _snap("g3", "win", 90.9, 60.0, date(2026, 9, 14)),
    ]
    _patch(monkeypatch, rows)
    body = TestClient(app).get(f"{settings.api_v1_prefix}/nfl/evaluation/daily?days=30").json()
    assert len(body) == 2
    d0 = next(d for d in body if d["date"] == "2026-09-13")
    assert d0["predictions"] == 2 and d0["wins"] == 1 and d0["losses"] == 1


def test_evaluation_endpoints_in_openapi():
    schema = TestClient(app).get("/openapi.json").json()
    assert f"{settings.api_v1_prefix}/nfl/evaluation/summary" in schema["paths"]
    assert f"{settings.api_v1_prefix}/nfl/evaluation/daily" in schema["paths"]
