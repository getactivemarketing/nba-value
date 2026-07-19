# tests/unit/test_nfl_api.py
"""P4 Task 5: api/nfl.py picks/games endpoints.

Runs against a MOCKED async_session (no real DB, no network). The router uses
`async with async_session() as session:` so we patch `src.api.nfl.async_session`
with a factory returning a fake async-context-manager session whose `execute`
returns seeded snapshot/game rows.
"""
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

from src.api import nfl as nfl_api
from src.config import settings
from src.main import app
from src.models import NFLGame, NFLPredictionSnapshot


def _snap(game_id, value_score, kickoff, best_bet_type="total", best_bet_team=None):
    return NFLPredictionSnapshot(
        game_id=game_id, home_team="KC", away_team="CIN", kickoff_utc=kickoff,
        snapshot_time=kickoff, game_date=kickoff.date(),
        predicted_margin=2.5, predicted_total=48.0,
        best_bet_type=best_bet_type, best_bet_team=best_bet_team,
        best_bet_line=47.5, best_bet_odds=1.91,
        best_bet_value_score=value_score, best_bet_edge=0.07,
    )


def _scalars(items):
    res = MagicMock()
    res.scalars.return_value.all.return_value = list(items)
    return res


def _patch_session(monkeypatch, execute_side_effect):
    """Patch src.api.nfl.async_session -> factory yielding a fake session."""
    session = MagicMock()
    session.execute = AsyncMock(side_effect=execute_side_effect)

    @asynccontextmanager
    async def _factory():
        yield session

    monkeypatch.setattr(nfl_api, "async_session", _factory)
    return session


def test_picks_returns_seeded_pick_and_filters_below_threshold(monkeypatch):
    kickoff = datetime(2026, 9, 13, 17, 0, tzinfo=timezone.utc)
    # DB query already applies the >= min_value_score filter, so the seeded
    # result reflects what the default-40 query would return: only the 55.
    _patch_session(monkeypatch, [_scalars([_snap("2026_02_CIN_KC", 55.0, kickoff)])])

    client = TestClient(app)
    resp = client.get(f"{settings.api_v1_prefix}/nfl/picks")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 1
    assert body["min_value_score"] == 40
    assert body["picks"][0]["game_id"] == "2026_02_CIN_KC"
    assert body["picks"][0]["best_bet_type"] == "total"
    assert body["picks"][0]["best_bet_value_score"] == 55.0


def test_picks_min_value_score_query_is_passed_through(monkeypatch):
    kickoff = datetime(2026, 9, 13, 17, 0, tzinfo=timezone.utc)
    session = _patch_session(monkeypatch, [_scalars([])])

    client = TestClient(app)
    resp = client.get(f"{settings.api_v1_prefix}/nfl/picks", params={"min_value_score": 70})
    assert resp.status_code == 200
    assert resp.json()["min_value_score"] == 70
    # the query was executed once (threshold applied in SQL)
    session.execute.assert_awaited_once()


def test_picks_drops_rows_without_a_best_bet(monkeypatch):
    kickoff = datetime(2026, 9, 13, 17, 0, tzinfo=timezone.utc)
    good = _snap("2026_02_CIN_KC", 55.0, kickoff)
    no_bet = _snap("2026_02_NYJ_NE", 55.0, kickoff, best_bet_type=None)
    _patch_session(monkeypatch, [_scalars([good, no_bet])])

    client = TestClient(app)
    body = client.get(f"{settings.api_v1_prefix}/nfl/picks").json()
    assert [p["game_id"] for p in body["picks"]] == ["2026_02_CIN_KC"]


def test_games_returns_upcoming_with_snapshot_join(monkeypatch):
    kickoff = datetime(2026, 9, 13, 17, 0, tzinfo=timezone.utc)
    game = NFLGame(
        game_id="2026_02_CIN_KC", season=2026, week=2, home_team="KC", away_team="CIN",
        kickoff_utc=kickoff, status="scheduled", is_divisional=False, is_primetime=True,
    )
    snap = _snap("2026_02_CIN_KC", 55.0, kickoff)
    # first execute -> games, second execute -> snapshots for those game_ids
    _patch_session(monkeypatch, [_scalars([game]), _scalars([snap])])

    client = TestClient(app)
    body = client.get(f"{settings.api_v1_prefix}/nfl/games").json()
    assert body["total"] == 1
    g = body["games"][0]
    assert g["game_id"] == "2026_02_CIN_KC" and g["is_primetime"] is True
    assert g["best_bet_type"] == "total" and g["best_bet_value_score"] == 55.0


def test_picks_endpoint_is_in_openapi_schema():
    client = TestClient(app)
    schema = client.get("/openapi.json").json()
    assert f"{settings.api_v1_prefix}/nfl/picks" in schema["paths"]
    assert f"{settings.api_v1_prefix}/nfl/games" in schema["paths"]
