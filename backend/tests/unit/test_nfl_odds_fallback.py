# tests/unit/test_nfl_odds_fallback.py
"""refresh_odds falls back to nflverse when The Odds API cannot be reached."""
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.services.nfl import season_update
from src.tasks import nfl_scheduler as sched


async def test_refresh_odds_falls_back_when_the_odds_api_raises(monkeypatch):
    """The 401 that cost Weeks 1-2 raised out of odds_to_markets every time.

    A raise must not end the refresh: it rolls the transaction back (the failed
    call may have left it dirty) and collects the nflverse board instead.
    """
    session = MagicMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()

    monkeypatch.setattr(season_update, "odds_to_markets",
                        AsyncMock(side_effect=RuntimeError("401 Unauthorized")))
    fallback = AsyncMock(return_value=48)
    monkeypatch.setattr(season_update, "nflverse_to_markets", fallback)

    result = await sched.refresh_odds(session)

    assert result["markets"] == 48 and result["source"] == "nflverse"
    fallback.assert_awaited_once()
    session.rollback.assert_awaited_once()
    session.commit.assert_awaited_once()


async def test_refresh_odds_falls_back_when_the_odds_api_returns_nothing(monkeypatch):
    """A dead key is not the only way to collect nothing: an empty board
    reaches the same place, and the season must not stall on it."""
    session = MagicMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()

    monkeypatch.setattr(season_update, "odds_to_markets", AsyncMock(return_value=0))
    monkeypatch.setattr(season_update, "nflverse_to_markets", AsyncMock(return_value=12))

    result = await sched.refresh_odds(session)

    assert result["markets"] == 12 and result["source"] == "nflverse"
    session.rollback.assert_not_awaited()


async def test_refresh_odds_prefers_the_odds_api_when_it_works(monkeypatch):
    """The paid feed is the better instrument: real per-book prices and real
    movement. The fallback must never pre-empt it or double-write on top."""
    session = MagicMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()

    monkeypatch.setattr(season_update, "odds_to_markets", AsyncMock(return_value=225))
    fallback = AsyncMock(return_value=48)
    monkeypatch.setattr(season_update, "nflverse_to_markets", fallback)

    result = await sched.refresh_odds(session)

    assert result["markets"] == 225 and result["source"] == "odds_api"
    fallback.assert_not_awaited()


async def test_nflverse_to_markets_writes_only_upcoming_games(monkeypatch):
    now = datetime(2026, 9, 24, 18, 0, tzinfo=timezone.utc)
    import pandas as pd
    monkeypatch.setattr(season_update, "load_schedules", lambda seasons: pd.DataFrame([
        {"game_id": "2026_03_ATL_GB", "game_type": "REG", "home_team": "GB", "away_team": "ATL",
         "spread_line": 6.0, "total_line": 43.5, "home_moneyline": -290.0, "away_moneyline": 235.0},
        {"game_id": "2026_03_CAR_CLE", "game_type": "REG", "home_team": "CLE", "away_team": "CAR",
         "spread_line": -2.5, "total_line": 42.5, "home_moneyline": 124.0, "away_moneyline": -148.0},
    ]))

    games = MagicMock()
    games.all.return_value = [
        MagicMock(game_id="2026_03_ATL_GB", kickoff_utc=now + timedelta(hours=6)),
        MagicMock(game_id="2026_03_CAR_CLE", kickoff_utc=now - timedelta(hours=2)),  # played
    ]
    snaps = MagicMock()
    snaps.scalars.return_value.all.return_value = []

    session = MagicMock()
    session.execute = AsyncMock(side_effect=[games, snaps])

    written = await season_update.nflverse_to_markets(session, 2026, now=now)

    added = [c[0][0] for c in session.add.call_args_list]
    market_games = {m.game_id for m in added if type(m).__name__ == "NFLMarket"}
    assert market_games == {"2026_03_ATL_GB"}
    assert written == 3  # spread, total, moneyline for the one upcoming game

    quotes = [m for m in added if type(m).__name__ == "NFLOddsSnapshot"]
    assert len(quotes) == 1
    assert quotes[0].spread_line == -6.0 and quotes[0].book == "nflverse"


async def test_nflverse_to_markets_skips_history_when_the_line_has_not_moved(monkeypatch):
    """nfl_odds_snapshots is append-only line movement, not a heartbeat: a
    consensus line that has not moved since the last 4-hourly run adds nothing."""
    now = datetime(2026, 9, 24, 18, 0, tzinfo=timezone.utc)
    import pandas as pd
    monkeypatch.setattr(season_update, "load_schedules", lambda seasons: pd.DataFrame([
        {"game_id": "2026_03_ATL_GB", "game_type": "REG", "home_team": "GB", "away_team": "ATL",
         "spread_line": 6.0, "total_line": 43.5, "home_moneyline": -290.0, "away_moneyline": 235.0},
    ]))

    games = MagicMock()
    games.all.return_value = [MagicMock(game_id="2026_03_ATL_GB", kickoff_utc=now + timedelta(hours=6))]

    prior = MagicMock()
    prior.game_id = "2026_03_ATL_GB"
    prior.ml_home, prior.ml_away = 1.3448, 3.35
    prior.spread_line, prior.spread_home_odds, prior.spread_away_odds = -6.0, 1.9091, 1.9091
    prior.total_line, prior.over_odds, prior.under_odds = 43.5, 1.9091, 1.9091
    snaps = MagicMock()
    snaps.scalars.return_value.all.return_value = [prior]

    session = MagicMock()
    session.execute = AsyncMock(side_effect=[games, snaps])

    await season_update.nflverse_to_markets(session, 2026, now=now)

    added = [c[0][0] for c in session.add.call_args_list]
    assert [m for m in added if type(m).__name__ == "NFLOddsSnapshot"] == []
    # markets are still refreshed: they are the current board, not history
    assert len([m for m in added if type(m).__name__ == "NFLMarket"]) == 3
