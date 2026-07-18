# tests/unit/test_nfl_scheduler_tasks.py
"""P4 Task 4: nfl_scheduler weekly orchestration. Ships DISABLED.

All tests here run against a MOCKED async session (no real DB, no network).
`_score_one` is the pure per-game helper factored out of `snapshot_due_games`
so the happy-path scoring logic can be unit-tested directly without needing
to mock a full SQLAlchemy session chain.
"""
from datetime import date, datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from src.models import NFLGame, NFLGameContext, NFLMarket, NFLPredictionSnapshot, NFLTeamStats
from src.services.nfl.training_data import MOV_FEATURES, TOTALS_FEATURES
from src.tasks import nfl_scheduler as sched


# ---------------------------------------------------------------------------
# Fake model bundle (mirrors tests/unit/test_nfl_scorer.py's fixture style)
# ---------------------------------------------------------------------------
class _Booster:
    def __init__(self, val):
        self.val = val

    best_iteration = 1

    def predict(self, X, num_iteration=None):
        return [self.val] * len(X)


def _bundle(pred, cols, std, calibrator=None):
    return {"model": _Booster(pred), "feature_cols": cols, "resid_std": std,
            "calibrator": calibrator}


def _mov_bundle():
    return _bundle(3.0, MOV_FEATURES, 13.0)


def _totals_bundle():
    return _bundle(45.0, TOTALS_FEATURES, 13.7, calibrator=None)


def _game_dict(**overrides):
    row = {
        "game_id": "2026_02_CIN_KC", "home_team": "KC", "away_team": "CIN",
        "kickoff_utc": datetime(2026, 9, 13, 17, 0, tzinfo=timezone.utc),
        "game_date": date(2026, 9, 13), "week": 2,
        "is_divisional": False, "is_primetime": True,
    }
    row.update(overrides)
    return row


def _stats(off, deff, pw, pace):
    return {"off_epa_play": off, "def_epa_play": deff, "pass_epa": 0.1, "rush_epa": 0.0,
            "success_rate": 0.47, "pace": pace, "power_rating": pw}


def _context():
    return {"home_rest_days": 7, "away_rest_days": 7, "wind_mph": 6.0, "temp_f": 70.0,
            "is_dome": False}


def _markets():
    return [
        {"market_type": "spread", "line": 3.0, "home_odds": 1.91, "away_odds": 1.91},
        {"market_type": "total", "line": 47.5, "over_odds": 1.91, "under_odds": 1.91},
    ]


# ---------------------------------------------------------------------------
# 1. _current_season (pure)
# ---------------------------------------------------------------------------
def test_current_season_regular_season_start():
    assert sched._current_season(date(2026, 9, 13)) == 2026


def test_current_season_january_points_at_prior_calendar_year():
    assert sched._current_season(date(2027, 1, 10)) == 2026


def test_current_season_offseason_points_at_upcoming_season():
    assert sched._current_season(date(2026, 7, 18)) == 2026


# ---------------------------------------------------------------------------
# 2. _score_one (pure per-game helper)
# ---------------------------------------------------------------------------
def test_score_one_builds_a_snapshot_dict():
    snap = sched._score_one(
        _game_dict(), _stats(0.15, -0.05, 0.20, 62.0), _stats(0.0, 0.05, -0.05, 64.0),
        _context(), _markets(), _mov_bundle(), _totals_bundle(),
    )
    assert snap is not None
    assert snap["game_id"] == "2026_02_CIN_KC"
    assert snap["home_team"] == "KC" and snap["away_team"] == "CIN"
    assert snap["predicted_margin"] == 3.0
    assert snap["predicted_total"] == 45.0
    # every key build_snapshot produces must line up with the model's columns
    # so the caller can filter-and-construct NFLPredictionSnapshot(**snap).
    cols = {c.name for c in NFLPredictionSnapshot.__table__.columns}
    assert set(snap.keys()) <= cols


def test_score_one_returns_none_when_prior_week_stats_missing():
    snap = sched._score_one(
        _game_dict(), None, _stats(0.0, 0.05, -0.05, 64.0),
        _context(), _markets(), _mov_bundle(), _totals_bundle(),
    )
    assert snap is None


def test_score_one_returns_none_when_no_market_line():
    snap = sched._score_one(
        _game_dict(), _stats(0.15, -0.05, 0.20, 62.0), _stats(0.0, 0.05, -0.05, 64.0),
        _context(), [], _mov_bundle(), _totals_bundle(),
    )
    assert snap is None


# ---------------------------------------------------------------------------
# session-result helpers for mocking AsyncSession.execute(...)
# ---------------------------------------------------------------------------
def _scalars_result(items):
    res = MagicMock()
    res.scalars.return_value.all.return_value = list(items)
    return res


def _scalar_result(obj):
    res = MagicMock()
    res.scalar_one_or_none.return_value = obj
    return res


def _all_result(items):
    """For multi-column select(...).all() results (e.g. grade_finals' join)."""
    res = MagicMock()
    res.all.return_value = list(items)
    return res


# ---------------------------------------------------------------------------
# 3. snapshot_due_games
# ---------------------------------------------------------------------------
async def test_snapshot_due_games_no_op_when_nothing_due(monkeypatch):
    session = MagicMock()
    session.execute = AsyncMock(return_value=_scalars_result([]))  # no due games
    session.commit = AsyncMock()

    score_mock = MagicMock(side_effect=AssertionError("scoring should not be attempted"))
    monkeypatch.setattr(sched, "score_game", score_mock)
    load_bundle_mock = MagicMock(side_effect=AssertionError("bundles should not be loaded"))
    monkeypatch.setattr(sched.model_training, "load_bundle", load_bundle_mock)

    # Bundles passed in explicitly (as fakes) so we can also prove they're
    # never touched when there's nothing to score.
    result = await sched.snapshot_due_games(
        session, minutes_before=90, mov_bundle=object(), totals_bundle=object(),
    )

    assert result == {"snapshotted": 0}
    score_mock.assert_not_called()
    load_bundle_mock.assert_not_called()
    session.add.assert_not_called()


async def test_snapshot_due_games_one_due_game_inserts_snapshot():
    game = NFLGame(
        game_id="2026_02_CIN_KC", season=2026, week=2, home_team="KC", away_team="CIN",
        kickoff_utc=datetime(2026, 9, 13, 17, 0, tzinfo=timezone.utc), status="scheduled",
        is_divisional=False, is_primetime=True,
    )
    home_stats = NFLTeamStats(team="KC", season=2026, through_week=1, off_epa_play=0.15,
                               def_epa_play=-0.05, pass_epa=0.1, rush_epa=0.0,
                               success_rate=0.47, pace=62.0, power_rating=0.20)
    away_stats = NFLTeamStats(team="CIN", season=2026, through_week=1, off_epa_play=0.0,
                               def_epa_play=0.05, pass_epa=0.1, rush_epa=0.0,
                               success_rate=0.47, pace=64.0, power_rating=-0.05)
    context = NFLGameContext(game_id=game.game_id, home_rest_days=7, away_rest_days=7,
                              wind_mph=6.0, temp_f=70.0, is_dome=False)
    spread_mkt = NFLMarket(game_id=game.game_id, market_type="spread", line=3.0,
                            home_odds=1.91, away_odds=1.91,
                            captured_at=datetime(2026, 9, 12, tzinfo=timezone.utc))
    total_mkt = NFLMarket(game_id=game.game_id, market_type="total", line=47.5,
                           over_odds=1.91, under_odds=1.91,
                           captured_at=datetime(2026, 9, 12, tzinfo=timezone.utc))

    session = MagicMock()
    session.execute = AsyncMock(side_effect=[
        _scalars_result([game]),              # due games query
        _scalar_result(home_stats),           # home team_stats
        _scalar_result(away_stats),           # away team_stats
        _scalar_result(context),              # game_context
        _scalars_result([spread_mkt, total_mkt]),  # markets
    ])
    session.commit = AsyncMock()

    result = await sched.snapshot_due_games(
        session, minutes_before=90, mov_bundle=_mov_bundle(), totals_bundle=_totals_bundle(),
    )

    assert result == {"snapshotted": 1}
    session.add.assert_called_once()
    added = session.add.call_args[0][0]
    assert isinstance(added, NFLPredictionSnapshot)
    assert added.game_id == "2026_02_CIN_KC"
    session.commit.assert_awaited_once()


# ---------------------------------------------------------------------------
# 4. grade_finals
# ---------------------------------------------------------------------------
async def test_grade_finals_no_op_when_nothing_ungraded():
    session = MagicMock()
    session.execute = AsyncMock(return_value=_all_result([]))
    session.commit = AsyncMock()

    result = await sched.grade_finals(session)

    assert result == {"graded": 0}
    session.commit.assert_awaited_once()


async def test_grade_finals_grades_one_final_game():
    game = NFLGame(
        game_id="2026_02_CIN_KC", season=2026, week=2, home_team="KC", away_team="CIN",
        status="final", home_score=30, away_score=20,
    )
    snap = NFLPredictionSnapshot(
        game_id="2026_02_CIN_KC", snapshot_time=datetime(2026, 9, 13, tzinfo=timezone.utc),
        home_team="KC", away_team="CIN",
        best_total_direction="over", best_total_line=44.0, best_total_odds=1.91,
        best_spread_team="home", best_spread_line=3.0, best_spread_odds=1.91,
        best_ml_team="home", best_ml_odds=1.5,
        best_bet_type="total", best_bet_result=None,
    )

    session = MagicMock()
    session.execute = AsyncMock(return_value=_all_result([(snap, game)]))
    session.commit = AsyncMock()

    result = await sched.grade_finals(session)

    assert result == {"graded": 1}
    assert snap.best_bet_result == "win"
    assert snap.home_score == 30 and snap.away_score == 20
    assert snap.actual_margin == 10 and snap.actual_total == 50
    session.commit.assert_awaited_once()


# ---------------------------------------------------------------------------
# 5. weekly_refresh / refresh_odds no-op paths
# ---------------------------------------------------------------------------
async def test_weekly_refresh_returns_counts_and_commits(monkeypatch):
    session = MagicMock()
    session.commit = AsyncMock()

    refresh_schedule_mock = AsyncMock(return_value=0)
    recompute_team_stats_mock = AsyncMock(return_value=0)
    monkeypatch.setattr(sched.season_update, "refresh_schedule", refresh_schedule_mock)
    monkeypatch.setattr(sched.season_update, "recompute_team_stats", recompute_team_stats_mock)

    result = await sched.weekly_refresh(session)

    assert result == {"games": 0, "stats": 0}
    refresh_schedule_mock.assert_awaited_once()
    recompute_team_stats_mock.assert_awaited_once()
    session.commit.assert_awaited_once()


async def test_refresh_odds_no_op_out_of_season(monkeypatch):
    session = MagicMock()
    session.commit = AsyncMock()

    odds_to_markets_mock = AsyncMock(return_value=0)
    monkeypatch.setattr(sched.season_update, "odds_to_markets", odds_to_markets_mock)

    result = await sched.refresh_odds(session)

    assert result == {"markets": 0}
    session.commit.assert_awaited_once()


# ---------------------------------------------------------------------------
# 6. start_scheduler guard
# ---------------------------------------------------------------------------
def test_start_scheduler_early_returns_when_disabled(monkeypatch):
    monkeypatch.setattr(sched.settings, "nfl_scheduler_enabled", False)

    scheduler_ctor = MagicMock(side_effect=AssertionError("should never build a Scheduler"))
    monkeypatch.setattr(sched.schedule, "Scheduler", scheduler_ctor)
    sleep_mock = MagicMock(side_effect=AssertionError("should never sleep"))
    monkeypatch.setattr(sched.time, "sleep", sleep_mock)
    init_engine_mock = MagicMock(side_effect=AssertionError("should never init the engine"))
    monkeypatch.setattr(sched, "_init_engine", init_engine_mock)

    # Should return immediately without raising.
    sched.start_scheduler()

    scheduler_ctor.assert_not_called()
    sleep_mock.assert_not_called()
    init_engine_mock.assert_not_called()
