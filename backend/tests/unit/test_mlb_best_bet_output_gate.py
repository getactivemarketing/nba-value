"""best_bet output pause: nothing reaches anyone, everything keeps measuring.

The pause exists because the selection rule is adversely selecting the model's
own error (docs/engineering/03 §10). The constraint that shapes the design is
that CLV must keep accruing — the CLV job selects on `best_bet_team IS NOT
NULL`, so a pause implemented by nulling best_bet at the scorer would destroy
the only instrument with enough power to justify ever un-pausing.
"""
from datetime import datetime, timedelta, timezone

import pytest

from src.api.mlb import eligible_for_preview
from src.services.mlb.value_calculator import MLBValueCalculator, MLBValueResult
from src.services.notifications.pick_alerts import pick_alerts_enabled


class _Snap:
    """Minimal snapshot stand-in for the preview gate."""
    def __init__(self, **kw):
        self.best_bet_type = kw.get("best_bet_type", "moneyline")
        self.best_ml_team = kw.get("best_ml_team", "CWS")
        self.best_ml_odds = kw.get("best_ml_odds", 2.55)
        self.game_time = kw.get(
            "game_time", datetime.now(timezone.utc) + timedelta(hours=4)
        )


class TestPickAlertsGate:
    def test_alerts_are_off_when_paused(self):
        assert pick_alerts_enabled(live=False) is False

    def test_alerts_are_on_when_live(self):
        assert pick_alerts_enabled(live=True) is True

    def test_the_gate_is_the_only_thing_that_decides(self):
        # No secondary condition may re-enable texting while paused — the
        # founder's phone is the one surface that prompts an actual bet.
        for extra in (True, False):
            assert pick_alerts_enabled(live=False, has_picks=extra) is False


class TestPreviewGate:
    def test_previews_are_refused_when_paused(self):
        # A pick-preview video is a public recommendation with odds attached.
        # Pausing best_bet while still publishing videos of it would be
        # incoherent.
        assert eligible_for_preview(_Snap(), live=False) is False

    def test_previews_still_work_when_live(self):
        assert eligible_for_preview(_Snap(), live=True) is True

    def test_the_pause_does_not_replace_the_existing_gates(self):
        # Being live must not smuggle a past-lead-time or non-moneyline pick
        # through — the pause is an ADDITIONAL gate, not a substitute.
        started = _Snap(game_time=datetime.now(timezone.utc) - timedelta(minutes=5))
        assert eligible_for_preview(started, live=True) is False
        totals = _Snap(best_bet_type="total")
        assert eligible_for_preview(totals, live=True) is False


class TestMeasurementSurvivesThePause:
    """The whole point: picks keep being computed so CLV keeps accruing."""

    def _values(self):
        return [
            MLBValueResult(
                market_type="moneyline", bet_type="away_ml", team="CWS", line=None,
                model_prob=0.48, market_prob=0.392,
                raw_edge=0.088, edge_pct=22.4,
                value_score=62, confidence="medium",
                odds_decimal=2.55, odds_american=155,
                is_value_bet=True, sort_score=22.4,
            ),
        ]

    def test_find_best_bet_is_untouched_by_the_pause(self):
        # find_best_bet takes no `live` flag on purpose. If the pause reached
        # the scorer, best_bet_team would be NULL and the CLV job — which
        # filters on best_bet_team IS NOT NULL — would measure nothing.
        import inspect
        params = inspect.signature(MLBValueCalculator.find_best_bet).parameters
        assert "live" not in params
        assert "mlb_best_bet_live" not in params

    def test_a_pick_is_still_selected_while_paused(self):
        best = MLBValueCalculator.find_best_bet(self._values())
        assert best is not None
        assert best.market_type == "moneyline"


class TestTheAlertJobActuallyNoOps:
    """The gate must stop the job before it reads or writes anything.

    Returning early only after querying would still consume snapshots and set
    sms_alert_sent, so lifting the pause would silently skip every pick that
    came up while it was on.
    """

    def _run(self, monkeypatch, live: bool):
        import asyncio
        import src.tasks.social_scheduler as ss
        from src.config import settings

        monkeypatch.setattr(settings, "mlb_best_bet_live", live)

        class Exploding:
            def __call__(self):
                raise AssertionError(
                    "the paused alert job opened a DB session"
                )

        monkeypatch.setattr(ss, "_social_session_factory", Exploding())
        monkeypatch.setattr(
            "src.services.notifications.sms.send_sms",
            lambda *a, **k: (_ for _ in ()).throw(
                AssertionError("the paused alert job sent an SMS")
            ),
        )
        return asyncio.run(ss._post_pick_alerts_async())

    def test_paused_job_touches_neither_db_nor_sms(self, monkeypatch):
        result = self._run(monkeypatch, live=False)
        assert result["sent"] == 0
        assert result["paused"] is True

    def test_the_guard_is_what_stops_it_not_an_empty_slate(self, monkeypatch):
        # Same harness with the pause lifted must reach the session factory —
        # proving the no-op above came from the gate, not from the fakes.
        with pytest.raises(AssertionError, match="opened a DB session"):
            self._run(monkeypatch, live=True)

