"""The pick-preview endpoint's selection rules.

Selection is where the disabled markets are mechanically enforced. Runline is
paused and totals suppressed; publishing them would advertise markets we have
deliberately turned off.
"""
from datetime import datetime, timedelta, timezone

import pytest

from src.api.mlb import eligible_for_preview


class _Snap:
    def __init__(self, **kw):
        self.best_bet_type = kw.get("best_bet_type", "moneyline")
        self.best_ml_team = kw.get("best_ml_team", "CWS")
        self.best_ml_odds = kw.get("best_ml_odds", 2.55)
        self.game_time = kw.get("game_time")


def _in(minutes: int) -> datetime:
    return datetime.now(timezone.utc) + timedelta(minutes=minutes)


class TestEligibility:
    def test_moneyline_with_enough_lead_time_is_eligible(self):
        assert eligible_for_preview(_Snap(game_time=_in(120))) is True

    def test_runline_is_never_eligible(self):
        assert eligible_for_preview(_Snap(best_bet_type="runline", game_time=_in(120))) is False

    def test_total_is_never_eligible(self):
        assert eligible_for_preview(_Snap(best_bet_type="total", game_time=_in(120))) is False

    def test_inside_the_lead_time_gate_is_rejected(self):
        assert eligible_for_preview(_Snap(game_time=_in(30))) is False

    def test_already_started_is_rejected(self):
        assert eligible_for_preview(_Snap(game_time=_in(-10))) is False

    def test_exactly_at_the_gate_is_rejected(self):
        """Boundary is strict — Blotato can hold a post for minutes."""
        assert eligible_for_preview(_Snap(game_time=_in(45)), min_lead_minutes=45) is False

    def test_missing_price_is_rejected(self):
        assert eligible_for_preview(_Snap(best_ml_odds=None, game_time=_in(120))) is False

    def test_missing_game_time_is_rejected(self):
        assert eligible_for_preview(_Snap(game_time=None)) is False
