from src.config import settings
from src.edge.social import ledger


def test_enabled_respects_kind_and_sport_lists(monkeypatch):
    monkeypatch.setattr(settings, "social_kinds", "blind, final")
    monkeypatch.setattr(settings, "social_sports", "nba")
    assert ledger.enabled("blind", "nba") and ledger.enabled("final", "nba")
    assert not ledger.enabled("value", "nba") and not ledger.enabled("blind", "nfl")
    assert not ledger.enabled("bli", "nba")  # whole-name match, not substring
