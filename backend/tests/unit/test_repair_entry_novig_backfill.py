"""The backfilled entry price is unrecoverable, so it must become NULL.

These columns already document that NULL means "not measured, never neutral".
A derived value that does not calibrate is worse than a gap: market_move and
vig_paid are computed from it, and CLV is what model changes get promoted on.
"""
from datetime import date

from src.tasks.repair_entry_novig_backfill import (
    DERIVED_COLUMNS, ODDS_HISTORY_START, is_unverifiable,
)


class TestCutoff:
    def test_cutoff_is_the_first_recorded_pre_game_price(self):
        # mlb_odds_snapshots begins 2026-07-31T14:49:52Z. Before that no market
        # price was ever recorded, so nothing is reconstructible.
        assert ODDS_HISTORY_START == date(2026, 7, 31)

    def test_clv_is_not_retracted(self):
        # clv = closing_novig - 1/odds. It never touched entry_novig, so it
        # survives the repair; retracting it would destroy real measurements.
        assert "clv" not in DERIVED_COLUMNS
        assert "closing_novig_prob" not in DERIVED_COLUMNS

    def test_everything_derived_from_the_entry_price_is_retracted(self):
        assert set(DERIVED_COLUMNS) == {"entry_novig_prob", "market_move", "vig_paid"}


class TestIsUnverifiable:
    def test_a_game_before_the_cutoff_is_unverifiable(self):
        assert is_unverifiable(date(2026, 7, 30)) is True

    def test_the_cutoff_day_itself_is_verifiable(self):
        # Odds history starts on this date, and stored values from here agree
        # with a rebuilt consensus to a median of 0.0068.
        assert is_unverifiable(date(2026, 7, 31)) is False

    def test_a_later_game_is_verifiable(self):
        assert is_unverifiable(date(2026, 8, 4)) is False

    def test_the_whole_backfilled_season_is_unverifiable(self):
        for d in (date(2026, 4, 1), date(2026, 5, 15), date(2026, 6, 30), date(2026, 7, 29)):
            assert is_unverifiable(d) is True, d

    def test_a_null_game_date_fails_closed(self):
        # Cannot be shown to fall after the cutoff. Retracting a good value
        # costs a gap; keeping a bad one costs a wrong CLV number.
        assert is_unverifiable(None) is True

    def test_the_cutoff_is_injectable_so_the_rule_is_not_hardcoded_to_one_season(self):
        assert is_unverifiable(date(2026, 8, 4), cutoff=date(2026, 8, 10)) is True
        assert is_unverifiable(date(2026, 8, 4), cutoff=date(2026, 8, 1)) is False
