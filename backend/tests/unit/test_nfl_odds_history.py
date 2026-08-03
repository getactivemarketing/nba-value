"""NFL odds history — the append-only record nfl_markets never kept.

`nfl_markets` is overwritten in place, exactly like `mlb_markets` was. That
already cost MLB every historical candidate set: no backtest can ask "what else
was available that night", and closing-line value was uncomputable until an
append-only table was added.

NFL has not opened yet, so this is the one time the mistake can be avoided
rather than repaired. Capture must be live before the first preseason slate.

Three-way markets make this stricter than MLB. A spread and a total both carry
a line AND a price, so a book can move either independently — juice moving on a
held line is a real signal and must not be filtered out as "no movement".
"""
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from src.services.nfl.odds_history import (
    NFLOddsQuote,
    minutes_to_kickoff,
    nfl_prices_changed,
    quote_from_bookmaker,
    select_closing_quote,
)

T0 = datetime(2026, 9, 13, 17, 0, tzinfo=timezone.utc)
NAME_TO_ABBR = {"Buffalo Bills": "BUF", "Miami Dolphins": "MIA"}


def q(**kw) -> NFLOddsQuote:
    base = dict(
        ml_home=1.55, ml_away=2.55,
        spread_line=-3.5, spread_home_odds=1.91, spread_away_odds=1.91,
        total_line=47.5, over_odds=1.91, under_odds=1.91,
    )
    base.update(kw)
    return NFLOddsQuote(**base)


class TestPricesChanged:
    def test_first_observation_is_always_recorded(self):
        assert nfl_prices_changed(None, q()) is True

    def test_identical_quote_is_not_a_change(self):
        assert nfl_prices_changed(q(), q()) is False

    def test_spread_line_move_is_a_change(self):
        assert nfl_prices_changed(q(), q(spread_line=-3.0)) is True

    def test_spread_juice_move_on_a_held_line_is_a_change(self):
        """-3.5 at -110 and -3.5 at -120 are different bets."""
        assert nfl_prices_changed(q(), q(spread_home_odds=1.83)) is True

    def test_total_line_move_is_a_change(self):
        assert nfl_prices_changed(q(), q(total_line=48.0)) is True

    def test_moneyline_move_is_a_change(self):
        assert nfl_prices_changed(q(), q(ml_home=1.50)) is True

    def test_decimal_and_float_compare_equal(self):
        same = NFLOddsQuote(
            ml_home=Decimal("1.55"), ml_away=Decimal("2.55"),
            spread_line=Decimal("-3.5"), spread_home_odds=Decimal("1.91"),
            spread_away_odds=Decimal("1.91"), total_line=Decimal("47.5"),
            over_odds=Decimal("1.91"), under_odds=Decimal("1.91"),
        )
        assert nfl_prices_changed(q(), same) is False

    def test_market_appearing_is_a_change(self):
        assert nfl_prices_changed(q(total_line=None, over_odds=None, under_odds=None), q()) is True


class TestMinutesToKickoff:
    def test_positive_before_kickoff(self):
        assert minutes_to_kickoff(T0, T0 + timedelta(minutes=90)) == 90

    def test_negative_after_kickoff(self):
        assert minutes_to_kickoff(T0, T0 - timedelta(minutes=15)) == -15

    def test_zero_at_kickoff(self):
        assert minutes_to_kickoff(T0, T0) == 0


class TestSelectClosingQuote:
    def test_latest_quote_at_or_before_kickoff(self):
        rows = [(1, T0 - timedelta(hours=3)), (2, T0 - timedelta(minutes=4)),
                (3, T0 - timedelta(hours=1))]
        assert select_closing_quote(rows, T0) == 2

    def test_ignores_in_game_prices(self):
        rows = [(1, T0 - timedelta(minutes=2)), (2, T0 + timedelta(minutes=45))]
        assert select_closing_quote(rows, T0) == 1

    def test_none_when_everything_is_post_kickoff(self):
        assert select_closing_quote([(1, T0 + timedelta(minutes=1))], T0) is None


class TestQuoteFromBookmaker:
    def _bm(self, markets):
        return {"key": "draftkings", "markets": markets}

    def test_parses_all_three_markets(self):
        bm = self._bm([
            {"key": "h2h", "outcomes": [
                {"name": "Miami Dolphins", "price": 2.55},
                {"name": "Buffalo Bills", "price": 1.55}]},
            {"key": "spreads", "outcomes": [
                {"name": "Buffalo Bills", "price": 1.91, "point": -3.5},
                {"name": "Miami Dolphins", "price": 1.91, "point": 3.5}]},
            {"key": "totals", "outcomes": [
                {"name": "Over", "price": 1.91, "point": 47.5},
                {"name": "Under", "price": 1.87, "point": 47.5}]},
        ])
        got = quote_from_bookmaker(bm, home_team="BUF", name_to_abbr=NAME_TO_ABBR)
        assert got.ml_home == 1.55 and got.ml_away == 2.55
        assert got.spread_line == -3.5          # SIGNED from the home perspective
        assert got.spread_home_odds == 1.91
        assert got.total_line == 47.5 and got.under_odds == 1.87

    def test_home_resolved_by_name_not_position(self):
        bm = self._bm([{"key": "h2h", "outcomes": [
            {"name": "Buffalo Bills", "price": 1.55},
            {"name": "Miami Dolphins", "price": 2.55}]}])
        got = quote_from_bookmaker(bm, home_team="BUF", name_to_abbr=NAME_TO_ABBR)
        assert got.ml_home == 1.55

    def test_home_underdog_spread_keeps_its_sign(self):
        bm = self._bm([{"key": "spreads", "outcomes": [
            {"name": "Buffalo Bills", "price": 1.91, "point": 2.5},
            {"name": "Miami Dolphins", "price": 1.91, "point": -2.5}]}])
        got = quote_from_bookmaker(bm, home_team="BUF", name_to_abbr=NAME_TO_ABBR)
        assert got.spread_line == 2.5

    def test_missing_markets_are_none(self):
        assert quote_from_bookmaker(self._bm([]), "BUF", NAME_TO_ABBR) == NFLOddsQuote()
