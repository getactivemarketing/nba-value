"""MLB odds history capture — the append-only record `mlb_markets` never kept.

`mlb_markets` is overwritten in place, so line movement and closing prices are
destroyed on every 30-minute ingest. These tests pin the change-detection rule
that decides when a new history row is worth writing, and the closing-line
selection that CLV is computed against.
"""
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from src.services.mlb.odds_history import (
    OddsQuote,
    prices_changed,
    quote_from_bookmaker,
    select_closing_quote,
    minutes_to_first_pitch,
)

NAME_TO_ABBR = {"Toronto Blue Jays": "TOR", "St. Louis Cardinals": "STL"}

T0 = datetime(2026, 8, 1, 18, 0, tzinfo=timezone.utc)


def q(**kw) -> OddsQuote:
    """An otherwise-complete quote, overridden field by field."""
    base = dict(
        ml_home=2.10, ml_away=1.80,
        rl_line=-1.5, rl_home_odds=1.45, rl_away_odds=2.70,
        total_line=8.5, over_odds=1.91, under_odds=1.95,
    )
    base.update(kw)
    return OddsQuote(**base)


class TestPricesChanged:
    def test_no_previous_quote_is_always_a_change(self):
        """The first observation of a game/book must always be recorded."""
        assert prices_changed(None, q()) is True

    def test_identical_quote_is_not_a_change(self):
        assert prices_changed(q(), q()) is False

    def test_moneyline_move_is_a_change(self):
        assert prices_changed(q(), q(ml_home=2.15)) is True

    def test_total_line_move_is_a_change(self):
        assert prices_changed(q(), q(total_line=9.0)) is True

    def test_juice_only_move_is_a_change(self):
        """Hold moving while the line holds is exactly the signal we want."""
        assert prices_changed(q(), q(over_odds=1.87, under_odds=2.00)) is True

    def test_runline_flip_is_a_change(self):
        assert prices_changed(q(), q(rl_line=1.5)) is True

    def test_decimal_and_float_compare_equal(self):
        """Values round-trip through Numeric columns as Decimal."""
        prev = q()
        same = OddsQuote(
            ml_home=Decimal("2.10"), ml_away=Decimal("1.80"),
            rl_line=Decimal("-1.5"), rl_home_odds=Decimal("1.45"),
            rl_away_odds=Decimal("2.70"), total_line=Decimal("8.5"),
            over_odds=Decimal("1.91"), under_odds=Decimal("1.95"),
        )
        assert prices_changed(prev, same) is False

    def test_appearing_market_is_a_change(self):
        """A book that had no total and now posts one has moved."""
        assert prices_changed(q(total_line=None, over_odds=None, under_odds=None), q()) is True

    def test_disappearing_market_is_a_change(self):
        assert prices_changed(q(), q(ml_home=None, ml_away=None)) is True

    def test_both_sides_missing_stays_unchanged(self):
        a = q(ml_home=None, ml_away=None)
        assert prices_changed(a, q(ml_home=None, ml_away=None)) is False


class TestMinutesToFirstPitch:
    def test_positive_before_first_pitch(self):
        assert minutes_to_first_pitch(T0, T0 + timedelta(minutes=90)) == 90

    def test_zero_at_first_pitch(self):
        assert minutes_to_first_pitch(T0, T0) == 0

    def test_negative_after_first_pitch(self):
        """NBA's writer recorded values down to -48; keep the sign meaningful."""
        assert minutes_to_first_pitch(T0, T0 - timedelta(minutes=20)) == -20

    def test_rounds_to_whole_minutes(self):
        assert minutes_to_first_pitch(T0, T0 + timedelta(seconds=150)) == 2


class TestSelectClosingQuote:
    """The closing line is the last price observed BEFORE first pitch.

    `is_closing_line` was hardcoded False in the NBA writer and has never been
    true for any row, which is why CLV has never been computable.
    """

    def test_picks_latest_quote_before_first_pitch(self):
        rows = [
            (1, T0 - timedelta(minutes=180)),
            (2, T0 - timedelta(minutes=10)),
            (3, T0 - timedelta(minutes=60)),
        ]
        assert select_closing_quote(rows, T0) == 2

    def test_ignores_quotes_after_first_pitch(self):
        """A live in-game price is not a closing line."""
        rows = [
            (1, T0 - timedelta(minutes=5)),
            (2, T0 + timedelta(minutes=30)),
        ]
        assert select_closing_quote(rows, T0) == 1

    def test_quote_exactly_at_first_pitch_counts_as_closing(self):
        rows = [(1, T0 - timedelta(minutes=45)), (2, T0)]
        assert select_closing_quote(rows, T0) == 2

    def test_returns_none_when_every_quote_is_post_pitch(self):
        rows = [(1, T0 + timedelta(minutes=1))]
        assert select_closing_quote(rows, T0) is None

    def test_returns_none_for_empty_input(self):
        assert select_closing_quote([], T0) is None


class TestQuoteFromBookmaker:
    """Parse one bookmaker's Odds API payload into a single quote.

    Home/away must be resolved by team name, never by outcome order — the API
    does not guarantee ordering, and getting this backwards is exactly the
    class of bug that produced the runline sign-pairing incident.
    """

    def _bookmaker(self, markets):
        return {"key": "fanduel", "markets": markets}

    def test_parses_all_three_markets(self):
        bm = self._bookmaker([
            {"key": "h2h", "outcomes": [
                {"name": "St. Louis Cardinals", "price": 2.57},
                {"name": "Toronto Blue Jays", "price": 1.52},
            ]},
            {"key": "spreads", "outcomes": [
                {"name": "Toronto Blue Jays", "price": 2.10, "point": -1.5},
                {"name": "St. Louis Cardinals", "price": 1.74, "point": 1.5},
            ]},
            {"key": "totals", "outcomes": [
                {"name": "Over", "price": 1.91, "point": 8.5},
                {"name": "Under", "price": 1.95, "point": 8.5},
            ]},
        ])
        quote = quote_from_bookmaker(bm, home_team="TOR", name_to_abbr=NAME_TO_ABBR)

        assert quote.ml_home == 1.52
        assert quote.ml_away == 2.57
        assert quote.rl_line == -1.5          # signed from the HOME perspective
        assert quote.rl_home_odds == 2.10
        assert quote.rl_away_odds == 1.74
        assert quote.total_line == 8.5
        assert quote.over_odds == 1.91
        assert quote.under_odds == 1.95

    def test_home_is_resolved_by_name_not_position(self):
        """Same game, outcomes listed home-first — must give the same answer."""
        bm = self._bookmaker([
            {"key": "h2h", "outcomes": [
                {"name": "Toronto Blue Jays", "price": 1.52},
                {"name": "St. Louis Cardinals", "price": 2.57},
            ]},
        ])
        quote = quote_from_bookmaker(bm, home_team="TOR", name_to_abbr=NAME_TO_ABBR)
        assert quote.ml_home == 1.52
        assert quote.ml_away == 2.57

    def test_runline_sign_follows_the_home_side(self):
        """Home dog on +1.5 must store rl_line = +1.5, not 1.5 stripped of sign."""
        bm = self._bookmaker([
            {"key": "spreads", "outcomes": [
                {"name": "Toronto Blue Jays", "price": 1.74, "point": 1.5},
                {"name": "St. Louis Cardinals", "price": 2.10, "point": -1.5},
            ]},
        ])
        quote = quote_from_bookmaker(bm, home_team="TOR", name_to_abbr=NAME_TO_ABBR)
        assert quote.rl_line == 1.5
        assert quote.rl_home_odds == 1.74
        assert quote.rl_away_odds == 2.10

    def test_missing_markets_are_none_not_an_error(self):
        quote = quote_from_bookmaker(
            self._bookmaker([]), home_team="TOR", name_to_abbr=NAME_TO_ABBR
        )
        assert quote == OddsQuote()

    def test_unknown_team_name_does_not_crash(self):
        bm = self._bookmaker([
            {"key": "h2h", "outcomes": [
                {"name": "Some Expansion Team", "price": 2.00},
                {"name": "Toronto Blue Jays", "price": 1.80},
            ]},
        ])
        quote = quote_from_bookmaker(bm, home_team="TOR", name_to_abbr=NAME_TO_ABBR)
        assert quote.ml_home == 1.80
        assert quote.ml_away == 2.00   # the unmatched side is still the away side

    def test_ignores_market_keys_we_do_not_track(self):
        bm = self._bookmaker([
            {"key": "h2h_lay", "outcomes": [{"name": "Toronto Blue Jays", "price": 9.9}]},
        ])
        assert quote_from_bookmaker(
            bm, home_team="TOR", name_to_abbr=NAME_TO_ABBR
        ) == OddsQuote()
