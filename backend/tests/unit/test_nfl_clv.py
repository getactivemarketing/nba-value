"""NFL closing-line value: LINE clv and PRICE clv, kept separate.

MLB needed only price CLV — a moneyline has no line to move. NFL spreads and
totals move on two axes at once, and collapsing them loses the distinction
that matters: catching -3.5 before it became -4.5 is a different achievement
from catching -110 before it became -120, and only one of them survives a
key-number crossing.

Promotion is gated on CLV before win rate or P&L, so these have to be right.
"""
import pytest

from src.services.nfl.clv import (
    KEY_NUMBERS,
    crossed_key_number,
    line_clv,
    price_clv,
    novig_two_way,
)


class TestLineClv:
    def test_positive_when_the_line_moves_toward_our_side(self):
        """Took home -3.5; it closed -4.5. We got the better number."""
        assert line_clv(bet_line=-3.5, closing_line=-4.5, side="home") == pytest.approx(1.0)

    def test_negative_when_the_line_moves_against_us(self):
        assert line_clv(bet_line=-3.5, closing_line=-2.5, side="home") == pytest.approx(-1.0)

    def test_away_side_mirrors_the_home_side(self):
        assert line_clv(bet_line=-3.5, closing_line=-4.5, side="away") == pytest.approx(-1.0)

    def test_over_gains_when_the_total_drops(self):
        assert line_clv(bet_line=44.5, closing_line=43.0, side="over") == pytest.approx(1.5)

    def test_under_gains_when_the_total_rises(self):
        assert line_clv(bet_line=44.5, closing_line=46.0, side="under") == pytest.approx(1.5)

    def test_unmoved_line_is_zero_not_none(self):
        assert line_clv(bet_line=-3.0, closing_line=-3.0, side="home") == 0.0

    def test_missing_closing_line_is_unmeasured(self):
        """None means 'no measurement', never 'neutral'."""
        assert line_clv(bet_line=-3.0, closing_line=None, side="home") is None


class TestPriceClv:
    def test_positive_when_we_beat_the_closing_fair_price(self):
        clv = price_clv(bet_odds=2.10, closing_novig_prob=0.50)
        assert clv == pytest.approx(0.50 - 1 / 2.10, abs=1e-9)
        assert clv > 0

    def test_negative_when_the_market_moved_against_us(self):
        assert price_clv(bet_odds=1.80, closing_novig_prob=0.50) < 0

    def test_missing_inputs_are_unmeasured(self):
        assert price_clv(bet_odds=None, closing_novig_prob=0.5) is None
        assert price_clv(bet_odds=2.0, closing_novig_prob=None) is None

    def test_rejects_impossible_odds(self):
        assert price_clv(bet_odds=1.0, closing_novig_prob=0.5) is None


class TestNovigTwoWay:
    def test_removes_the_overround(self):
        h, a = novig_two_way(1.91, 1.91)
        assert h + a == pytest.approx(1.0)
        assert h == pytest.approx(0.5)

    def test_favours_the_shorter_price(self):
        h, a = novig_two_way(1.50, 2.60)
        assert h > a and h + a == pytest.approx(1.0)

    def test_missing_side_is_unusable(self):
        assert novig_two_way(1.91, None) == (None, None)


class TestCrossedKeyNumber:
    def test_detects_crossing_three(self):
        """The most valuable move in NFL spread betting."""
        assert crossed_key_number(-2.5, -3.5) is True

    def test_detects_crossing_seven(self):
        assert crossed_key_number(-6.5, -7.5) is True

    def test_move_within_a_gap_is_not_a_crossing(self):
        assert crossed_key_number(-4.5, -5.5) is False

    def test_direction_does_not_matter(self):
        assert crossed_key_number(-3.5, -2.5) is True

    def test_landing_on_a_key_number_counts(self):
        assert crossed_key_number(-2.5, -3.0) is True

    def test_three_and_seven_are_both_key(self):
        assert 3 in KEY_NUMBERS and 7 in KEY_NUMBERS

    def test_missing_line_is_none(self):
        assert crossed_key_number(-3.5, None) is None
