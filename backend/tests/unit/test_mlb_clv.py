"""Closing-line value for MLB picks.

CLV is the fastest honest signal that the model knows something the market
does not: it resolves in tens of bets where win rate needs thousands. The
moneyline record is +1.18 SE from zero over 352 bets, so realized P&L cannot
yet distinguish edge from luck — this is the instrument that can.

Definition used throughout:

    clv = closing_novig_prob - (1 / bet_decimal_odds)

i.e. the market's vig-free consensus probability for the side we took, minus
the probability we needed just to break even at the price we got. Positive
means the closing market says our price was a profitable one, regardless of
whether the bet won.
"""
import pytest

from src.services.mlb.clv import (
    ClosingQuote,
    clv_from_closing,
    consensus_novig_prob,
    novig_prob_for_side,
)


class TestNovigProbForSide:
    def test_moneyline_home_side(self):
        q = ClosingQuote(ml_home=1.52, ml_away=2.57)
        p = novig_prob_for_side(q, market="moneyline", is_home=True)
        # raw 1/1.52 = .658, 1/2.57 = .389, overround .658+.389 = 1.047
        assert p == pytest.approx(0.6581 / 1.0472, abs=1e-3)
        assert 0.60 < p < 0.65

    def test_moneyline_away_side_is_the_complement(self):
        q = ClosingQuote(ml_home=1.52, ml_away=2.57)
        home = novig_prob_for_side(q, market="moneyline", is_home=True)
        away = novig_prob_for_side(q, market="moneyline", is_home=False)
        assert home + away == pytest.approx(1.0, abs=1e-9)

    def test_moneyline_missing_a_side_is_unusable(self):
        """One-sided prices cannot be devigged, so they must not be guessed at."""
        assert novig_prob_for_side(
            ClosingQuote(ml_home=1.52), market="moneyline", is_home=True
        ) is None

    def test_runline_matches_on_the_signed_home_line(self):
        q = ClosingQuote(rl_line=-1.5, rl_home_odds=2.10, rl_away_odds=1.74)
        p = novig_prob_for_side(q, market="runline", is_home=True, line=-1.5)
        assert p is not None and 0.40 < p < 0.50

    def test_runline_away_side_uses_the_mirrored_line(self):
        """Away +1.5 against a stored home line of -1.5."""
        q = ClosingQuote(rl_line=-1.5, rl_home_odds=2.10, rl_away_odds=1.74)
        p = novig_prob_for_side(q, market="runline", is_home=False, line=1.5)
        assert p is not None and 0.50 < p < 0.60

    def test_runline_rejects_a_different_line(self):
        """A close at +1.5 is not the same bet as a bet at -1.5."""
        q = ClosingQuote(rl_line=1.5, rl_home_odds=1.74, rl_away_odds=2.10)
        assert novig_prob_for_side(q, market="runline", is_home=True, line=-1.5) is None

    def test_total_over_matches_on_line(self):
        q = ClosingQuote(total_line=8.5, over_odds=1.91, under_odds=1.95)
        p = novig_prob_for_side(q, market="total", is_home=True, line=8.5, direction="over")
        assert p is not None and 0.45 < p < 0.55

    def test_total_rejects_a_moved_line(self):
        """8.5 -> 9.0 is a different bet; CLV across it would be meaningless."""
        q = ClosingQuote(total_line=9.0, over_odds=1.91, under_odds=1.95)
        assert novig_prob_for_side(
            q, market="total", is_home=True, line=8.5, direction="over"
        ) is None

    def test_unknown_market_is_none(self):
        assert novig_prob_for_side(
            ClosingQuote(ml_home=2.0, ml_away=2.0), market="parlay", is_home=True
        ) is None


class TestConsensusNovigProb:
    def test_averages_across_books(self):
        assert consensus_novig_prob([0.40, 0.42, 0.44]) == pytest.approx(0.42)

    def test_requires_a_minimum_number_of_books(self):
        """A single book is a quote, not a consensus."""
        assert consensus_novig_prob([0.40], min_books=2) is None

    def test_drops_none_values(self):
        assert consensus_novig_prob([0.40, None, 0.44], min_books=2) == pytest.approx(0.42)

    def test_empty_input_is_none(self):
        assert consensus_novig_prob([]) is None


class TestClvFromClosing:
    def test_positive_when_we_beat_the_close(self):
        """Bet 2.57 (break-even 38.9%) on a side the close prices at 42%."""
        clv = clv_from_closing(bet_odds=2.57, closing_novig_prob=0.42)
        assert clv == pytest.approx(0.42 - 1 / 2.57, abs=1e-9)
        assert clv > 0

    def test_negative_when_the_market_moved_against_us(self):
        assert clv_from_closing(bet_odds=2.10, closing_novig_prob=0.40) < 0

    def test_zero_when_our_price_exactly_matches_the_close(self):
        assert clv_from_closing(bet_odds=2.5, closing_novig_prob=0.4) == pytest.approx(0.0)

    def test_missing_inputs_are_none_not_zero(self):
        """A missing close must not read as neutral CLV — it is no measurement."""
        assert clv_from_closing(bet_odds=None, closing_novig_prob=0.42) is None
        assert clv_from_closing(bet_odds=2.5, closing_novig_prob=None) is None

    def test_rejects_nonsense_odds(self):
        assert clv_from_closing(bet_odds=0.0, closing_novig_prob=0.42) is None
