"""Closing-line attachment and CLV settlement.

Two closes are attached to every row, never one:

  same-book   what the bettor could actually have got at that book. This is
              the honest bettor-oriented number.
  consensus   median across books, robust to a single stale or wide book.
              This is the number that survives one book being broken.

Where they disagree, the disagreement is itself information — so both are
stored with explicit staleness and fallback metadata rather than silently
collapsed into a single "closing line".

Frozen candidate and model fields are never rewritten. Settlement only fills
settlement columns.
"""
from datetime import datetime, timedelta, timezone

import pytest

from src.services.nfl.settlement import (
    CloseSource,
    canonical_line,
    canonical_price,
    consensus_close,
    select_book_close,
    settle_candidate,
)

KO = datetime(2026, 9, 13, 17, 0, tzinfo=timezone.utc)


def quote(book, minutes_before, line=-3.5, home=1.91, away=1.91):
    return {
        "book": book,
        "snapshot_time": KO - timedelta(minutes=minutes_before),
        "spread_line": line, "spread_home_odds": home, "spread_away_odds": away,
    }


class TestCanonicalisation:
    def test_line_trailing_zeros_collapse(self):
        assert canonical_line(-3.50) == canonical_line(-3.5) == "-3.5"

    def test_negative_zero_is_zero(self):
        """-0.0 and 0.0 are the same pick'em line and must not split identity."""
        assert canonical_line(-0.0) == canonical_line(0.0) == "0"

    def test_half_points_survive(self):
        assert canonical_line(-3.5) != canonical_line(-3.0)

    def test_none_is_a_stable_sentinel(self):
        assert canonical_line(None) == "NONE"
        assert canonical_line(None) != canonical_line(0.0)

    def test_price_rounds_to_book_precision(self):
        """1.9100000001 and 1.91 are the same price; float noise must not split it."""
        assert canonical_price(1.9100000001) == canonical_price(1.91) == "1.91"

    def test_price_distinguishes_real_juice_moves(self):
        assert canonical_price(1.91) != canonical_price(1.83)

    def test_decimal_and_float_canonicalise_alike(self):
        from decimal import Decimal
        assert canonical_line(Decimal("-3.5")) == canonical_line(-3.5)
        assert canonical_price(Decimal("1.910")) == canonical_price(1.91)


class TestSelectBookClose:
    def test_latest_quote_before_kickoff_wins(self):
        quotes = [quote("dk", 180), quote("dk", 5), quote("dk", 60)]
        got = select_book_close(quotes, KO)
        assert got["staleness_min"] == 5

    def test_in_game_quotes_are_excluded(self):
        quotes = [quote("dk", 10), {**quote("dk", 0), "snapshot_time": KO + timedelta(minutes=30)}]
        assert select_book_close(quotes, KO)["staleness_min"] == 10

    def test_quote_exactly_at_kickoff_counts(self):
        assert select_book_close([quote("dk", 0)], KO)["staleness_min"] == 0

    def test_no_pre_kickoff_quote_returns_none(self):
        late = [{**quote("dk", 0), "snapshot_time": KO + timedelta(minutes=1)}]
        assert select_book_close(late, KO) is None

    def test_staleness_is_reported_even_when_large(self):
        """A three-day-old 'close' is attachable but must be visibly stale."""
        got = select_book_close([quote("dk", 4320)], KO)
        assert got["staleness_min"] == 4320


class TestConsensusClose:
    def test_median_line_across_books(self):
        quotes = [quote("a", 5, line=-3.0), quote("b", 5, line=-3.5), quote("c", 5, line=-4.0)]
        got = consensus_close(quotes, KO, market="spread")
        assert got["line"] == pytest.approx(-3.5)
        assert got["book_count"] == 3

    def test_median_resists_one_broken_book(self):
        """A stale outlier must not drag the consensus; a mean would."""
        quotes = [quote("a", 5, line=-3.5), quote("b", 5, line=-3.5),
                  quote("c", 5, line=-14.0)]
        got = consensus_close(quotes, KO, market="spread")
        assert got["line"] == pytest.approx(-3.5)

    def test_requires_a_minimum_book_count(self):
        assert consensus_close([quote("a", 5)], KO, market="spread", min_books=3) is None

    def test_uses_each_book_s_own_latest_pre_kickoff_quote(self):
        quotes = [quote("a", 200, line=-2.5), quote("a", 5, line=-3.5), quote("b", 5, line=-3.5)]
        got = consensus_close(quotes, KO, market="spread", min_books=2)
        assert got["line"] == pytest.approx(-3.5)
        assert got["book_count"] == 2


class TestSettleCandidate:
    CAND = {"market_type": "spread", "side": "home", "line": -3.5, "odds_decimal": 1.91}

    def _close(self, line=-4.5, home=1.91, away=1.91, staleness=5):
        return {"line": line, "odds": home, "opposing_odds": away,
                "staleness_min": staleness, "source": CloseSource.BOOK}

    def test_line_clv_positive_when_the_number_improved(self):
        got = settle_candidate(self.CAND, book_close=self._close(), consensus=None,
                               actual_value=10.0)
        assert got["line_clv"] == pytest.approx(1.0)

    def test_price_clv_uses_the_devigged_close(self):
        got = settle_candidate(self.CAND, book_close=self._close(home=1.80, away=2.05),
                               consensus=None, actual_value=10.0)
        assert got["closing_novig_prob"] is not None
        assert got["price_clv"] == pytest.approx(
            got["closing_novig_prob"] - 1 / 1.91, abs=1e-9)

    def test_key_number_crossing_flagged(self):
        got = settle_candidate({**self.CAND, "line": -2.5},
                               book_close=self._close(line=-3.5), consensus=None,
                               actual_value=10.0)
        assert got["crossed_key"] is True

    def test_side_won_reflects_the_outcome(self):
        won = settle_candidate(self.CAND, self._close(), None, actual_value=10.0)
        lost = settle_candidate(self.CAND, self._close(), None, actual_value=1.0)
        assert won["side_won"] is True and lost["side_won"] is False

    def test_exact_push_is_none_not_false(self):
        """A half-point line can never push, so this needs a whole number.

        Odds API convention: home -3.0 means home lays 3, so the push is a
        home margin of exactly +3 — not -3.
        """
        cand = {**self.CAND, "line": -3.0}
        got = settle_candidate(cand, self._close(), None, actual_value=3.0)
        assert got["side_won"] is None
        assert settle_candidate(cand, self._close(), None, actual_value=4.0)["side_won"] is True
        assert settle_candidate(cand, self._close(), None, actual_value=2.0)["side_won"] is False

    def test_consensus_clv_reported_alongside_book_clv(self):
        got = settle_candidate(
            self.CAND, book_close=self._close(line=-4.5),
            consensus={"line": -4.0, "novig_prob": 0.52, "book_count": 9},
            actual_value=10.0)
        assert got["line_clv"] == pytest.approx(1.0)
        assert got["line_clv_consensus"] == pytest.approx(0.5)
        assert got["consensus_book_count"] == 9

    def test_missing_book_close_falls_back_to_consensus(self):
        got = settle_candidate(
            self.CAND, book_close=None,
            consensus={"line": -4.0, "novig_prob": 0.52, "book_count": 9},
            actual_value=10.0)
        assert got["closing_source"] == CloseSource.CONSENSUS_FALLBACK
        assert got["line_clv"] == pytest.approx(0.5)

    def test_no_close_at_all_is_unmeasured_not_zero(self):
        got = settle_candidate(self.CAND, None, None, actual_value=10.0)
        assert got["closing_source"] == CloseSource.NONE
        assert got["line_clv"] is None and got["price_clv"] is None
        # The outcome is still known even when CLV is not.
        assert got["side_won"] is True
