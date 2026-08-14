"""Separating market movement from vig in CLV.

A single CLV number conflates two different questions:

  did the model know something?   -> did the market move toward our side
  was the bet worth making?       -> did that movement exceed what we paid

Reported together they are indistinguishable, and the combined number is
dominated by vig. On the first 17 measured MLB picks the market moved toward
us 14 times (+0.34 points) while we paid 1.45 points of vig — so stored CLV
read -1.11 and "1 of 17 beat the close", which looks like a broken model and
is actually a small real signal buried under cost.

    clv          = closing_novig - 1/bet_odds        profitability
    market_move  = closing_novig - entry_novig       skill
    vig_paid     = 1/bet_odds     - entry_novig      cost
    clv          = market_move - vig_paid            (identity)
"""
import pytest

from src.services.mlb.clv import decompose_clv, summarize_by_type, summarize_clv


class TestDecomposeClv:
    def test_identity_holds(self):
        """clv must equal market_move - vig_paid, exactly."""
        d = decompose_clv(bet_odds=2.50, entry_novig_prob=0.385, closing_novig_prob=0.400)
        assert d["clv"] == pytest.approx(d["market_move"] - d["vig_paid"], abs=1e-12)

    def test_market_move_positive_when_the_close_agrees_with_us(self):
        d = decompose_clv(bet_odds=2.50, entry_novig_prob=0.385, closing_novig_prob=0.400)
        assert d["market_move"] == pytest.approx(0.015)

    def test_market_move_negative_when_the_close_moves_away(self):
        d = decompose_clv(bet_odds=2.50, entry_novig_prob=0.400, closing_novig_prob=0.385)
        assert d["market_move"] == pytest.approx(-0.015)

    def test_vig_paid_is_the_gap_between_price_and_fair(self):
        """Bet at 2.50 (break-even 40%) on a side fairly priced at 38.5%."""
        d = decompose_clv(bet_odds=2.50, entry_novig_prob=0.385, closing_novig_prob=0.400)
        assert d["vig_paid"] == pytest.approx(0.4 - 0.385)

    def test_a_good_price_can_still_lose_to_the_close(self):
        """Market moved our way, but not far enough to cover what we paid."""
        d = decompose_clv(bet_odds=2.50, entry_novig_prob=0.385, closing_novig_prob=0.390)
        assert d["market_move"] > 0
        assert d["clv"] < 0

    def test_enough_movement_overcomes_the_vig(self):
        d = decompose_clv(bet_odds=2.50, entry_novig_prob=0.385, closing_novig_prob=0.430)
        assert d["clv"] > 0

    def test_missing_entry_prob_leaves_movement_unmeasured(self):
        """clv is still computable; market_move is not. Never guess it."""
        d = decompose_clv(bet_odds=2.50, entry_novig_prob=None, closing_novig_prob=0.400)
        assert d["clv"] is not None
        assert d["market_move"] is None and d["vig_paid"] is None

    def test_missing_close_leaves_everything_unmeasured(self):
        d = decompose_clv(bet_odds=2.50, entry_novig_prob=0.385, closing_novig_prob=None)
        assert d["clv"] is None and d["market_move"] is None

    def test_nonsense_odds_are_rejected(self):
        d = decompose_clv(bet_odds=1.0, entry_novig_prob=0.385, closing_novig_prob=0.400)
        assert d["clv"] is None


class TestSummaryIncludesDecomposition:
    def test_reports_movement_alongside_clv(self):
        rows = [
            {"clv": -0.011, "market_move": 0.004, "vig_paid": 0.015},
            {"clv": -0.009, "market_move": 0.006, "vig_paid": 0.015},
        ]
        got = summarize_clv([r["clv"] for r in rows], rows=rows)
        assert got["mean_market_move"] == pytest.approx(0.005)
        assert got["mean_vig_paid"] == pytest.approx(0.015)
        assert got["market_move_positive_rate"] == pytest.approx(1.0)

    def test_movement_verdict_is_independent_of_clv_verdict(self):
        """Skill and profitability are separate findings and must not be merged."""
        rows = [{"clv": -0.01, "market_move": 0.004, "vig_paid": 0.014}] * 40
        got = summarize_clv([r["clv"] for r in rows], rows=rows)
        assert got["mean_clv"] < 0 < got["mean_market_move"]

    def test_absent_decomposition_degrades_gracefully(self):
        got = summarize_clv([-0.01, -0.02])
        assert got["mean_clv"] is not None
        assert got["mean_market_move"] is None


class TestSummarizeByType:
    """The by_type slice must decompose too.

    /mlb/evaluation/clv built its per-market buckets from bare clv floats and
    called summarize_clv without rows=, so every by_type entry reported
    mean_market_move / mean_vig_paid / market_move_positive_rate as null — even
    when all measured picks were that one market and the top level computed
    them fine. The per-market view is exactly where the skill-vs-cost split
    matters, since moneyline, runline and totals pay different vig.

    Testing summarize_clv directly cannot catch that: the bug was the caller
    dropping the rows, not the summariser. So the bucketing itself is the unit
    under test.
    """

    def test_a_single_market_slice_carries_the_decomposition(self):
        rows = [
            {"bet_type": "moneyline", "clv": -0.011, "market_move": 0.004, "vig_paid": 0.015},
            {"bet_type": "moneyline", "clv": -0.009, "market_move": 0.006, "vig_paid": 0.015},
        ]
        got = summarize_by_type(rows)
        assert got["moneyline"]["mean_market_move"] == pytest.approx(0.005)
        assert got["moneyline"]["mean_vig_paid"] == pytest.approx(0.015)
        assert got["moneyline"]["market_move_positive_rate"] == pytest.approx(1.0)

    def test_markets_are_decomposed_independently(self):
        rows = [
            {"bet_type": "moneyline", "clv": -0.005, "market_move": 0.010, "vig_paid": 0.015},
            {"bet_type": "total", "clv": -0.030, "market_move": -0.005, "vig_paid": 0.025},
        ]
        got = summarize_by_type(rows)
        # Different vig and opposite movement — merging them would hide both.
        assert got["moneyline"]["mean_vig_paid"] == pytest.approx(0.015)
        assert got["total"]["mean_vig_paid"] == pytest.approx(0.025)
        assert got["moneyline"]["market_move_positive_rate"] == pytest.approx(1.0)
        assert got["total"]["market_move_positive_rate"] == pytest.approx(0.0)

    def test_a_market_with_no_decomposition_still_reports_clv(self):
        rows = [{"bet_type": "runline", "clv": -0.02, "market_move": None, "vig_paid": None}]
        got = summarize_by_type(rows)
        assert got["runline"]["mean_clv"] == pytest.approx(-0.02)
        assert got["runline"]["mean_market_move"] is None

    def test_a_missing_bet_type_is_bucketed_not_dropped(self):
        rows = [{"bet_type": None, "clv": -0.01, "market_move": 0.003, "vig_paid": 0.013}]
        got = summarize_by_type(rows)
        assert "unknown" in got
        assert got["unknown"]["mean_market_move"] == pytest.approx(0.003)

    def test_bet_type_is_normalised(self):
        rows = [{"bet_type": "MoneyLine", "clv": -0.01, "market_move": 0.003, "vig_paid": 0.013}]
        assert "moneyline" in summarize_by_type(rows)

    def test_no_rows_is_an_empty_mapping_not_an_error(self):
        assert summarize_by_type([]) == {}
