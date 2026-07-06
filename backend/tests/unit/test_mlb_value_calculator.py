"""Unit tests for the retuned MLBValueCalculator (2026-07-06 spec).

Key invariants:
- Qualification gate identical to pre-retune formula, plus MAX_EDGE_PCT blowup cap.
- value_score (display) uses tanh of the market-regressed edge -> no saturation at 100.
- sort_score (selection) is the unclamped edge_pct * confidence * market multipliers.
"""

import math

import pytest

from src.services.mlb.value_calculator import MLBValueCalculator, MLBValueResult


def calc(
    market_type="runline",
    model_prob=0.60,
    market_prob=0.50,
    odds=2.0,
    conf=0.5,
):
    bet_type = {"runline": "home_rl", "moneyline": "home_ml", "total": "over"}[market_type]
    return MLBValueCalculator.calculate_value(
        market_type=market_type,
        bet_type=bet_type,
        model_prob=model_prob,
        market_prob=market_prob,
        odds_decimal=odds,
        team="NYY" if market_type != "total" else None,
        line=1.5 if market_type == "runline" else None,
        model_confidence=conf,
    )


class TestQualificationGate:
    def test_min_edge_pick_still_qualifies(self):
        # raw_edge 0.10, edge_pct 20 -> legacy gate 20*4*1.0 = 80 >= 55
        result = calc(model_prob=0.60, market_prob=0.50)
        assert result.is_value_bet is True

    def test_below_min_edge_rejected(self):
        # raw_edge 0.09 -> gate score 72 passes but MIN_EDGE filter rejects
        result = calc(model_prob=0.59, market_prob=0.50)
        assert result.is_value_bet is False

    def test_blowup_capped(self):
        # raw_edge 0.50, edge_pct 111 > MAX_EDGE_PCT 80 -> rejected
        result = calc(model_prob=0.95, market_prob=0.45, odds=2.22)
        assert result.edge_pct > MLBValueCalculator.MAX_EDGE_PCT
        assert result.is_value_bet is False

    def test_edge_pct_at_cap_boundary_qualifies(self):
        # edge_pct exactly 80 (raw 0.40 / market 0.50) passes (<=)
        result = calc(model_prob=0.90, market_prob=0.50)
        assert result.edge_pct == pytest.approx(80.0)
        assert result.is_value_bet is True


class TestDisplayScore:
    def test_moderate_edge_not_saturated(self):
        # edge_pct 20 -> blended 10 -> 100*tanh(0.5) = 46.2 (conf 0.5 -> mult 1.0)
        result = calc(model_prob=0.60, market_prob=0.50)
        assert result.value_score == pytest.approx(100 * math.tanh(0.5), abs=0.1)
        assert result.value_score < 100

    def test_huge_edge_stays_below_100_without_bonus(self):
        # edge_pct 60 -> blended 30 -> 100*tanh(1.5) = 90.5; adjusted prob
        # 0.50 + 0.30*0.5 = 0.65, not > 0.65 -> no favorite bonus
        result = calc(model_prob=0.80, market_prob=0.50)
        assert result.value_score == pytest.approx(100 * math.tanh(1.5), abs=0.1)

    def test_ml_multiplier_applies_to_display(self):
        rl = calc(market_type="runline", model_prob=0.60, market_prob=0.50)
        ml = calc(market_type="moneyline", model_prob=0.60, market_prob=0.50)
        assert ml.value_score == pytest.approx(rl.value_score * 0.95, abs=0.1)


class TestSortScore:
    def test_sort_score_is_unclamped(self):
        # edge_pct 60 with conf 0.5 (mult 1.0) -> sort_score 60, far above
        # what the old clamped score could express
        result = calc(model_prob=0.80, market_prob=0.50)
        assert result.sort_score == pytest.approx(60.0, abs=0.01)

    def test_find_best_value_ranks_by_sort_score_not_display(self):
        # Both would have clamped to 100 under the old formula; the bigger
        # edge must win regardless of list order.
        smaller = calc(model_prob=0.75, market_prob=0.50)   # edge_pct 50
        bigger = calc(model_prob=0.80, market_prob=0.50)    # edge_pct 60
        best = MLBValueCalculator.find_best_value([smaller, bigger])
        assert best is bigger
        best = MLBValueCalculator.find_best_value([bigger, smaller])
        assert best is bigger

    def test_find_best_value_still_requires_is_value_bet(self):
        blowup = calc(model_prob=0.95, market_prob=0.45, odds=2.22)  # capped
        ok = calc(model_prob=0.60, market_prob=0.50)
        best = MLBValueCalculator.find_best_value([blowup, ok])
        assert best is ok
