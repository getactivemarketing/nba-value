"""Defense-in-depth: a total must never be written to snapshot best_bet_*
fields while totals_in_best_bet is off, even if the scorer produced one
(e.g. stale prediction object or env drift)."""

from src.services.mlb.value_calculator import MLBValueCalculator
from src.tasks.mlb_scheduler import resolve_best_bet


def calc(market_type, model_prob, market_prob=0.50):
    bet_type = {"runline": "home_rl", "moneyline": "home_ml", "total": "over"}[market_type]
    return MLBValueCalculator.calculate_value(
        market_type=market_type,
        bet_type=bet_type,
        model_prob=model_prob,
        market_prob=market_prob,
        odds_decimal=2.0,
        team="NYY" if market_type != "total" else None,
        line=1.5 if market_type == "runline" else None,
        model_confidence=0.5,
    )


def test_non_total_passes_through():
    rl = calc("runline", 0.62)
    assert resolve_best_bet(rl, None, rl, totals_allowed=False) is rl


def test_total_replaced_by_best_of_ml_rl():
    total = calc("total", 0.70)
    ml = calc("moneyline", 0.61)
    rl = calc("runline", 0.62)
    assert resolve_best_bet(total, ml, rl, totals_allowed=False) is rl


def test_total_kept_when_allowed():
    total = calc("total", 0.70)
    ml = calc("moneyline", 0.61)
    assert resolve_best_bet(total, ml, None, totals_allowed=True) is total


def test_total_with_no_qualifying_fallback_returns_none():
    total = calc("total", 0.70)
    weak_ml = calc("moneyline", 0.55)  # raw_edge 0.05 < MIN_EDGE
    assert resolve_best_bet(total, weak_ml, None, totals_allowed=False) is None


def test_none_passes_through():
    assert resolve_best_bet(None, None, None, totals_allowed=False) is None
