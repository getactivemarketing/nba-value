"""Runline paused out of best_bet (2026-07-21): a sign-pairing bug inflated
runline picks (labeled +1.5, priced -1.5). Until the scorer is fixed, runline
must not reach best_bet via either find_best_bet or the resolve_best_bet
defense-in-depth fallback."""

from src.services.mlb.value_calculator import MLBValueCalculator
from src.tasks.mlb_scheduler import resolve_best_bet


def calc(market_type, model_prob, market_prob=0.50):
    bet_type = {"runline": "home_rl", "moneyline": "home_ml", "total": "over"}[market_type]
    return MLBValueCalculator.calculate_value(
        market_type=market_type, bet_type=bet_type, model_prob=model_prob,
        market_prob=market_prob, odds_decimal=2.0,
        team="NYY" if market_type != "total" else None,
        line=1.5 if market_type == "runline" else None,
    )


def test_find_best_bet_excludes_runline_when_paused():
    rl = calc("runline", 0.70)   # strong runline
    ml = calc("moneyline", 0.62)  # weaker ML
    # include_runline=False -> runline dropped, ML chosen despite lower score
    assert MLBValueCalculator.find_best_bet([rl, ml], include_runline=False) is ml


def test_find_best_bet_none_when_only_runline_and_paused():
    rl = calc("runline", 0.70)
    assert MLBValueCalculator.find_best_bet([rl], include_runline=False) is None


def test_find_best_bet_keeps_runline_by_default():
    rl = calc("runline", 0.70)
    ml = calc("moneyline", 0.55)
    assert MLBValueCalculator.find_best_bet([rl, ml]) is rl


def test_resolve_best_bet_does_not_fall_back_to_runline_when_paused():
    total = calc("total", 0.70)
    ml = calc("moneyline", 0.61)
    rl = calc("runline", 0.80)  # highest score, but paused
    out = resolve_best_bet(total, ml, rl, totals_allowed=False, runline_allowed=False)
    assert out is ml


def test_resolve_best_bet_strips_runline_best_bet_when_paused():
    rl = calc("runline", 0.80)
    ml = calc("moneyline", 0.61)
    # scorer shouldn't hand a runline best_bet while paused, but defense must strip it
    out = resolve_best_bet(rl, ml, rl, totals_allowed=False, runline_allowed=False)
    assert out is ml


def test_resolve_best_bet_runline_allowed_preserves_old_behavior():
    rl = calc("runline", 0.62)
    assert resolve_best_bet(rl, None, rl, totals_allowed=False, runline_allowed=True) is rl
