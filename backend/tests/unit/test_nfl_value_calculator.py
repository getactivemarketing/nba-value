import math
from src.services.nfl.value_calculator import NFLValueCalculator, NFLValueResult


def calc(market_type="total", model_prob=0.58, market_prob=0.50, odds=1.909, conf=0.6):
    bt = {"total": "over", "spread": "home_spread", "moneyline": "home_ml"}[market_type]
    return NFLValueCalculator.calculate_value(
        market_type=market_type, bet_type=bt, model_prob=model_prob,
        market_prob=market_prob, odds_decimal=odds, model_confidence=conf)


def test_edge_and_scores_follow_mlb_formula():
    r = calc(model_prob=0.58, market_prob=0.50)
    assert round(r.raw_edge, 3) == 0.08
    assert round(r.edge_pct, 1) == 16.0            # 0.08/0.50*100
    # value_score uses tanh of regressed edge -> never pegs at 100
    assert 0 < r.value_score < 100
    # sort_score is unclamped edge_pct*conf*market (total market_mult 0.90)
    conf_mult = 0.8 + 0.6 * 0.4                     # 1.04
    assert round(r.sort_score, 2) == round(16.0 * conf_mult * 0.90, 2)


def test_edge_ceiling_rejects_overconfident_pick(monkeypatch):
    # The ceiling MECHANISM: with an active ceiling (0.12), a raw_edge 0.20 pick
    # is rejected. (The shipped default is 0.99 = no ceiling, since calibration
    # handles overconfidence — so we set an explicit ceiling to test the mechanism.)
    from src.services.nfl import value_calculator as vc
    monkeypatch.setattr(vc.settings, "nfl_max_edge", 0.12, raising=False)
    r = calc(model_prob=0.70, market_prob=0.50)
    assert r.raw_edge >= 0.12
    assert r.is_value_bet is False


def test_in_band_pick_qualifies():
    r = calc(model_prob=0.57, market_prob=0.50)   # raw_edge 0.07 >= floor 0.05, under ceiling
    assert r.is_value_bet is True


def test_find_best_bet_respects_enabled_markets():
    over = calc("total", model_prob=0.57)          # qualifies, total
    spread = NFLValueCalculator.calculate_value(
        "spread", "home_spread", 0.60, 0.50, 1.909, model_confidence=0.6)  # bigger edge
    # Only totals enabled -> best_bet is the total even though spread edge is larger
    best = NFLValueCalculator.find_best_bet([over, spread], enabled_market_types={"total"})
    assert best is not None and best.market_type == "total"
    # Spread enabled too -> the larger-edge spread wins by sort_score
    best2 = NFLValueCalculator.find_best_bet([over, spread], enabled_market_types={"total", "spread"})
    assert best2.market_type == "spread"
