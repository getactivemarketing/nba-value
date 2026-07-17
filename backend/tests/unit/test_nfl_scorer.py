# tests/unit/test_nfl_scorer.py
from src.services.nfl.scorer import score_game


class _Booster:
    def __init__(self, val): self.val = val
    best_iteration = 1
    def predict(self, X, num_iteration=None):
        return [self.val] * len(X)


def _bundle(pred, cols, std):
    return {"model": _Booster(pred), "feature_cols": cols, "resid_std": std}


def test_score_game_picks_total_as_best_bet_when_only_totals_enabled(monkeypatch):
    from src.services.nfl import value_calculator as vc
    # loosen gate for the test so the in-band total qualifies
    monkeypatch.setattr(vc.settings, "nfl_min_edge", 0.02, raising=False)
    monkeypatch.setattr(vc.settings, "nfl_max_edge", 0.30, raising=False)
    monkeypatch.setattr(vc.settings, "nfl_moderate_threshold", 5.0, raising=False)
    monkeypatch.setattr(vc.settings, "nfl_totals_in_best_bet", True, raising=False)
    monkeypatch.setattr(vc.settings, "nfl_spread_in_best_bet", False, raising=False)
    monkeypatch.setattr(vc.settings, "nfl_ml_in_best_bet", False, raising=False)

    feat = {c: 0.0 for c in ["off_epa_diff", "def_epa_diff", "pass_epa_diff",
            "rush_epa_diff", "success_rate_diff", "pace_diff", "power_diff",
            "rest_diff", "is_divisional", "is_primetime", "spread_line",
            "off_epa_sum", "pace_sum", "is_dome", "wind_mph", "temp_f", "total_line"]}
    feat["spread_line"] = 3.0
    feat["total_line"] = 44.0
    # model predicts total 52 (well over 44) -> strong over edge; margin ~ line (no spread edge)
    mov = _bundle(3.0, [c for c in feat if c in (
        "off_epa_diff","def_epa_diff","pass_epa_diff","rush_epa_diff","success_rate_diff",
        "pace_diff","power_diff","rest_diff","is_divisional","is_primetime","spread_line")], 13.0)
    tot = _bundle(52.0, ["off_epa_sum","pace_sum","pass_epa_diff","is_dome",
                         "wind_mph","temp_f","total_line"], 13.7)
    markets = [
        {"market_type": "spread", "line": 3.0, "home_odds": 1.909, "away_odds": 1.909},
        {"market_type": "moneyline", "line": None, "home_odds": 1.6, "away_odds": 2.5},
        {"market_type": "total", "line": 44.0, "over_odds": 1.909, "under_odds": 1.909},
    ]
    out = score_game(feat, markets, mov, tot)
    assert round(out["predicted_total"]) == 52
    assert out["best_total"] is not None and out["best_total"].bet_type == "over"
    assert out["best_bet"] is not None and out["best_bet"].market_type == "total"
    # spread edge ~0 -> best_spread may be None or not the best_bet
    assert out["best_bet"].market_type == "total"
