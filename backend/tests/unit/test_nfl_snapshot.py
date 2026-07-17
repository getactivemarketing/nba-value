# tests/unit/test_nfl_snapshot.py
from src.services.nfl.snapshot import build_snapshot, grade_snapshot
from src.services.nfl.value_calculator import NFLValueResult


def _result(market_type, bet_type, team=None, line=None, odds=1.909, value_score=55.0, raw_edge=0.06):
    return NFLValueResult(
        market_type=market_type, bet_type=bet_type, team=team, line=line,
        model_prob=0.56, market_prob=0.50, raw_edge=raw_edge, edge_pct=raw_edge / 0.50 * 100,
        value_score=value_score, confidence="medium", odds_decimal=odds, is_value_bet=True,
        sort_score=10.0)


def test_grade_total_over_win_and_push():
    snap = {"best_total_direction": "over", "best_total_line": 44.0, "best_total_odds": 1.909,
            "best_bet_type": "total", "best_bet_line": 44.0, "best_bet_odds": 1.909,
            "best_bet_team": None}
    g = grade_snapshot(snap, home_score=30, away_score=20, spread_line=3.0, total_line=44.0)
    assert g["actual_total"] == 50 and g["actual_margin"] == 10
    assert g["best_total_result"] == "win"
    assert round(g["best_total_profit"], 1) == 90.9
    assert g["best_bet_result"] == "win"
    # push case
    g2 = grade_snapshot(snap, home_score=22, away_score=22, spread_line=3.0, total_line=44.0)
    assert g2["best_total_result"] == "push" and g2["best_total_profit"] == 0


def test_grade_total_under_win_and_loss():
    snap = {"best_total_direction": "under", "best_total_line": 44.0, "best_total_odds": 1.909,
            "best_bet_type": None, "best_bet_team": None}
    win = grade_snapshot(snap, home_score=17, away_score=13, spread_line=3.0, total_line=44.0)
    assert win["actual_total"] == 30
    assert win["best_total_result"] == "win"
    assert round(win["best_total_profit"], 1) == 90.9
    loss = grade_snapshot(snap, home_score=30, away_score=27, spread_line=3.0, total_line=44.0)
    assert loss["best_total_result"] == "loss" and loss["best_total_profit"] == -100.0


def test_grade_spread_home_cover_win():
    # home pick, line 3.0 -> home must win by MORE than 3 to cover.
    snap = {"best_spread_team": "home", "best_spread_line": 3.0, "best_spread_odds": 1.909}
    g = grade_snapshot(snap, home_score=30, away_score=20, spread_line=3.0, total_line=44.0)
    assert g["actual_margin"] == 10
    assert g["best_spread_result"] == "win"
    assert round(g["best_spread_profit"], 1) == 90.9


def test_grade_spread_home_cover_loss():
    snap = {"best_spread_team": "home", "best_spread_line": 3.0, "best_spread_odds": 1.909}
    g = grade_snapshot(snap, home_score=21, away_score=20, spread_line=3.0, total_line=44.0)
    assert g["actual_margin"] == 1
    assert g["best_spread_result"] == "loss" and g["best_spread_profit"] == -100.0


def test_grade_spread_away_cover_win():
    # away pick, line 3.0 -> away covers iff actual_margin < 3.0 (home wins by less than 3, or away wins).
    snap = {"best_spread_team": "away", "best_spread_line": 3.0, "best_spread_odds": 1.909}
    g = grade_snapshot(snap, home_score=21, away_score=20, spread_line=3.0, total_line=44.0)
    assert g["actual_margin"] == 1
    assert g["best_spread_result"] == "win"
    assert round(g["best_spread_profit"], 1) == 90.9


def test_grade_spread_push():
    snap = {"best_spread_team": "home", "best_spread_line": 3.0, "best_spread_odds": 1.909}
    g = grade_snapshot(snap, home_score=23, away_score=20, spread_line=3.0, total_line=44.0)
    assert g["actual_margin"] == 3
    assert g["best_spread_result"] == "push" and g["best_spread_profit"] == 0

    snap_away = {"best_spread_team": "away", "best_spread_line": 3.0, "best_spread_odds": 1.909}
    g2 = grade_snapshot(snap_away, home_score=23, away_score=20, spread_line=3.0, total_line=44.0)
    assert g2["best_spread_result"] == "push" and g2["best_spread_profit"] == 0


def test_grade_moneyline_home_win():
    snap = {"best_ml_team": "home", "best_ml_odds": 1.6}
    g = grade_snapshot(snap, home_score=24, away_score=20, spread_line=3.0, total_line=44.0)
    assert g["best_ml_result"] == "win"
    assert round(g["best_ml_profit"], 2) == round((1.6 - 1) * 100, 2)


def test_grade_moneyline_away_win():
    snap = {"best_ml_team": "away", "best_ml_odds": 2.5}
    g = grade_snapshot(snap, home_score=17, away_score=24, spread_line=3.0, total_line=44.0)
    assert g["best_ml_result"] == "win"
    assert round(g["best_ml_profit"], 2) == round((2.5 - 1) * 100, 2)


def test_grade_moneyline_loss():
    snap = {"best_ml_team": "home", "best_ml_odds": 1.6}
    g = grade_snapshot(snap, home_score=17, away_score=24, spread_line=3.0, total_line=44.0)
    assert g["best_ml_result"] == "loss" and g["best_ml_profit"] == -100.0


def test_grade_moneyline_tie_is_push():
    # NFL ties are astronomically rare; documented decision: margin==0 grades as a push, not a loss.
    snap = {"best_ml_team": "home", "best_ml_odds": 1.6}
    g = grade_snapshot(snap, home_score=20, away_score=20, spread_line=3.0, total_line=44.0)
    assert g["best_ml_result"] == "push" and g["best_ml_profit"] == 0


def test_best_bet_mirrors_spread_result():
    snap = {"best_spread_team": "home", "best_spread_line": 3.0, "best_spread_odds": 1.909,
            "best_bet_type": "spread", "best_bet_team": "home", "best_bet_line": 3.0,
            "best_bet_odds": 1.909}
    g = grade_snapshot(snap, home_score=30, away_score=20, spread_line=3.0, total_line=44.0)
    assert g["best_bet_result"] == g["best_spread_result"] == "win"
    assert g["best_bet_profit"] == g["best_spread_profit"]


def test_best_bet_mirrors_moneyline_result():
    snap = {"best_ml_team": "away", "best_ml_odds": 2.5,
            "best_bet_type": "moneyline", "best_bet_team": "away", "best_bet_odds": 2.5}
    g = grade_snapshot(snap, home_score=17, away_score=24, spread_line=3.0, total_line=44.0)
    assert g["best_bet_result"] == g["best_ml_result"] == "win"
    assert g["best_bet_profit"] == g["best_ml_profit"]


def test_build_snapshot_maps_bests_and_nulls_missing_markets():
    game = {"game_id": "2024_10_KC_BUF", "home_team": "BUF", "away_team": "KC",
            "kickoff_utc": None, "game_date": None, "snapshot_time": None}
    total = _result("total", "over", line=44.0, odds=1.909, value_score=62.3, raw_edge=0.055)
    bet = _result("total", "over", line=44.0, odds=1.909, value_score=62.3, raw_edge=0.055)
    scored = {"predicted_margin": 2.5, "predicted_total": 47.0,
              "best_spread": None, "best_ml": None, "best_total": total, "best_bet": bet}
    snap = build_snapshot(game, scored)
    assert snap["game_id"] == "2024_10_KC_BUF"
    assert snap["home_team"] == "BUF" and snap["away_team"] == "KC"
    assert snap["predicted_margin"] == 2.5 and snap["predicted_total"] == 47.0
    # totals populated
    assert snap["best_total_direction"] == "over"
    assert snap["best_total_line"] == 44.0
    assert snap["best_total_odds"] == 1.909
    assert snap["best_total_value_score"] == 62.3
    assert snap["best_bet_type"] == "total"
    assert snap["best_bet_line"] == 44.0
    # spread/ML are None -> every field NULL
    assert snap["best_spread_team"] is None and snap["best_spread_line"] is None
    assert snap["best_spread_odds"] is None and snap["best_spread_value_score"] is None
    assert snap["best_ml_team"] is None and snap["best_ml_odds"] is None
    assert snap["snapshot_time"] is not None  # defaults to now() when game omits it
