from src.services.nfl.backtest import grade_spread_pick, grade_total_pick


def test_home_cover_win_and_loss():
    # Model predicts home by 7, line home -3 (spread_line=3). Big edge -> bet home.
    # Actual margin 10 -> home covers -3 -> win.
    r = grade_spread_pick(pred_mov=7.0, resid_std=13.0, spread_line=3.0,
                          actual_margin=10, threshold=0.03)
    assert r is not None and r["side"] == "home" and r["won"] is True
    assert round(r["profit"], 1) == 90.9

    # Actual margin 1 -> home does NOT cover -3 (1 < 3) -> loss.
    r2 = grade_spread_pick(pred_mov=7.0, resid_std=13.0, spread_line=3.0,
                           actual_margin=1, threshold=0.03)
    assert r2["won"] is False and r2["profit"] == -100


def test_no_edge_returns_none():
    # Model predicts home by 3, line home -3 -> fair, no edge.
    r = grade_spread_pick(pred_mov=3.0, resid_std=13.0, spread_line=3.0,
                          actual_margin=5, threshold=0.05)
    assert r is None


def test_push_returns_none_or_zero():
    # Predicted home by 7 (bet home), line 3, actual margin exactly 3 -> push.
    r = grade_spread_pick(pred_mov=7.0, resid_std=13.0, spread_line=3.0,
                          actual_margin=3, threshold=0.03)
    assert r is None


def test_total_over_win():
    # pred_total 52 clearly above line 44 (std 10) -> over, edge ~0.288.
    # grade_total_pick calls mov_to_total_prob(pred_total, 0.0, total_line, total_std);
    # a buggy 3-arg call `mov_to_total_prob(pred_total, total_line, total_std)` would
    # instead sum 52+44=96 into the CDF (with total_std silently defaulting to 12.0
    # and the real total_std landing in the total_line slot), pushing edge to ~0.4999
    # instead of ~0.288 -- pin the correct probability, not just the side.
    r = grade_total_pick(pred_total=52.0, total_std=10.0, total_line=44.0,
                         actual_total=47.0, threshold=0.05)
    assert r is not None and r["side"] == "over" and r["won"] is True
    assert round(r["profit"], 1) == 90.9
    assert abs(r["edge"] - 0.2881) < 0.001  # buggy call would give ~0.4999


def test_total_under_win():
    # pred_total 36 clearly below line 44 (std 10) -> under, edge ~-0.288.
    # Under the buggy 3-arg call, home+away sums to 36+44=80, which is still far
    # ABOVE the (misplaced) line/std inputs and flips the side to "over" entirely --
    # this is the strongest regression pin: side itself flips under the bug.
    r = grade_total_pick(pred_total=36.0, total_std=10.0, total_line=44.0,
                         actual_total=40.0, threshold=0.05)
    assert r is not None and r["side"] == "under" and r["won"] is True
    assert round(r["profit"], 1) == 90.9
    assert abs(r["edge"] + 0.2881) < 0.001  # buggy call flips this to positive ("over")


def test_total_no_edge_returns_none():
    # pred_total == total_line -> fair line, no edge.
    r = grade_total_pick(pred_total=44.0, total_std=10.0, total_line=44.0,
                         actual_total=50.0, threshold=0.05)
    assert r is None


def test_total_push_returns_none():
    # Pred clears the edge threshold (52 vs line 44), but actual_total == total_line -> push.
    r = grade_total_pick(pred_total=52.0, total_std=10.0, total_line=44.0,
                         actual_total=44.0, threshold=0.05)
    assert r is None


def test_reliability_buckets_by_edge():
    from src.services.nfl.backtest import _reliability
    picks = [
        {"side": "home", "edge": 0.04, "won": True, "profit": 90.9},
        {"side": "home", "edge": 0.04, "won": False, "profit": -100},
        {"side": "home", "edge": 0.12, "won": True, "profit": 90.9},
        None,
    ]
    rel = _reliability(picks)
    band_004 = next(b for b in rel if b["edge_band"] == "0.03-0.06")
    band_010 = next(b for b in rel if b["edge_band"] == "0.10-0.15")
    assert band_004["n"] == 2 and band_004["win_pct"] == 50.0
    assert band_010["n"] == 1 and band_010["win_pct"] == 100.0
