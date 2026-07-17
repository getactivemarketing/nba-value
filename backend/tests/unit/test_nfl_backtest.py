from src.services.nfl.backtest import grade_spread_pick


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
    assert r is None or r["profit"] == 0


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
