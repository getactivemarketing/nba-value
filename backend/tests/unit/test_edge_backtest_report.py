import pytest

from src.edge.backtest import report


def pick(label="t6h", ev=0.03, clv=0.01, result="win", profit=0.9, market="player_receptions",
         book="fanduel", season=2023, player="A", pin_ev=0.03, verify=False, event_id="e1", side="over", line=4.5):
    return {"season": season, "week": 1, "event_id": event_id, "label": label, "market": market,
            "player": player, "side": side, "line": line, "book": book, "price": -110, "ev": ev, "clv": clv,
            "result": result, "profit": profit, "pin_ev": pin_ev, "verify": verify}


def test_bucket_edges():
    assert [report.bucket(x) for x in (0.02, 0.0399, 0.04, 0.0599, 0.06, 0.3)] == \
        ["2–4%", "2–4%", "4–6%", "4–6%", "6%+", "6%+"]


def test_first_flag_counts_a_pick_once_at_its_earliest_label():
    rows = [pick("t1h"), pick("t24h"), pick("t6h"), pick("t6h", player="B")]
    got = report.first_flag(rows)
    assert sorted((r["player"], r["label"]) for r in got) == [("A", "t24h"), ("B", "t6h")]


def test_metrics_roi_excludes_voids_and_counts_pushes_as_staked():
    rows = [pick(result="win", profit=1.0), pick(result="loss", profit=-1.0), pick(result="push", profit=0.0),
            pick(result="void", profit=None, clv=None), pick(result=None, profit=None, clv=-0.02)]
    m = report.metrics(rows)
    assert m["picks"] == 5 and m["graded"] == 3
    assert m["win_pct"] == pytest.approx(0.5)
    assert m["roi"] == pytest.approx(0.0)
    assert m["avg_clv"] == pytest.approx((0.01 * 3 - 0.02) / 4)
    assert m["beat_close"] == pytest.approx(3 / 4)
    assert m["small"] is True and m["roi_ci"] is not None


def test_metrics_on_empty_rows():
    m = report.metrics([])
    assert m["picks"] == 0 and m["roi"] is None and m["avg_clv"] is None and m["win_pct"] is None


def test_recommendations_follow_the_fixed_rules():
    good = [pick(market="player_receptions", book="fanduel", ev=0.03, clv=0.02, player=f"p{i}")
            for i in range(200)]
    bad = [pick(market="player_pass_tds", book="bovada", ev=0.07, clv=-0.01, player=f"q{i}")
           for i in range(250)]
    early = [pick(label="t24h", market="player_rush_yds", book="betmgm", ev=0.05, clv=0.05, player=f"r{i}")
             for i in range(10)]
    text = "\n".join(report.recommend(good + bad + early))
    assert "player_receptions" in text and "player_pass_tds" not in text.split("Markets to show")[1].split("\n")[0]
    assert "fanduel" in text
    assert "EV ≥ 2%" in text
    assert "t24h" in text


def test_build_report_has_every_section_and_flags_small_samples():
    rows = [pick(), pick(label="t24h", season=2025, book="pinnacle", pin_ev=None)]
    md = report.build_report(rows, {"run_id": "r1", "snapshots": 10, "credits": 2000})
    for heading in ("## Overall", "## By season", "## By market", "## By book", "## By EV bucket",
                    "## By lead time", "## Season × lead time", "## Pinnacle-only consensus",
                    "## Recommendations"):
        assert heading in md, heading
    assert "small sample" in md and "r1" in md and "2,000" in md
