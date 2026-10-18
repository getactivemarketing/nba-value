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


# ---- matchup tag verdicts ----
from statistics import NormalDist  # noqa: E402

MKT = "player_reception_yds"


_seq = iter(range(10 ** 9))


def tp(clv, tags=(), season=2023, market=MKT, result="win", profit=0.9):
    p = pick(clv=clv, season=season, market=market, result=result, profit=profit, player=f"P{next(_seq)}")
    p["tags"] = [{"key": k, "direction": d} for k, d in tags]
    return p


def spread(base, n, tags=(), season=2023, amp=0.01):
    """n picks whose clv alternates base +/- amp (mean base, nonzero variance)."""
    return [tp(base + (amp if i % 2 else -amp), tags, season) for i in range(n)]


def by(verdicts, tag="def_soft", direction="tailwind"):
    return next(v for v in verdicts if v.tag == tag and v.direction == direction)


def test_verdict_proven_needs_200_and_lower_bound_above_zero_and_two_seasons():
    T = [("def_soft", "tailwind")]
    rows = []
    for s in (2023, 2024, 2025):
        rows += spread(0.03, 70, T, s) + spread(0.0, 70, (), s)
    v = by(report.tag_verdicts(rows))
    assert v.n == 210 and v.verdict == "proven" and v.lo > 0 and v.seasons_positive == 3
    assert v.lift == pytest.approx(0.03)


def test_199_tagged_is_neutral_even_with_a_big_lift():
    T = [("def_soft", "tailwind")]
    rows = spread(0.10, 199, T) + spread(0.0, 300)
    v = by(report.tag_verdicts(rows))
    assert v.n == 199 and v.lift > 0.09 and v.verdict == "neutral"


def test_lift_positive_in_one_season_only_is_neutral():
    T = [("def_soft", "tailwind")]
    rows = (spread(0.20, 100, T, 2023) + spread(0.0, 100, (), 2023)
            + spread(-0.001, 60, T, 2024) + spread(0.0, 60, (), 2024)
            + spread(-0.001, 60, T, 2025) + spread(0.0, 60, (), 2025))
    v = by(report.tag_verdicts(rows))
    assert v.n == 220 and v.lo > 0 and v.seasons_positive == 1 and v.verdict == "neutral"


def test_negative_mirror():
    T = [("def_soft", "headwind")]
    rows = []
    for s in (2023, 2024, 2025):
        rows += spread(-0.03, 70, T, s) + spread(0.0, 70, (), s)
    v = by(report.tag_verdicts(rows), direction="headwind")
    assert v.verdict == "negative" and v.hi < 0 and v.seasons_positive == 0 and v.seasons_negative == 3


def test_m_counts_only_pairs_with_200_plus():
    keys = report.TAG_KEYS
    a = [(k, "tailwind") for k in keys]
    b = [(k, "headwind") for k in keys[2:]]
    rows = spread(0.01, 200, a) + spread(0.0, 200, b) + spread(0.0, 150, ()) 
    # a pair with n < 200 must not count toward m
    rows += spread(0.0, 50, [("def_soft", "headwind")])
    vs = report.tag_verdicts(rows)
    v = by(vs)
    ctrl = [r["clv"] for r in rows if not any(t["key"] == "def_soft" for t in r["tags"])]
    tagged = [r["clv"] for r in rows if {"key": "def_soft", "direction": "tailwind"} in r["tags"]]
    from statistics import mean, pvariance
    from math import sqrt
    se = sqrt(pvariance(tagged) / len(tagged) + pvariance(ctrl) / len(ctrl))
    z = (v.hi - v.lo) / 2 / se
    assert round(z, 2) == 2.91 == round(NormalDist().inv_cdf(1 - 0.05 / 28), 2)
    assert by(vs, "def_soft", "headwind").verdict == "neutral"


def test_single_qualifying_pair_uses_z_196_and_no_pairs_is_all_neutral():
    T = [("def_soft", "tailwind")]
    rows = spread(0.01, 200, T) + spread(0.0, 200)
    v = by(report.tag_verdicts(rows))
    se = (0.01 ** 2 / 200 + 0.01 ** 2 / 200) ** 0.5
    assert (v.hi - v.lo) / 2 / se == pytest.approx(NormalDist().inv_cdf(0.975), rel=1e-6)
    few = report.tag_verdicts(spread(0.5, 10, T) + spread(0.0, 10))
    assert few and all(x.verdict == "neutral" for x in few)
    assert report.tag_verdicts([tp(0.1)]) == []


def test_control_excludes_picks_carrying_the_same_key_in_either_direction():
    off = spread(9.0, 4)
    for r in off:
        r["market"] = "player_anytime_td"  # not a def_soft market
    rows = (spread(0.02, 4, [("def_soft", "tailwind")]) + spread(0.5, 4, [("def_soft", "headwind")])
            + spread(0.0, 4, [("usage_up", "tailwind")]) + spread(0.0, 4) + off)
    v = by(report.tag_verdicts(rows))
    # control = usage_up-tagged + untagged on def_soft markets (mean 0): not the headwind or off-market picks
    assert v.lift == pytest.approx(0.02)


def test_null_tags_missing_tags_and_empty_control_do_not_crash():
    legacy = pick()  # no "tags" key at all
    nulled = pick()
    nulled["tags"] = None
    only = tp(0.01, [("def_soft", "tailwind")])
    v = by(report.tag_verdicts([legacy, nulled, only]))
    assert v.n == 1 and v.verdict == "neutral"
    v = by(report.tag_verdicts([only]))
    assert v.lift is None and v.lo is None and v.verdict == "neutral"


def test_report_has_matchup_tags_table():
    rows = spread(0.03, 4, [("def_soft", "tailwind")]) + spread(0.0, 4)
    md = report.build_report(rows, {"run_id": "r1", "snapshots": 1, "credits": 1})
    assert "## Matchup tags" in md
    assert "| Tag | Direction | Picks | CLV lift (CI) | Beat close | ROI | Seasons + | Verdict |" in md
    assert "def_soft" in md
