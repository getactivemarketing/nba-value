from datetime import date, datetime, timedelta, timezone

import pytest

from src.edge.backtest import matchup_gate as mg
from src.edge.teams import ABBR_BY_NAME

ABBRS = sorted(ABBR_BY_NAME.values())
NAME = {v: k for k, v in ABBR_BY_NAME.items()}


def test_gate_rule_exact():
    assert mg.gate([0.02] * 250, [0.015] * 250)["pass"]
    assert not mg.gate([0.02] * 150, [0.0] * 250)["pass"]  # n < 200
    assert not mg.gate([0.010] * 250, [0.0130] * 250)["pass"]  # trails by 0.3pp


def test_gate_boundaries():
    assert mg.gate([0.010] * 200, [0.0124] * 200)["pass"]  # trails by 0.24pp: inside tolerance
    assert not mg.gate([0.010] * 199, [0.0] * 200)["pass"]  # one bet short
    varied = [0.02, -0.02] * 100  # mean 0, CI straddles 0
    assert not mg.gate(varied, [0.0] * 200)["pass"]
    out = mg.gate([0.01 + 0.001 * (i % 3) for i in range(250)], [0.0] * 250)
    assert out["pass"] and out["n_fav"] == 250 and out["fav_ci95"][0] > 0


def test_gate_min_n_parameter():
    assert mg.gate([0.02] * 5, [0.0] * 5, min_n=5)["pass"]
    assert not mg.gate([0.02] * 5, [0.0] * 5)["pass"]  # default stays 200


def test_gate_empty_groups_fail():
    assert not mg.gate([], [0.01] * 5)["pass"]
    assert not mg.gate([0.02] * 300, [])["pass"]


def _opp(first_seen, book="fanduel", clv=0.01, **kw):
    base = dict(event_id="e1", market="player_points", player="A B", side="over", line=20.5,
                book=book, clv=clv, first_seen=first_seen)
    return {**base, **kw}


def test_dedupe_keeps_earliest_sighting():
    t = datetime(2026, 11, 1, 12, tzinfo=timezone.utc)
    rows = [_opp(t + timedelta(hours=2), "betmgm", 0.05), _opp(t, "fanduel", 0.01),
            _opp(t + timedelta(hours=1), "dk", 0.03),
            _opp(t, "fanduel", 0.02, line=21.5), _opp(t, "fanduel", 0.02, side="under")]
    out = mg.dedupe(rows)
    assert len(out) == 3
    same = [r for r in out if r["line"] == 20.5 and r["side"] == "over"][0]
    assert same["book"] == "fanduel" and same["clv"] == 0.01


def test_position_coverage_is_minutes_weighted():
    games = [{"minutes": 30, "position": "F"}, {"minutes": 10, "position": None},
             {"minutes": 20, "position": "G"}, {"minutes": None, "position": None}]
    assert mg.position_coverage(games) == pytest.approx(50 / 60)
    assert mg.position_coverage([]) is None


def _games(extra_after=False):
    """30 teams in 15 pairs; each pair plays 5 games Nov 1-5. A scores 10+i with F, G scores 5."""
    rows, gid = [], 0
    for k in range(15):
        a, b = ABBRS[2 * k], ABBRS[2 * k + 1]
        for day in range(5):
            gid += 1
            for team, opp, i in ((a, b, 2 * k), (b, a, 2 * k + 1)):
                # pair 14 (the defender's partner) scores a lot: its opponent allows the most
                f_pts = 50 if team == ABBRS[28] else 10 + i
                for pid, pos, pts in ((1000 + i, "F", f_pts), (2000 + i, "G", 5)):
                    rows.append(dict(bdl_player_id=pid, bdl_game_id=gid, game_date=date(2026, 11, 1 + day),
                                     team=team, opponent=opp, minutes=30, pts=pts, reb=0, ast=0, fg3m=0,
                                     tov=1, fga=10, fta=2, oreb=1, position=pos))
    return rows


def _event_bet(side="over", hero="A", day=date(2026, 11, 20)):
    # hero team ABBRS[0] vs defender ABBRS[29] (partner of ABBRS[28], who scores 50)
    return dict(event_id="e1", market="player_points", player="Hero", side=side, line=20.5, book="x",
                clv=0.02, first_seen=datetime(2026, 11, 19, tzinfo=timezone.utc), result=None, profit_units=None,
                home_team=NAME[ABBRS[0]], away_team=NAME[ABBRS[29]],
                commence_time=datetime(day.year, day.month, day.day, 23, tzinfo=timezone.utc),
                bdl_player_id=1000, position="F")


def test_classify_favorable_over_vs_weak_defender_and_not_under():
    out = mg.classify([_event_bet("over"), _event_bet("under")], _games())
    by_side = {o["side"]: o for o in out}
    assert by_side["over"]["favorable"] is True
    assert by_side["under"]["favorable"] is False and by_side["under"]["why_not"] == "not favorable"


def test_classify_uses_only_games_before_the_bet_date():
    # Games played ON the bet date (or later) must not change the answer: with the bet on Nov 3,
    # only 2 prior games exist for the hero, so history is too thin to be favorable.
    early = _event_bet("over", day=date(2026, 11, 3))
    out = mg.classify([early], _games())
    assert out[0]["favorable"] is False


def test_classify_unclassifiable_bets_are_rest():
    nopos = {**_event_bet(), "position": None}
    ghost = {**_event_bet(), "bdl_player_id": 99999}
    wrong_team = {**_event_bet(), "home_team": NAME[ABBRS[5]], "away_team": NAME[ABBRS[6]]}
    out = mg.classify([nopos, ghost, wrong_team], _games())
    assert sorted(o["why_not"] for o in out) == ["no position", "no prior games", "team not in event"]
    assert not any(o["favorable"] for o in out)


def test_nfl_gate_passes_on_nonnegative_mean():
    assert mg.gate_nfl([0.01, 0.0, 0.02])["pass"]
    assert mg.gate_nfl([0.0] * 3)["pass"]  # exactly 0 passes
    assert mg.gate_nfl([0.02] * 5)["n_fav"] == 5  # no minimum n


def test_nfl_gate_fails_on_negative_mean_or_empty():
    assert not mg.gate_nfl([0.01, -0.03, -0.01])["pass"]
    assert not mg.gate_nfl([])["pass"]


def test_nfl_evaluate_splits_on_def_soft_tags(monkeypatch):
    t0 = datetime(2026, 11, 1, tzinfo=timezone.utc)
    soft = [{"key": "def_soft", "direction": "tailwind", "detail": ""}]
    head = soft + [{"key": "questionable", "direction": "headwind", "detail": ""}]

    def row(player, tags, clv):
        return _opp(t0, player=player, tags=tags, clv=clv, result="win", profit_units=0.9,
                    market="player_pass_yds")
    rows = [row("A", soft, 0.02), row("B", head, -0.05), row("C", [], -0.01)]
    monkeypatch.setattr(mg, "load_nfl_opportunities", lambda conn, season: rows)
    out = mg.evaluate_nfl(None, 2026)
    assert [b["player"] for b in out["fav"]] == ["A"] and len(out["rest"]) == 2
    assert out["gate"]["pass"]
    assert "Verdict: PASS" in mg.render_report_nfl(out, 2026, date(2026, 11, 20))


def test_report_mentions_position_coverage_and_verdict():
    fav = [{**_event_bet(), "favorable": True, "why_not": None, "result": "win", "profit_units": 0.9,
            "clv": 0.02 + i * 0.001, "player": f"P{i}"} for i in range(3)]
    rest = [{**_event_bet(), "favorable": False, "why_not": "no position", "result": "loss",
             "profit_units": -1.0, "clv": 0.0, "player": f"R{i}"} for i in range(3)]
    result = {"gate": mg.gate([b["clv"] for b in fav], [b["clv"] for b in rest]), "fav": fav, "rest": rest,
              "raw_rows": 9, "bets": 6, "position_coverage": 0.93, "why_not": {"no position": 3}}
    md = mg.render_report(result, 2026, date(2026, 11, 20))
    assert "Position coverage: 93.00%" in md and "Verdict: FAIL" in md and "| Favorable | 3 |" in md
