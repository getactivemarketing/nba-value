# backend/tests/unit/test_edge_grading.py
from datetime import datetime, timedelta, timezone

import pytest

from src.edge import engine, grading
from src.edge.quotes import Quote

TIP = datetime(2026, 10, 20, 23, 0, tzinfo=timezone.utc)


def test_over_under():
    assert grading.grade_over_under(26, 25.5, "over") == "win"
    assert grading.grade_over_under(26, 25.5, "under") == "loss"
    assert grading.grade_over_under(25, 25.0, "over") == "push"


def test_spread_home_perspective_line():
    # away +5.5 is stored as line -5.5 (home -5.5). Home wins by 3 -> away covers.
    assert grading.grade_game("spreads", "away", -5.5, 103, 100) == "win"
    assert grading.grade_game("spreads", "home", -5.5, 103, 100) == "loss"
    assert grading.grade_game("spreads", "home", -3.0, 103, 100) == "push"


def test_h2h_and_totals():
    assert grading.grade_game("h2h", "home", 0.0, 110, 100) == "win"
    assert grading.grade_game("h2h", "away", 0.0, 110, 100) == "loss"
    assert grading.grade_game("totals", "over", 215.5, 110, 106) == "win"


def test_profit_units():
    assert grading.profit_units("win", 150) == pytest.approx(1.5)
    assert grading.profit_units("win", -200) == pytest.approx(0.5)
    assert grading.profit_units("loss", 150) == -1.0
    assert grading.profit_units("void", 150) == 0.0


def _hq(book, side, price, at):
    return Quote("e1", "player_points", "A", side, 25.5, book, price, None, at)


def test_a_line_pulled_hours_before_tip_is_not_in_the_close():
    at = TIP - timedelta(minutes=5)
    quotes = []
    for b in ("pinnacle", "fanduel"):
        quotes += [_hq(b, "over", -120, at), _hq(b, "under", 100, at)]
    pulled = TIP - timedelta(hours=2)
    quotes += [_hq("draftkings", "over", -120, pulled), _hq("draftkings", "under", 100, pulled)]
    opp = {"event_id": "e1", "market": "player_points", "player": "A", "side": "over",
           "line": 25.5, "book": "betmgm", "price": -110}
    prob, clv, note = grading.closing_for(quotes, opp, TIP)
    assert prob is None and note == "thin close (2 books, sharp=True)"


def test_closing_excludes_own_book_and_uses_last_pre_tip_quote():
    """Grading a pick against a close that contains its own price flatters it (an exchange
    weighs 2x), so the close is built from the other books only."""
    at = TIP - timedelta(minutes=5)
    hist = []
    for b in ("pinnacle", "fanduel", "draftkings"):
        hist += [_hq(b, "over", -120, at), _hq(b, "under", 100, at)]
    hist += [_hq("novig", "over", 130, at), _hq("novig", "under", -150, at)]  # own book: ignored
    hist.append(_hq("pinnacle", "over", 150, TIP - timedelta(hours=5)))  # older, superseded
    opp = {"event_id": "e1", "market": "player_points", "player": "A", "side": "over",
           "line": 25.5, "book": "novig", "price": 130}
    prob, clv, note = grading.closing_for(hist, opp, TIP)
    others = [q for q in hist if q.book != "novig" and q.observed_at == at]
    expected = engine.fair_prob(engine.group(others)[("e1", "player_points", "A", 25.5)],
                                "player_points", "over", exclude_book=None).prob
    assert note is None and prob == pytest.approx(expected)
    assert clv == pytest.approx(prob - 1 / 2.3)


def test_thin_close_has_no_clv_and_says_why():
    at = TIP - timedelta(minutes=5)
    hist = [_hq("fanduel", "over", -110, at), _hq("fanduel", "under", -110, at)]
    opp = {"event_id": "e1", "market": "player_points", "player": "A", "side": "over",
           "line": 25.5, "book": "fanduel", "price": -110}
    prob, clv, note = grading.closing_for(hist, opp, TIP)
    assert prob is None and clv is None and note.startswith("thin close")


def test_prop_result_void_only_once_team_box_scores_exist():
    opp = {"market": "player_points", "side": "over", "line": 25.5}
    assert grading.prop_result(opp, None, team_has_box=False) is None     # not ingested yet
    assert grading.prop_result(opp, None, team_has_box=True) == "void"    # DNP
    assert grading.prop_result(opp, {"pts": 30, "reb": 0, "ast": 0, "fg3m": 0, "minutes": 33.0}, True) == "win"
    assert grading.prop_result(opp, {"pts": 0, "reb": 0, "ast": 0, "fg3m": 0, "minutes": 0.0}, True) == "void"
