"""best_bet is graded from its OWN frozen columns, not by copying components.

Before this, grading copied the component result:
    best_bet_result = best_ml_result   # only set `if snapshot.best_ml_team:`
so a snapshot whose best_ml_team was NULL (see
test_mlb_best_ml_across_books.py) could never be graded — and since the
grader's "already done" sentinel (actual_winner) was stamped in the same
pass, the row was stranded permanently. 109 snapshots ended up in that state.

Grading from best_bet_team/line/odds also fixes a second defect: profit was
computed from the *component's* odds (a different book's price) rather than
the price actually recorded — and advertised in the SMS — for the pick.
"""
import pytest

from src.tasks.mlb_scheduler import grade_best_bet


def _ml(team, home_score, away_score, odds=2.50, home_team="HOM"):
    return grade_best_bet(
        best_bet_type="moneyline", best_bet_team=team, best_bet_line=None,
        best_bet_odds=odds, best_total_direction=None,
        home_team=home_team, home_score=home_score, away_score=away_score,
    )


def test_moneyline_win_pays_the_recorded_price():
    # HOM wins 5-3; pick HOM at 2.50 -> +150 on a $100 stake.
    assert _ml("HOM", 5, 3) == ("win", 150.0)


def test_moneyline_loss():
    assert _ml("HOM", 3, 5) == ("loss", -100.0)


def test_moneyline_away_pick_win():
    assert _ml("AWY", 3, 5, odds=3.00) == ("win", 200.0)


def test_profit_uses_best_bet_odds_not_a_component_price():
    """The pick was recorded at 2.20; grading must pay 2.20."""
    result, profit = _ml("HOM", 4, 1, odds=2.20)
    assert result == "win"
    assert profit == pytest.approx(120.0)


def test_runline_push_when_adjusted_equals_opponent():
    # HOM -1.5 ... using line -1.5 from the bet team's perspective: 5 + (-1.5)
    # = 3.5 vs 3 -> win. Use an exact push with a whole line instead.
    assert grade_best_bet(
        best_bet_type="runline", best_bet_team="HOM", best_bet_line=-2.0,
        best_bet_odds=2.0, best_total_direction=None,
        home_team="HOM", home_score=5, away_score=3,
    ) == ("push", 0.0)


def test_runline_win_and_loss():
    win = grade_best_bet(
        best_bet_type="runline", best_bet_team="AWY", best_bet_line=1.5,
        best_bet_odds=1.90, best_total_direction=None,
        home_team="HOM", home_score=4, away_score=3,
    )
    assert win == ("win", pytest.approx(90.0))
    loss = grade_best_bet(
        best_bet_type="runline", best_bet_team="HOM", best_bet_line=-1.5,
        best_bet_odds=2.10, best_total_direction=None,
        home_team="HOM", home_score=4, away_score=3,
    )
    assert loss == ("loss", -100.0)


def test_total_over_and_under():
    over = grade_best_bet(
        best_bet_type="total", best_bet_team=None, best_bet_line=8.5,
        best_bet_odds=1.95, best_total_direction="over",
        home_team="HOM", home_score=6, away_score=4,   # 10 > 8.5
    )
    assert over == ("win", pytest.approx(95.0))
    under = grade_best_bet(
        best_bet_type="total", best_bet_team=None, best_bet_line=8.5,
        best_bet_odds=1.95, best_total_direction="under",
        home_team="HOM", home_score=6, away_score=4,
    )
    assert under == ("loss", -100.0)


def test_total_push_on_exact_line():
    assert grade_best_bet(
        best_bet_type="total", best_bet_team=None, best_bet_line=9.0,
        best_bet_odds=1.95, best_total_direction="over",
        home_team="HOM", home_score=5, away_score=4,
    ) == ("push", 0.0)


def test_returns_none_rather_than_fabricating_when_data_is_missing():
    # No odds -> cannot compute profit; no line -> cannot grade a runline;
    # no direction -> cannot grade a total. Never guess.
    assert grade_best_bet(
        best_bet_type="moneyline", best_bet_team="HOM", best_bet_line=None,
        best_bet_odds=None, best_total_direction=None,
        home_team="HOM", home_score=5, away_score=3,
    ) == (None, None)
    assert grade_best_bet(
        best_bet_type="runline", best_bet_team="HOM", best_bet_line=None,
        best_bet_odds=2.0, best_total_direction=None,
        home_team="HOM", home_score=5, away_score=3,
    ) == (None, None)
    assert grade_best_bet(
        best_bet_type="total", best_bet_team=None, best_bet_line=8.5,
        best_bet_odds=2.0, best_total_direction=None,
        home_team="HOM", home_score=5, away_score=3,
    ) == (None, None)
    assert grade_best_bet(
        best_bet_type=None, best_bet_team=None, best_bet_line=None,
        best_bet_odds=None, best_total_direction=None,
        home_team="HOM", home_score=5, away_score=3,
    ) == (None, None)


def test_grades_without_any_component_columns():
    """The whole point: a snapshot with NULL best_ml_* still grades, because
    this helper never looks at the component columns."""
    assert _ml("HOM", 7, 2, odds=1.80) == ("win", pytest.approx(80.0))


# --- legacy-runline guard ---------------------------------------------------
# The first fix guarded only the SQL sentinel's second arm; rows entering via
# the actual_winner-IS-NULL arm (results-sync marking an old game final) were
# still graded at their untrustworthy stored line. The guard now lives next to
# the grading call, so it applies no matter how the row was selected.
from datetime import date

from src.tasks.mlb_scheduler import RUNLINE_SIGN_FIX_DATE, is_legacy_runline


def test_legacy_runline_is_flagged_regardless_of_how_the_row_was_selected():
    assert is_legacy_runline("runline", date(2026, 7, 21)) is True
    assert is_legacy_runline("runline", date(2026, 4, 15)) is True


def test_post_fix_runline_and_other_markets_are_not_flagged():
    assert is_legacy_runline("runline", RUNLINE_SIGN_FIX_DATE) is False
    assert is_legacy_runline("runline", date(2026, 8, 1)) is False
    assert is_legacy_runline("moneyline", date(2026, 4, 15)) is False
    assert is_legacy_runline("total", date(2026, 4, 15)) is False
    assert is_legacy_runline(None, date(2026, 4, 15)) is False
    assert is_legacy_runline("runline", None) is False
