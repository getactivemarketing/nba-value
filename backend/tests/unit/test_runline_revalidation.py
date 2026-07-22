"""Corrected runline grading helper (flat 1u): sign-aware, matches prod rule."""

import pytest
from scripts.runline_revalidation import grade_runline


def test_plus15_covers_when_team_loses_by_one():
    # away DET +1.5; DET loses by 1 (LAA 3, DET 2) -> +1.5 covers (win)
    res, profit = grade_runline("DET", 1.5, home_team="LAA", home_score=3, away_score=2, odds_decimal=1.5)
    assert res == "win" and profit == pytest.approx(0.5)


def test_minus15_loses_when_favorite_wins_by_one():
    # DET -1.5 needs win by 2+. DET 2 - LAA 1 = win by 1 -> loss.
    res, profit = grade_runline("DET", -1.5, home_team="LAA", home_score=1, away_score=2, odds_decimal=2.5)
    assert res == "loss" and profit == pytest.approx(-1.0)


def test_minus15_wins_when_favorite_wins_by_two():
    res, profit = grade_runline("DET", -1.5, home_team="LAA", home_score=1, away_score=3, odds_decimal=2.5)
    assert res == "win" and profit == pytest.approx(1.5)


def test_home_team_plus15_perspective():
    # home LAA +1.5, LAA loses by 1 (LAA 2, DET 3) -> covers (win)
    res, profit = grade_runline("LAA", 1.5, home_team="LAA", home_score=2, away_score=3, odds_decimal=1.8)
    assert res == "win" and profit == pytest.approx(0.8)
