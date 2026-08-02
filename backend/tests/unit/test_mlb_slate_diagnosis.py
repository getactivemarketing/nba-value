"""Tell "no edge" apart from "no data".

Every external dependency in this system degrades into the same observable
outcome: no picks tonight. A broken odds feed, an unmapped team name, and a
genuinely quiet market are indistinguishable from the outside — which is how
111 Athletics games were skipped all season on a team-name lookup while the
scheduler logged success every 30 minutes.

With only moneyline live, silence is now the EXPECTED output on many nights,
so a broken feed hides in plain sight. These tests pin a per-game reason and a
slate-level verdict that separates the two.
"""
import pytest

from src.services.mlb.slate_diagnosis import (
    MIN_HEALTHY_BOOKS,
    GameSlot,
    diagnose_game,
    diagnose_slate,
)


def slot(**kw) -> GameSlot:
    base = dict(matchup="AWY@HOM", ml_books=11, has_pick=False, decided=True)
    base.update(kw)
    return GameSlot(**base)


class TestDiagnoseGame:
    def test_a_pick_is_reported_as_a_pick(self):
        assert diagnose_game(slot(has_pick=True))["reason"] == "pick"

    def test_full_market_without_a_pick_is_genuinely_no_edge(self):
        got = diagnose_game(slot(ml_books=11, has_pick=False))
        assert got["reason"] == "no_edge"
        assert got["is_data_gap"] is False

    def test_zero_books_is_a_data_gap_not_no_edge(self):
        """The Athletics case: game exists, odds never arrived."""
        got = diagnose_game(slot(ml_books=0))
        assert got["reason"] == "no_market"
        assert got["is_data_gap"] is True

    def test_thin_market_is_flagged_as_a_partial_gap(self):
        got = diagnose_game(slot(ml_books=1))
        assert got["reason"] == "thin_market"
        assert got["is_data_gap"] is True

    def test_threshold_is_the_boundary_not_below_it(self):
        assert diagnose_game(slot(ml_books=MIN_HEALTHY_BOOKS))["reason"] == "no_edge"
        assert diagnose_game(slot(ml_books=MIN_HEALTHY_BOOKS - 1))["reason"] == "thin_market"

    def test_a_pick_on_a_thin_market_still_counts_as_a_pick(self):
        """We bet it, so the data was sufficient; do not cry gap."""
        assert diagnose_game(slot(ml_books=1, has_pick=True))["reason"] == "pick"


class TestDiagnoseSlate:
    def test_healthy_slate_with_picks(self):
        got = diagnose_slate([slot(has_pick=True), slot(), slot()])
        assert got["verdict"] == "ok"
        assert got["picks"] == 1
        assert got["data_gaps"] == 0

    def test_quiet_night_with_full_markets_is_healthy(self):
        """No picks is a legitimate outcome and must not raise an alarm."""
        got = diagnose_slate([slot(), slot(), slot()])
        assert got["picks"] == 0
        assert got["verdict"] == "ok"
        assert "no_edge" in got["explanation"]

    def test_silence_caused_by_missing_odds_is_NOT_healthy(self):
        got = diagnose_slate([slot(ml_books=0), slot(ml_books=0), slot(ml_books=0)])
        assert got["picks"] == 0
        assert got["verdict"] == "data_gap"
        assert got["data_gaps"] == 3

    def test_partial_gap_is_surfaced_even_when_picks_exist(self):
        """A pick elsewhere must not mask games the model never got to see."""
        got = diagnose_slate([slot(has_pick=True), slot(ml_books=0), slot(ml_books=0)])
        assert got["picks"] == 1
        assert got["verdict"] == "data_gap"

    def test_single_gap_on_a_full_slate_is_a_warning_not_an_alarm(self):
        games = [slot() for _ in range(14)] + [slot(ml_books=0)]
        got = diagnose_slate(games)
        assert got["verdict"] == "warn"

    def test_empty_slate_is_its_own_verdict(self):
        """Zero games ingested is a different failure from zero odds."""
        got = diagnose_slate([])
        assert got["verdict"] == "no_games"

    def test_explanation_names_the_affected_games(self):
        got = diagnose_slate([slot(matchup="DET@OAK", ml_books=0), slot()])
        assert "DET@OAK" in got["explanation"]

    def test_pre_snapshot_games_are_pending_not_no_edge(self):
        """Before the snapshot window the model has not been asked yet."""
        got = diagnose_slate([slot(decided=False), slot(decided=False)])
        assert got["pending"] == 2
        assert got["no_edge"] == 0
        assert got["verdict"] == "ok"

    def test_a_pending_game_with_no_market_is_still_a_gap(self):
        """Missing odds is knowable before the snapshot; do not hide it."""
        got = diagnose_game(slot(decided=False, ml_books=0))
        assert got["reason"] == "no_market"
        assert got["is_data_gap"] is True

    def test_counts_add_up_to_the_slate_size(self):
        games = [slot(has_pick=True), slot(), slot(ml_books=0), slot(ml_books=1)]
        got = diagnose_slate(games)
        assert (got["picks"] + got["no_edge"] + got["pending"]
                + got["data_gaps"]) == len(games)
