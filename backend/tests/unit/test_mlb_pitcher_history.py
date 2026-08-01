"""Reconstructing point-in-time pitcher stats from MLB game logs.

mlb_pitcher_stats only ever held a single season-to-date snapshot taken today,
so a training row for an April game would have been fed July's numbers — pure
leakage. Game logs give per-start lines with dates, so cumulative stats can be
rebuilt as they stood BEFORE any given game.

The trap this module exists to get right: MLB writes innings pitched in
baseball notation. "6.1" is six and one THIRD innings, not 6.1. Averaging or
summing it as a decimal silently corrupts every rate stat downstream.
"""
import pytest

from src.services.mlb.pitcher_history import (
    GameLogEntry,
    cumulative_through,
    parse_innings,
    build_stat_history,
)


class TestParseInnings:
    @pytest.mark.parametrize("raw,expected", [
        ("6.0", 6.0),
        ("6.1", 6 + 1 / 3),
        ("6.2", 6 + 2 / 3),
        ("0.1", 1 / 3),
        ("0.0", 0.0),
        ("200.2", 200 + 2 / 3),
    ])
    def test_baseball_notation(self, raw, expected):
        assert parse_innings(raw) == pytest.approx(expected)

    def test_whole_number_without_decimal(self):
        assert parse_innings("7") == pytest.approx(7.0)

    def test_numeric_input_is_treated_as_notation_too(self):
        """The API sometimes returns a float; 6.1 still means 6 1/3."""
        assert parse_innings(6.1) == pytest.approx(6 + 1 / 3)

    def test_missing_is_none(self):
        assert parse_innings(None) is None
        assert parse_innings("") is None

    def test_garbage_is_none_not_zero(self):
        """Zero would quietly deflate ERA; None is honest."""
        assert parse_innings("-.--") is None


def log(date, ip="6.0", er=2, h=5, bb=1, k=6):
    return GameLogEntry(
        date=date, innings_pitched=ip, earned_runs=er,
        hits=h, walks=bb, strikeouts=k,
    )


class TestCumulativeThrough:
    def test_accumulates_only_games_strictly_before_the_date(self):
        logs = [log("2026-04-10"), log("2026-04-16"), log("2026-04-22")]
        got = cumulative_through(logs, "2026-04-20")
        assert got["innings_pitched"] == pytest.approx(12.0)   # first two only

    def test_excludes_the_game_being_predicted(self):
        """A start on the target date must not inform its own prediction."""
        logs = [log("2026-04-10"), log("2026-04-20")]
        got = cumulative_through(logs, "2026-04-20")
        assert got["innings_pitched"] == pytest.approx(6.0)

    def test_era_uses_thirds_not_decimals(self):
        # 6 1/3 innings, 2 earned runs -> 9*2/6.3333 = 2.842
        got = cumulative_through([log("2026-04-10", ip="6.1", er=2)], "2026-05-01")
        assert got["era"] == pytest.approx(9 * 2 / (6 + 1 / 3), abs=1e-3)

    def test_whip_counts_hits_plus_walks(self):
        got = cumulative_through([log("2026-04-10", ip="6.0", h=5, bb=1)], "2026-05-01")
        assert got["whip"] == pytest.approx(6 / 6.0)

    def test_rate_stats_across_multiple_starts(self):
        logs = [log("2026-04-10", ip="6.0", er=2, k=6), log("2026-04-16", ip="6.0", er=4, k=12)]
        got = cumulative_through(logs, "2026-05-01")
        assert got["era"] == pytest.approx(9 * 6 / 12.0)
        assert got["k_per_9"] == pytest.approx(9 * 18 / 12.0)

    def test_no_prior_games_returns_none(self):
        """A pitcher's first start has no history — must not be 0.00 ERA."""
        assert cumulative_through([log("2026-04-20")], "2026-04-10") is None

    def test_zero_innings_does_not_divide_by_zero(self):
        assert cumulative_through([log("2026-04-10", ip="0.0")], "2026-05-01") is None


class TestBuildStatHistory:
    def test_emits_one_row_per_start_dated_the_day_after(self):
        """Stats become knowable the day AFTER the start that produced them."""
        logs = [log("2026-04-10"), log("2026-04-16")]
        hist = build_stat_history(logs)
        assert [h["stat_date"] for h in hist] == ["2026-04-11", "2026-04-17"]

    def test_each_row_reflects_only_games_up_to_that_point(self):
        logs = [log("2026-04-10", ip="6.0"), log("2026-04-16", ip="6.0")]
        hist = build_stat_history(logs)
        assert hist[0]["innings_pitched"] == pytest.approx(6.0)
        assert hist[1]["innings_pitched"] == pytest.approx(12.0)

    def test_unsorted_logs_are_handled(self):
        logs = [log("2026-04-16"), log("2026-04-10")]
        hist = build_stat_history(logs)
        assert [h["stat_date"] for h in hist] == ["2026-04-11", "2026-04-17"]

    def test_empty_logs_produce_no_rows(self):
        assert build_stat_history([]) == []
