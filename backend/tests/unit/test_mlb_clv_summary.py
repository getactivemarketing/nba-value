"""Aggregating CLV into a reportable summary.

The point of reporting CLV is to act on it sooner than win rate allows — but
that only works if the summary refuses to look conclusive before it is. These
tests pin the honesty rules: unmeasured picks never count as neutral, and the
verdict reflects the sample actually in hand.
"""
import pytest

from src.services.mlb.clv import summarize_clv


class TestSummarizeClv:
    def test_empty_sample_reports_nothing_rather_than_zero(self):
        s = summarize_clv([])
        assert s["measured"] == 0
        assert s["mean_clv"] is None
        assert s["beat_close_rate"] is None
        assert s["verdict"] == "insufficient"

    def test_computes_mean_and_beat_rate(self):
        s = summarize_clv([0.02, -0.01, 0.03, -0.02])
        assert s["measured"] == 4
        assert s["mean_clv"] == pytest.approx(0.005)
        assert s["beat_close_rate"] == pytest.approx(0.5)

    def test_median_is_reported_alongside_mean(self):
        """A couple of outliers should not be the whole story."""
        s = summarize_clv([0.01, 0.01, 0.01, 0.50])
        assert s["median_clv"] == pytest.approx(0.01)
        assert s["mean_clv"] > s["median_clv"]

    def test_standard_error_shrinks_with_sample_size(self):
        small = summarize_clv([0.02, -0.02] * 5)
        large = summarize_clv([0.02, -0.02] * 50)
        assert large["std_error"] < small["std_error"]

    def test_single_measurement_has_no_standard_error(self):
        s = summarize_clv([0.02])
        assert s["measured"] == 1
        assert s["std_error"] is None

    def test_none_values_are_excluded_not_treated_as_zero(self):
        """An unmeasured pick must not drag the mean toward zero."""
        with_gaps = summarize_clv([0.04, None, 0.04, None])
        assert with_gaps["measured"] == 2
        assert with_gaps["mean_clv"] == pytest.approx(0.04)

    def test_verdict_is_insufficient_below_thirty(self):
        assert summarize_clv([0.01] * 29)["verdict"] == "insufficient"

    def test_verdict_is_provisional_in_the_middle(self):
        assert summarize_clv([0.01] * 50)["verdict"] == "provisional"

    def test_verdict_is_meaningful_at_a_hundred(self):
        assert summarize_clv([0.01] * 100)["verdict"] == "meaningful"

    def test_verdict_ignores_how_good_the_number_looks(self):
        """A huge positive mean on a tiny sample is still not evidence."""
        assert summarize_clv([0.25] * 5)["verdict"] == "insufficient"

    def test_values_are_rounded_for_transport(self):
        s = summarize_clv([0.0123456, 0.0234567])
        assert s["mean_clv"] == round(s["mean_clv"], 5)
