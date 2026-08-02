"""Model status: make silent degradation visible.

Every serious defect found on 2026-07-31/08-01 reported success while doing
nothing:

  - ingest_pitcher_stats logged count=0 and the scheduler printed
    "pitchers_updated=0" as success, for the life of the system
  - team rate stats were never fetched at all, so six columns sat at 0%
  - 24 of 28 model features served league-average constants, indistinguishable
    from a genuinely average team
  - the Athletics failed a name lookup and 111 games were skipped
  - the run-diff model was six months stale and nothing said so

The common thread is that nothing ever asserted what SHOULD be true. These
tests pin a status summary that answers three questions the system could not
previously answer about itself: which model is serving, how old is it, and is
a fallback quietly engaged.
"""
import pytest

from src.services.mlb.model_status import (
    STALE_AFTER_DAYS,
    describe_artifact,
    summarize_models,
)


class TestDescribeArtifact:
    def test_reports_version_and_feature_count(self):
        got = describe_artifact(
            {"version": "1.0", "trained_at": "2026-02-09T21:32:48", "feature_cols": ["a", "b"]},
            path="models/x.joblib",
            as_of="2026-08-01",
        )
        assert got["version"] == "1.0"
        assert got["features"] == 2
        assert got["path"] == "models/x.joblib"

    def test_computes_age_in_days(self):
        got = describe_artifact(
            {"trained_at": "2026-07-01T00:00:00"}, path="p", as_of="2026-08-01"
        )
        assert got["age_days"] == 31

    def test_flags_stale_models(self):
        fresh = describe_artifact({"trained_at": "2026-07-25T00:00:00"}, "p", "2026-08-01")
        stale = describe_artifact({"trained_at": "2026-02-09T00:00:00"}, "p", "2026-08-01")
        assert fresh["stale"] is False
        assert stale["stale"] is True
        assert stale["age_days"] > STALE_AFTER_DAYS

    def test_missing_trained_at_is_unknown_not_fresh(self):
        """mlb_totals_v2 shipped with trained_at=None; that must not read as new."""
        got = describe_artifact({"version": "2.0"}, path="p", as_of="2026-08-01")
        assert got["age_days"] is None
        assert got["stale"] is True

    def test_absent_artifact_is_reported_as_not_loaded(self):
        got = describe_artifact(None, path="models/gone.joblib", as_of="2026-08-01")
        assert got["loaded"] is False
        assert got["stale"] is True


class TestSummarizeModels:
    def _art(self, trained_at):
        return {"version": "1", "trained_at": trained_at, "feature_cols": ["a"]}

    def test_healthy_when_everything_is_loaded_and_fresh(self):
        got = summarize_models(
            run_diff=self._art("2026-07-28T00:00:00"),
            totals=self._art("2026-07-28T00:00:00"),
            run_diff_path="a", totals_path="b", as_of="2026-08-01",
        )
        assert got["status"] == "ok"
        assert got["degraded"] == []

    def test_degrades_when_the_run_diff_model_is_missing(self):
        """A missing run-diff model silently falls back to a hand-written
        heuristic — the most dangerous failure in the system."""
        got = summarize_models(
            run_diff=None, totals=self._art("2026-07-28T00:00:00"),
            run_diff_path="a", totals_path="b", as_of="2026-08-01",
        )
        assert got["status"] == "degraded"
        assert any("run_diff" in d for d in got["degraded"])

    def test_stale_model_is_surfaced_as_degraded(self):
        got = summarize_models(
            run_diff=self._art("2026-02-09T00:00:00"),
            totals=self._art("2026-07-28T00:00:00"),
            run_diff_path="a", totals_path="b", as_of="2026-08-01",
        )
        assert got["status"] == "degraded"
        assert any("stale" in d for d in got["degraded"])

    def test_reports_both_models_regardless_of_status(self):
        got = summarize_models(
            run_diff=None, totals=None,
            run_diff_path="a", totals_path="b", as_of="2026-08-01",
        )
        assert "run_diff" in got["models"] and "totals" in got["models"]
