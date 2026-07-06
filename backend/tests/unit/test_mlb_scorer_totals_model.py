"""Totals model path is configurable with fallback to v1 then heuristic."""

from src.services.mlb.scorer import MLBScorer


def test_explicit_path_loads():
    scorer = MLBScorer(None, totals_model_path="models/mlb_totals_v1.joblib")
    assert scorer.totals_model is not None


def test_missing_path_falls_back_to_v1():
    scorer = MLBScorer(None, totals_model_path="models/mlb_totals_v99_missing.joblib")
    assert scorer.totals_model is not None  # fell back to v1


def test_default_uses_configured_setting():
    from src.config import settings
    scorer = MLBScorer(None)
    # default setting points at v2 (shadow mode since 2026-07-06), so the model loads
    assert settings.mlb_totals_model_path == "models/mlb_totals_v2.joblib"
    assert scorer.totals_model is not None
