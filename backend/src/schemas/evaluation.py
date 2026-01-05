"""Evaluation and analytics Pydantic schemas."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


class AlgorithmMetrics(BaseModel):
    """Performance metrics for one algorithm."""

    brier_score: float = Field(description="Brier score (lower is better)")
    log_loss: float = Field(description="Log loss (lower is better)")
    clv_avg: float = Field(description="Average closing line value")
    roi: float = Field(description="Return on investment")
    win_rate: float = Field(ge=0, le=1, description="Win rate")
    bet_count: int = Field(description="Number of bets evaluated")


class AlgorithmComparisonResponse(BaseModel):
    """A/B comparison between scoring algorithms."""

    period_start: date
    period_end: date
    algo_a_metrics: dict
    algo_b_metrics: dict
    recommendation: Literal["algo_a", "algo_b", "insufficient_data", "no_difference"]
    confidence_level: float | None = Field(
        default=None, description="Statistical confidence in recommendation"
    )


class CalibrationResponse(BaseModel):
    """Calibration curve data point."""

    predicted_prob_bin: float = Field(description="Center of prediction bin")
    actual_win_rate: float = Field(description="Actual outcomes in this bin")
    sample_count: int = Field(description="Number of samples in bin")
    confidence_interval_low: float | None = None
    confidence_interval_high: float | None = None


class PerformanceByBucketResponse(BaseModel):
    """Performance metrics for a score bucket."""

    bucket_start: float
    bucket_end: float
    bet_count: int
    win_rate: float
    roi: float
    clv_avg: float
    avg_odds: float


class TrendResponse(BaseModel):
    """Trend analysis result."""

    trend_type: str
    segment: str
    sample_size: int
    avg_edge: float
    roi: float
    win_rate: float
    significance: float = Field(description="Statistical significance p-value")
