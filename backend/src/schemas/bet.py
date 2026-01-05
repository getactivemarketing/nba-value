"""Bet detail Pydantic schemas."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ConfidenceBreakdown(BaseModel):
    """Breakdown of confidence multiplier components."""

    ensemble_agreement: float = Field(description="Model agreement score")
    calibration_reliability: float = Field(description="Historical calibration accuracy")
    injury_certainty: float = Field(description="Injury information certainty")
    segment_reliability: float | None = Field(
        default=None, description="Edge band reliability (Algo A only)"
    )
    final_multiplier: float = Field(description="Combined confidence multiplier")


class MarketQualityBreakdown(BaseModel):
    """Breakdown of market quality factor components."""

    liquidity_score: float = Field(description="Market liquidity assessment")
    book_consensus: float = Field(description="Agreement across sportsbooks")
    line_stability: float = Field(description="Line movement stability")
    final_multiplier: float = Field(description="Combined market quality factor")


class AlgorithmScore(BaseModel):
    """Value Score details for one algorithm."""

    algorithm: Literal["a", "b"]
    value_score: float = Field(ge=0, le=100)
    edge_score: float | None = Field(default=None, description="Algo A: tanh(edge)")
    combined_edge: float | None = Field(default=None, description="Algo B: edge × conf × mq")
    confidence: ConfidenceBreakdown
    market_quality: MarketQualityBreakdown


class BetDetailResponse(BaseModel):
    """Full bet detail with Value Score breakdown."""

    # Identifiers
    market_id: str
    game_id: str

    # Game context
    home_team: str
    away_team: str
    tip_time: datetime

    # Market info
    market_type: Literal["spread", "moneyline", "total", "prop"]
    outcome_label: str
    line: float | None = None
    odds_decimal: float
    odds_american: int
    book: str

    # Core probabilities
    p_true: float = Field(ge=0, le=1, description="Calibrated model probability")
    p_market: float = Field(ge=0, le=1, description="De-vigged market probability")
    raw_edge: float = Field(description="p_true - p_market")
    edge_percentage: float = Field(description="Edge as percentage")

    # Ensemble details
    p_ensemble_mean: float = Field(description="Mean of ensemble predictions")
    p_ensemble_std: float = Field(description="Std dev of ensemble predictions")

    # Algorithm scores
    algo_a: AlgorithmScore
    algo_b: AlgorithmScore

    # Active recommendation
    active_algorithm: Literal["a", "b"]
    recommended_score: float

    # Timing
    calc_time: datetime
    time_to_tip_minutes: int

    model_config = {"from_attributes": True}


class BetHistoryResponse(BaseModel):
    """Historical Value Score entry."""

    calc_time: datetime
    p_true: float
    p_market: float
    raw_edge: float
    algo_a_value_score: float
    algo_b_value_score: float
    odds_decimal: float
    line: float | None = None
