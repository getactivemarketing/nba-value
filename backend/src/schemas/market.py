"""Market-related Pydantic schemas."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class MarketFilters(BaseModel):
    """Query parameters for filtering markets."""

    algorithm: Literal["a", "b"] = "a"
    market_type: str | None = None
    min_value_score: float = Field(default=0, ge=0, le=100)
    min_confidence: float = Field(default=0, ge=0)
    limit: int = Field(default=50, ge=1, le=200)
    offset: int = Field(default=0, ge=0)


class GameInfo(BaseModel):
    """Embedded game information."""

    game_id: str
    home_team: str
    away_team: str
    tip_time: datetime
    status: Literal["scheduled", "in_progress", "final"]


class MarketResponse(BaseModel):
    """Response model for market list endpoints."""

    market_id: str
    game_id: str
    game: GameInfo | None = None
    market_type: Literal["spread", "moneyline", "total", "prop"]
    outcome_label: str
    line: float | None = None
    odds_decimal: float
    odds_american: int | None = None

    # Model outputs
    p_true: float = Field(ge=0, le=1, description="Calibrated model probability")
    p_market: float = Field(ge=0, le=1, description="Market-implied probability")
    raw_edge: float = Field(description="p_true - p_market")

    # Algorithm A outputs
    algo_a_value_score: float = Field(ge=0, le=100)
    algo_a_confidence: float
    algo_a_market_quality: float

    # Algorithm B outputs
    algo_b_value_score: float = Field(ge=0, le=100)
    algo_b_confidence: float
    algo_b_market_quality: float

    # Meta
    time_to_tip_minutes: int
    calc_time: datetime
    book: str | None = None

    model_config = {"from_attributes": True}


class MarketSummary(BaseModel):
    """Simplified market info for lists."""

    market_id: str
    market_type: str
    outcome_label: str
    value_score: float
    raw_edge: float
    time_to_tip_minutes: int
