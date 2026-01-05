"""Markets API endpoints."""

from typing import Literal

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import get_db
from src.schemas.market import MarketResponse, MarketFilters

router = APIRouter()


@router.get("/markets", response_model=list[MarketResponse])
async def get_markets(
    algorithm: Literal["a", "b"] = Query("a", description="Scoring algorithm to use"),
    market_type: str | None = Query(None, description="Filter by market type"),
    min_value_score: float = Query(0, ge=0, le=100, description="Minimum Value Score"),
    min_confidence: float = Query(0, ge=0, description="Minimum confidence"),
    limit: int = Query(50, ge=1, le=200, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    db: AsyncSession = Depends(get_db),
) -> list[MarketResponse]:
    """
    Get ranked betting markets by Value Score.

    Returns markets sorted by the selected algorithm's Value Score in descending order.
    Supports filtering by market type, minimum score, and confidence level.
    """
    # TODO: Implement actual database query
    # For now, return empty list as placeholder
    return []


@router.get("/markets/live")
async def get_live_markets(
    algorithm: Literal["a", "b"] = "a",
    db: AsyncSession = Depends(get_db),
) -> list[MarketResponse]:
    """
    Get markets for games starting within the next 24 hours.

    Optimized endpoint for the main Market Board view.
    """
    # TODO: Implement with time filter
    return []


@router.get("/markets/by-game/{game_id}")
async def get_markets_by_game(
    game_id: str,
    algorithm: Literal["a", "b"] = "a",
    db: AsyncSession = Depends(get_db),
) -> list[MarketResponse]:
    """Get all markets for a specific game."""
    # TODO: Implement
    return []
