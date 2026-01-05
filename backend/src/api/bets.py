"""Bet detail API endpoints."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import get_db
from src.schemas.bet import BetDetailResponse, BetHistoryResponse

router = APIRouter()


@router.get("/bet/{market_id}", response_model=BetDetailResponse)
async def get_bet_detail(
    market_id: str,
    db: AsyncSession = Depends(get_db),
) -> BetDetailResponse:
    """
    Get full bet detail with Value Score breakdown.

    Returns:
    - Model probability (p_true) and market probability (p_market)
    - Raw edge calculation
    - Confidence component breakdown
    - Market quality component breakdown
    - Both Algorithm A and B scores for comparison
    """
    # TODO: Implement actual database query
    raise HTTPException(status_code=404, detail="Market not found")


@router.get("/bet/{market_id}/history", response_model=list[BetHistoryResponse])
async def get_bet_history(
    market_id: str,
    hours: int = 24,
    db: AsyncSession = Depends(get_db),
) -> list[BetHistoryResponse]:
    """
    Get historical Value Scores for a market.

    Returns time-series of scores showing how the value has changed
    as odds move and model predictions update.
    """
    # TODO: Implement
    return []


@router.get("/bet/{market_id}/similar")
async def get_similar_bets(
    market_id: str,
    limit: int = 10,
    db: AsyncSession = Depends(get_db),
) -> list[dict]:
    """
    Get historical bets with similar characteristics.

    Useful for showing users how similar bets have performed.
    """
    # TODO: Implement similarity search
    return []
