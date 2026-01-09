"""Bet detail API endpoints."""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.database import get_db
from src.models.market import Market
from src.models.game import Game
from src.models.prediction import ModelPrediction
from src.models.score import ValueScore
from src.schemas.bet import (
    BetDetailResponse,
    BetHistoryResponse,
    AlgorithmScore,
    ConfidenceBreakdown,
    MarketQualityBreakdown,
)

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
    # Get market with game
    result = await db.execute(
        select(Market)
        .options(selectinload(Market.game))
        .where(Market.market_id == market_id)
    )
    market = result.scalar_one_or_none()

    if not market:
        raise HTTPException(status_code=404, detail="Market not found")

    # Get latest value score for this market
    score_result = await db.execute(
        select(ValueScore)
        .where(ValueScore.market_id == market_id)
        .order_by(ValueScore.calc_time.desc())
        .limit(1)
    )
    value_score = score_result.scalar_one_or_none()

    # Get latest prediction
    pred_result = await db.execute(
        select(ModelPrediction)
        .where(ModelPrediction.market_id == market_id)
        .order_by(ModelPrediction.prediction_time.desc())
        .limit(1)
    )
    prediction = pred_result.scalar_one_or_none()

    if not prediction:
        raise HTTPException(status_code=404, detail="No predictions found for this market")

    game = market.game

    # Build confidence breakdowns (using defaults since we don't store individual components)
    confidence_a = ConfidenceBreakdown(
        ensemble_agreement=0.8,
        calibration_reliability=0.9,
        injury_certainty=0.85,
        segment_reliability=0.75,
        final_multiplier=float(value_score.algo_a_confidence) if value_score and value_score.algo_a_confidence else 1.0,
    )

    confidence_b = ConfidenceBreakdown(
        ensemble_agreement=0.8,
        calibration_reliability=0.9,
        injury_certainty=0.85,
        segment_reliability=None,
        final_multiplier=float(value_score.algo_b_confidence) if value_score and value_score.algo_b_confidence else 1.0,
    )

    market_quality_a = MarketQualityBreakdown(
        liquidity_score=0.9,
        book_consensus=0.85,
        line_stability=0.8,
        final_multiplier=float(value_score.algo_a_market_quality) if value_score and value_score.algo_a_market_quality else 1.0,
    )

    market_quality_b = MarketQualityBreakdown(
        liquidity_score=0.9,
        book_consensus=0.85,
        line_stability=0.8,
        final_multiplier=float(value_score.algo_b_market_quality) if value_score and value_score.algo_b_market_quality else 1.0,
    )

    algo_a = AlgorithmScore(
        algorithm="a",
        value_score=float(value_score.algo_a_value_score) if value_score and value_score.algo_a_value_score else 0,
        edge_score=float(value_score.algo_a_edge_score) if value_score and value_score.algo_a_edge_score else None,
        combined_edge=None,
        confidence=confidence_a,
        market_quality=market_quality_a,
    )

    algo_b = AlgorithmScore(
        algorithm="b",
        value_score=float(value_score.algo_b_value_score) if value_score and value_score.algo_b_value_score else 0,
        edge_score=None,
        combined_edge=float(value_score.algo_b_combined_edge) if value_score and value_score.algo_b_combined_edge else None,
        confidence=confidence_b,
        market_quality=market_quality_b,
    )

    # Calculate time to tip
    now = datetime.utcnow()
    time_to_tip = int((game.tip_time_utc.replace(tzinfo=None) - now).total_seconds() / 60)
    time_to_tip = max(0, time_to_tip)

    # Determine active algorithm and recommended score
    active_algo = value_score.active_algorithm.lower() if value_score and value_score.active_algorithm else "b"
    recommended_score = (
        float(value_score.algo_a_value_score) if active_algo == "a" and value_score and value_score.algo_a_value_score
        else float(value_score.algo_b_value_score) if value_score and value_score.algo_b_value_score
        else 0
    )

    p_true = float(prediction.p_true)
    p_market = float(prediction.p_market)
    raw_edge = float(prediction.raw_edge)

    return BetDetailResponse(
        market_id=market.market_id,
        game_id=market.game_id,
        home_team=game.home_team_id,
        away_team=game.away_team_id,
        tip_time=game.tip_time_utc,
        market_type=market.market_type,
        outcome_label=market.outcome_label,
        line=float(market.line) if market.line else None,
        odds_decimal=float(market.odds_decimal),
        odds_american=market.odds_american,
        book=market.book or "unknown",
        p_true=p_true,
        p_market=p_market,
        raw_edge=raw_edge,
        edge_percentage=raw_edge * 100,
        p_ensemble_mean=float(prediction.p_ensemble_mean),
        p_ensemble_std=float(prediction.p_ensemble_std) if prediction.p_ensemble_std else 0.0,
        algo_a=algo_a,
        algo_b=algo_b,
        active_algorithm=active_algo,
        recommended_score=recommended_score,
        calc_time=value_score.calc_time if value_score else datetime.utcnow(),
        time_to_tip_minutes=time_to_tip,
    )


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
    from datetime import timedelta

    cutoff = datetime.utcnow() - timedelta(hours=hours)

    # Get value scores with their predictions
    result = await db.execute(
        select(ValueScore)
        .options(selectinload(ValueScore.prediction))
        .where(ValueScore.market_id == market_id)
        .where(ValueScore.calc_time >= cutoff)
        .order_by(ValueScore.calc_time.asc())
    )
    scores = result.scalars().all()

    # Also get the market for line info
    market_result = await db.execute(
        select(Market).where(Market.market_id == market_id)
    )
    market = market_result.scalar_one_or_none()

    history = []
    for score in scores:
        pred = score.prediction
        if pred:
            history.append(BetHistoryResponse(
                calc_time=score.calc_time,
                p_true=float(pred.p_true),
                p_market=float(pred.p_market),
                raw_edge=float(pred.raw_edge),
                algo_a_value_score=float(score.algo_a_value_score) if score.algo_a_value_score else 0,
                algo_b_value_score=float(score.algo_b_value_score) if score.algo_b_value_score else 0,
                odds_decimal=float(market.odds_decimal) if market else 2.0,
                line=float(market.line) if market and market.line else None,
            ))

    return history


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
