"""Algorithm B: Multiply by confidence and market quality FIRST, then apply tanh."""

import numpy as np
from dataclasses import dataclass

# Edge scale per market type (to be tuned via backtesting)
# These represent the edge at which tanh gives ~76% (tanh(1) ≈ 0.76)
# Based on trained model, edges typically range 5-15% so use 10% as baseline
EDGE_SCALE = {
    "spread": 0.12,  # 12% edge = ~76% value score
    "moneyline": 0.10,  # 10% edge = ~76% value score
    "total": 0.15,  # 15% edge = ~76% value score (totals often have larger model uncertainty)
    "prop": 0.08,
}


@dataclass
class AlgoBResult:
    """Result from Algorithm B Value Score calculation."""

    raw_edge: float
    combined_edge: float
    confidence: float
    market_quality: float
    value_score: float


def compute_value_score_algo_b(
    p_true: float,
    p_market: float,
    market_type: str,
    confidence: float,
    market_quality: float,
) -> AlgoBResult:
    """
    Compute Value Score using Algorithm B (Idea 2 style).

    Order of operations:
    1. Calculate raw edge: p_true - p_market
    2. If raw edge <= 0, return 0 (no value in negative edge bets)
    3. Multiply raw edge by confidence and market quality
    4. Apply tanh transformation to combined edge
    5. Scale to 0-100

    Args:
        p_true: Calibrated model probability
        p_market: De-vigged market probability
        market_type: Type of market (spread, moneyline, total, prop)
        confidence: Confidence multiplier (typically 0.5-1.5 range)
        market_quality: Market quality factor (typically 0.5-1.3 range)

    Returns:
        AlgoBResult with all intermediate values and final score
    """
    # Get edge scale for market type
    edge_scale = EDGE_SCALE.get(market_type, 0.05)

    # Step 1: Raw edge
    raw_edge = p_true - p_market

    # Step 2: Handle negative edge
    if raw_edge <= 0:
        return AlgoBResult(
            raw_edge=raw_edge,
            combined_edge=0.0,
            confidence=confidence,
            market_quality=market_quality,
            value_score=0.0,
        )

    # Step 3: Multiply raw edge by confidence and market quality (FIRST)
    combined_edge = raw_edge * confidence * market_quality

    # Step 4: Non-linear squashing (THEN apply tanh)
    squashed = float(np.tanh(combined_edge / edge_scale))

    # Step 5: Scale to 0-100
    value_score = 100.0 * squashed

    return AlgoBResult(
        raw_edge=raw_edge,
        combined_edge=combined_edge,
        confidence=confidence,
        market_quality=market_quality,
        value_score=value_score,
    )
