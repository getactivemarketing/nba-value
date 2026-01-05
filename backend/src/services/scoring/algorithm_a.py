"""Algorithm A: Apply tanh to raw edge FIRST, then multiply by confidence and market quality."""

import numpy as np
from dataclasses import dataclass

# Edge scale per market type (to be tuned via backtesting)
EDGE_SCALE = {
    "spread": 0.05,
    "moneyline": 0.04,
    "total": 0.045,
    "prop": 0.03,
}


@dataclass
class AlgoAResult:
    """Result from Algorithm A Value Score calculation."""

    raw_edge: float
    edge_score: float
    confidence: float
    market_quality: float
    value_score: float


def compute_value_score_algo_a(
    p_true: float,
    p_market: float,
    market_type: str,
    confidence: float,
    market_quality: float,
) -> AlgoAResult:
    """
    Compute Value Score using Algorithm A (Idea 1 style).

    Order of operations:
    1. Calculate raw edge: p_true - p_market
    2. Apply tanh transformation to raw edge (non-linear squashing)
    3. Multiply by confidence and market quality
    4. Scale to 0-100

    Args:
        p_true: Calibrated model probability
        p_market: De-vigged market probability
        market_type: Type of market (spread, moneyline, total, prop)
        confidence: Confidence multiplier (typically 0-2 range)
        market_quality: Market quality factor (typically 0-1 range)

    Returns:
        AlgoAResult with all intermediate values and final score
    """
    # Get edge scale for market type
    edge_scale = EDGE_SCALE.get(market_type, 0.05)

    # Step 1: Raw edge
    raw_edge = p_true - p_market

    # Step 2: Non-linear edge transformation (FIRST)
    # tanh compresses to (-1, 1) range
    edge_score = float(np.tanh(raw_edge / edge_scale))

    # Step 3 & 4: Multiply by confidence and market quality, scale to 0-100
    value_score = edge_score * confidence * market_quality * 100

    # Clamp to [0, 100]
    value_score = float(np.clip(value_score, 0, 100))

    return AlgoAResult(
        raw_edge=raw_edge,
        edge_score=edge_score,
        confidence=confidence,
        market_quality=market_quality,
        value_score=value_score,
    )
