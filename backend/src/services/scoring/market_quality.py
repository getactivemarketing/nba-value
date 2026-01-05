"""Market quality factor calculation."""

import numpy as np
from dataclasses import dataclass


@dataclass
class MarketQualityComponents:
    """Components of the market quality factor."""

    liquidity_score: float
    book_consensus: float
    line_stability: float
    final_multiplier: float


def compute_market_quality(
    market_type: str,
    num_books: int = 1,
    odds_variance: float = 0.0,
    line_moves_1h: int = 0,
    algorithm: str = "a",
) -> MarketQualityComponents:
    """
    Compute market quality factor from component factors.

    Components:
    - Liquidity Score: Based on market type and number of books offering
    - Book Consensus: Lower variance across books = more consensus
    - Line Stability: Fewer recent line moves = more stable

    Args:
        market_type: Type of market
        num_books: Number of sportsbooks offering this market
        odds_variance: Variance in odds across books
        line_moves_1h: Number of line movements in last hour
        algorithm: Which algorithm ('a' or 'b')

    Returns:
        MarketQualityComponents with all factors and final multiplier
    """
    # Liquidity Score by market type
    base_liquidity = {
        "spread": 1.0,
        "moneyline": 0.95,
        "total": 0.90,
        "prop": 0.70,
    }.get(market_type, 0.8)

    # Adjust by number of books (more books = more liquid)
    # 1 book = 0.8x, 5+ books = 1.0x
    book_adjustment = min(0.8 + (num_books * 0.05), 1.0)
    liquidity_score = base_liquidity * book_adjustment

    # Book Consensus
    # Lower variance = higher consensus
    # Map variance of [0, 0.05] to consensus of [1.0, 0.7]
    book_consensus = float(np.clip(1.0 - (odds_variance * 6), 0.7, 1.0))

    # Line Stability
    # Fewer moves = more stable = higher quality
    # 0 moves = 1.0, 5+ moves = 0.7
    line_stability = float(np.clip(1.0 - (line_moves_1h * 0.06), 0.7, 1.0))

    # Combine factors
    final = liquidity_score * book_consensus * line_stability

    if algorithm == "a":
        # Algorithm A: 0-1 range
        final = float(np.clip(final, 0, 1))
    else:
        # Algorithm B: 0.5-1.3 range
        final = float(np.clip(final, 0.5, 1.3))

    return MarketQualityComponents(
        liquidity_score=liquidity_score,
        book_consensus=book_consensus,
        line_stability=line_stability,
        final_multiplier=final,
    )
