"""Market quality factor calculation."""

import numpy as np
from dataclasses import dataclass


@dataclass
class MarketQualityResult:
    """Result of market quality calculation."""

    liquidity_score: float
    book_consensus: float
    line_stability: float
    time_decay: float
    final_score: float


# Legacy alias for backwards compatibility
MarketQualityComponents = MarketQualityResult


def compute_market_quality(
    odds_decimal: float,
    market_type: str,
    time_to_tip_minutes: int,
    book: str | None = None,
    num_books: int = 1,
    odds_variance: float = 0.0,
    line_moves_1h: int = 0,
    algorithm: str = "a",
) -> MarketQualityResult:
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

    # Time Decay Factor
    # Markets closer to tip time have more reliable odds
    # 24+ hours: 0.8, 6-24 hours: 0.9, 1-6 hours: 1.0, <1 hour: 0.95 (too close)
    if time_to_tip_minutes <= 0:
        time_decay = 0.5  # Game started
    elif time_to_tip_minutes < 60:
        time_decay = 0.95  # Approaching tip, slightly less reliable
    elif time_to_tip_minutes < 360:  # 1-6 hours
        time_decay = 1.0  # Optimal window
    elif time_to_tip_minutes < 1440:  # 6-24 hours
        time_decay = 0.9
    else:
        time_decay = 0.8  # More than 24 hours out

    # Book quality adjustment
    sharp_books = {"pinnacle", "circa", "bookmaker"}
    if book and book.lower() in sharp_books:
        liquidity_score *= 1.1  # Sharp book bonus

    # Combine factors
    final = liquidity_score * book_consensus * line_stability * time_decay

    if algorithm == "a":
        # Algorithm A: 0-1 range
        final = float(np.clip(final, 0, 1))
    else:
        # Algorithm B: 0.5-1.3 range
        final = float(np.clip(final, 0.5, 1.3))

    return MarketQualityResult(
        liquidity_score=liquidity_score,
        book_consensus=book_consensus,
        line_stability=line_stability,
        time_decay=time_decay,
        final_score=final,
    )
