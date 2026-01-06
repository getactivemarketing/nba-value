"""Probability conversion functions from MOV predictions."""

from scipy import stats


def mov_to_spread_prob(
    predicted_mov: float,
    spread_line: float,
    mov_std: float = 12.0,
) -> float:
    """
    Convert MOV prediction to spread cover probability.

    The spread line uses standard betting convention:
    - spread_line = -6.5 means home is favored, must win by 7+ to cover
    - spread_line = +6.5 means home is underdog, can lose by up to 6 and cover

    P(home covers) = P(actual_margin + spread_line > 0)
                   = P(actual_margin > -spread_line)

    For home favorite (-6.5): must win by MORE than 6.5
    For home underdog (+6.5): must not lose by more than 6.5

    Args:
        predicted_mov: Predicted home margin of victory
        spread_line: Spread line (negative = home favored, positive = home underdog)
        mov_std: Standard deviation of MOV predictions

    Returns:
        Probability that home team covers the spread (0 to 1)
    """
    # For spread betting, home covers if: actual_margin + spread_line > 0
    # E.g., home -6.5: covers if margin > 6.5 (need margin + (-6.5) > 0, so margin > 6.5)
    # E.g., home +6.5: covers if margin > -6.5 (need margin + 6.5 > 0, so margin > -6.5)
    #
    # P(margin > -spread_line) = P(Z > (-spread_line - predicted_mov) / std)
    #                          = 1 - CDF((-spread_line - predicted_mov) / std)
    #                          = CDF((predicted_mov + spread_line) / std)
    z = (predicted_mov + spread_line) / mov_std

    return float(stats.norm.cdf(z))


def mov_to_moneyline_prob(
    predicted_mov: float,
    mov_std: float = 12.0,
) -> float:
    """
    Convert MOV prediction to moneyline (win) probability.

    P(home wins) = P(actual_margin > 0)

    Args:
        predicted_mov: Predicted home margin of victory
        mov_std: Standard deviation of MOV predictions

    Returns:
        Probability that home team wins (0 to 1)
    """
    z = predicted_mov / mov_std
    return float(stats.norm.cdf(z))


def mov_to_total_prob(
    home_total_estimate: float,
    away_total_estimate: float,
    total_line: float,
    total_std: float = 12.0,
) -> float:
    """
    Convert total points estimate to over probability.

    For totals, we use a separate model or pace-based estimate
    rather than deriving from MOV directly.

    P(over) = P(actual_total > total_line)

    Args:
        home_total_estimate: Estimated points for home team
        away_total_estimate: Estimated points for away team
        total_line: The over/under line
        total_std: Standard deviation of total predictions

    Returns:
        Probability of game going over the total (0 to 1)
    """
    predicted_total = home_total_estimate + away_total_estimate
    z = (predicted_total - total_line) / total_std
    return float(stats.norm.cdf(z))


def estimate_game_total(
    home_pace: float,
    away_pace: float,
    home_ortg: float,
    home_drtg: float,
    away_ortg: float,
    away_drtg: float,
    league_pace: float = 100.0,
) -> tuple[float, float]:
    """
    Estimate total points using pace and efficiency model.

    This is a simplified version of the Possessions × Efficiency model.

    Args:
        home_pace: Home team pace (possessions per 48 min)
        away_pace: Away team pace
        home_ortg: Home team offensive rating (pts per 100 poss)
        home_drtg: Home team defensive rating
        away_ortg: Away team offensive rating
        away_drtg: Away team defensive rating
        league_pace: League average pace for normalization

    Returns:
        Tuple of (home_points_estimate, away_points_estimate)
    """
    # Expected game pace (average of both teams relative to league)
    game_pace = (home_pace + away_pace) / 2

    # Possessions in the game (scaled to 48 minutes)
    possessions = game_pace

    # Home team points:
    # Blend of home offense vs away defense
    home_efficiency = (home_ortg + away_drtg) / 2
    home_points = home_efficiency * possessions / 100

    # Away team points:
    # Blend of away offense vs home defense
    away_efficiency = (away_ortg + home_drtg) / 2
    away_points = away_efficiency * possessions / 100

    return home_points, away_points


def implied_prob_to_fair_odds(implied_prob: float) -> float:
    """
    Convert implied probability to fair decimal odds.

    Args:
        implied_prob: Probability (0 to 1)

    Returns:
        Fair decimal odds (e.g., 2.0 for 50% probability)
    """
    if implied_prob <= 0:
        return float("inf")
    if implied_prob >= 1:
        return 1.0
    return 1.0 / implied_prob


def decimal_odds_to_american(decimal_odds: float) -> int:
    """
    Convert decimal odds to American odds.

    Args:
        decimal_odds: Decimal odds (e.g., 2.0, 1.5, 3.0)

    Returns:
        American odds (e.g., +100, -200, +200)
    """
    if decimal_odds >= 2.0:
        return int((decimal_odds - 1) * 100)
    else:
        return int(-100 / (decimal_odds - 1))


def american_odds_to_decimal(american_odds: int) -> float:
    """
    Convert American odds to decimal odds.

    Args:
        american_odds: American odds (e.g., +100, -200)

    Returns:
        Decimal odds
    """
    if american_odds > 0:
        return (american_odds / 100) + 1
    else:
        return (100 / abs(american_odds)) + 1


def devig_two_way_odds(
    odds1: float,
    odds2: float,
    method: str = "multiplicative",
) -> tuple[float, float]:
    """
    Remove vig from two-way market to get fair probabilities.

    Args:
        odds1: Decimal odds for outcome 1
        odds2: Decimal odds for outcome 2
        method: Devigging method ('multiplicative' or 'additive')

    Returns:
        Tuple of (prob1, prob2) - fair probabilities
    """
    raw1 = 1 / odds1
    raw2 = 1 / odds2
    total = raw1 + raw2

    if method == "multiplicative":
        # Proportional method - most common
        prob1 = raw1 / total
        prob2 = raw2 / total
    elif method == "additive":
        # Shin method / power method could go here
        # For now, same as multiplicative
        prob1 = raw1 / total
        prob2 = raw2 / total
    else:
        prob1 = raw1 / total
        prob2 = raw2 / total

    return prob1, prob2
