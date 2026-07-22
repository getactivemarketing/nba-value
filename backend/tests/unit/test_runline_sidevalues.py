"""Runline side-value pairing (2026-07-21 sign fix): each side gets its own
cover probability, real price, and SIGNED line — no phantom from pairing the
+1.5 win-prob with the -1.5 plus-money price."""

from src.services.mlb.scorer import MLBScorer


def _by_team(values, team):
    return next(v for v in values if v.team == team)


def test_line_minus15_row_pairs_favorite_plus15_with_minus_money():
    # Home favored by ~0.55 (rd>0). Row: home -1.5 / away +1.5.
    # away favorite? No — home favored. away +1.5 is the underdog getting; but
    # the KEY invariant: away side line is +1.5 and its prob is p_away_plus.
    vals = MLBScorer._runline_side_values(
        predicted_run_diff=0.55, home_team="LAA", away_team="DET",
        line=-1.5, home_odds=2.64, away_odds=1.50,
    )
    home = _by_team(vals, "LAA")   # home -1.5
    away = _by_team(vals, "DET")   # away +1.5
    assert home.line == -1.5 and away.line == 1.5
    # away +1.5 prob = p_away_plus = 1 - P(home -1.5); home favored so P(home-1.5) modest,
    # so away +1.5 prob is high; paired with its real 1.50 (minus-money) price.
    assert away.odds_decimal == 1.50
    assert away.model_prob > 0.5


def test_line_plus15_row_pairs_away_minus15_with_its_own_low_prob():
    # Away favored (rd<0). Row stored as home +1.5 / away -1.5.
    # THE BUG CASE: away is the favorite, away -1.5 is plus-money. Correct pairing
    # must use p_away_minus (LOW), NOT p_away_plus, so no phantom edge.
    vals = MLBScorer._runline_side_values(
        predicted_run_diff=-0.55, home_team="PHI", away_team="LAD",
        line=1.5, home_odds=1.52, away_odds=2.55,
    )
    away = _by_team(vals, "LAD")   # away -1.5
    home = _by_team(vals, "PHI")   # home +1.5
    assert away.line == -1.5 and home.line == 1.5
    assert away.odds_decimal == 2.55
    # away -1.5 cover prob is P(away wins by 2+) with only 0.55 projected margin -> well under 0.5
    assert away.model_prob < 0.45
    # and therefore NOT a strong value bet at +155 (market ~0.39): small edge, not a phantom 0.35
    assert away.raw_edge < 0.15


def test_plus_and_minus_probs_are_complementary_across_the_two_rows():
    rd = 1.2
    row_minus = MLBScorer._runline_side_values(rd, "H", "A", -1.5, 2.5, 1.5)
    home_minus = _by_team(row_minus, "H").model_prob   # P(home -1.5)
    away_plus = _by_team(row_minus, "A").model_prob     # P(away +1.5) = 1 - P(home -1.5)
    assert abs((home_minus + away_plus) - 1.0) < 1e-9


def test_cover_prob_static_even_game_below_half_and_monotonic():
    # Even game: P(win by 2+) < 0.5; bigger favorite -> higher cover prob.
    assert MLBScorer._run_diff_to_cover_prob(0.0, 1.5) < 0.5
    assert MLBScorer._run_diff_to_cover_prob(2.0, 1.5) > MLBScorer._run_diff_to_cover_prob(0.5, 1.5)
