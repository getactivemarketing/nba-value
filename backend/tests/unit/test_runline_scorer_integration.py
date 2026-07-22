"""The runline block collects correctly-paired sides across all standard rows
and picks the best; the old bug's input (away favorite, line=+1.5 row) no longer
yields a phantom high-value +1.5-labeled pick at -1.5 odds."""

from dataclasses import dataclass
from src.services.mlb.scorer import MLBScorer


@dataclass
class FakeMarket:
    market_type: str
    line: float
    home_odds: float
    away_odds: float


def collect_runline_values(predicted_run_diff, home, away, markets):
    """Mirror the score_game runline loop using the public helper, so we can
    exercise selection without the async DB path."""
    from src.services.mlb.value_calculator import MLBValueCalculator
    vals = []
    for m in markets:
        if m.market_type != "runline":
            continue
        if not m.line or abs(float(m.line)) != 1.5:
            continue
        vals += MLBScorer._runline_side_values(
            predicted_run_diff, home, away, float(m.line),
            float(m.home_odds), float(m.away_odds),
        )
    return vals, MLBValueCalculator.find_best_value(vals)


def test_away_favorite_no_phantom_minus15_pick():
    # LAD (away) favored by 0.55. Books: some line=-1.5 (LAD +1.5 @1.50), some line=1.5 (LAD -1.5 @2.55).
    markets = [
        FakeMarket("runline", -1.5, 2.64, 1.50),
        FakeMarket("runline", 1.5, 1.52, 2.55),
        FakeMarket("moneyline", None, 1.6, 2.4),  # ignored
    ]
    vals, best = collect_runline_values(-0.55, "PHI", "LAD", markets)
    # No side should be a LAD -1.5 (line -1.5) with a big edge — the -1.5 side's
    # true cover prob is low, so it cannot be a phantom best bet.
    lad_minus = [v for v in vals if v.team == "LAD" and v.line == -1.5]
    assert all(v.raw_edge < 0.15 for v in lad_minus)
    # If LAD is the best runline, it must be the +1.5 side at minus-money (<2.0).
    if best and best.team == "LAD":
        assert best.line == 1.5 and best.odds_decimal < 2.0


def test_alt_lines_skipped():
    markets = [FakeMarket("runline", -1.0, 1.5, 2.5), FakeMarket("runline", 2.5, 2.1, 1.6)]
    vals, best = collect_runline_values(0.3, "H", "A", markets)
    assert vals == [] and best is None
