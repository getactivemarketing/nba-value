"""best_ml must be the best moneyline across ALL bookmaker rows, not the last one.

Regression test for the stranded-grading bug (found 2026-07-28):

`_calculate_market_values` accumulates every book's values into `all_values`
(which feeds `best_bet`), but `best_ml` was assigned INSIDE the per-market loop
from only that row's two values — so with ~11 books per game it ended up
reflecting whichever row happened to be iterated last. When that last row had no
qualifying value bet, `best_ml` came back None while `best_bet` was still a
perfectly good moneyline.

Downstream that stranded the row permanently: the grader only grades ML
`if snapshot.best_ml_team:` and then copies `best_bet_result = best_ml_result`,
so best_bet_result stayed NULL — and because the grader's "already done"
sentinel (`actual_winner`) *was* stamped in the same pass, the row was never
revisited. 17 moneyline snapshots were stranded this way (Apr 10 - Jul 26).

Runline already got this exact fix on 2026-07-21 (it accumulates `rl_values`
and selects after the loop); this pins the same contract for moneyline.
"""
from dataclasses import dataclass
from datetime import date
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.services.mlb.scorer import MLBGamePrediction, MLBScorer


@dataclass
class FakeMarket:
    market_type: str
    line: float | None
    home_odds: float | None
    away_odds: float | None
    over_odds: float | None = None
    under_odds: float | None = None


@dataclass
class FakeGame:
    game_id: str
    home_team: str
    away_team: str


def _scorer_with_markets(markets):
    """Bypass __init__ (which loads joblib models off disk) — this exercises
    only the market-value loop, which needs nothing but `self.session`."""
    scorer = object.__new__(MLBScorer)
    result = MagicMock()
    result.scalars.return_value.all.return_value = markets
    session = MagicMock()
    session.execute = AsyncMock(return_value=result)
    scorer.session = session
    return scorer


def _prediction():
    return MLBGamePrediction(
        game_id="g1", home_team="HOM", away_team="AWY",
        game_date=date(2026, 7, 27), game_time=None,
        predicted_run_diff=0.5, predicted_total=8.5,
        p_home_win=0.60, p_away_win=0.40,
        p_home_cover=0.5, p_away_cover=0.5,
        p_over=0.5, p_under=0.5,
    )


# Book A: home is genuinely underpriced -> devigged home ~0.443 vs model 0.60,
#         raw_edge ~0.157 (>= MIN_EDGE 0.10) -> QUALIFIES.
# Book B: home priced ~fairly -> devigged home ~0.582 vs model 0.60,
#         raw_edge ~0.018 (< 0.10) -> neither side qualifies.
BOOK_A = FakeMarket("moneyline", None, 2.20, 1.75)
BOOK_B = FakeMarket("moneyline", None, 1.65, 2.30)


@pytest.mark.asyncio
async def test_best_ml_survives_when_the_last_book_has_no_qualifying_price():
    """The pre-fix failure: qualifying price in an earlier book, none in the last.

    best_bet correctly picks the good price (it sees all books), so best_ml must
    too — otherwise the snapshot stores a moneyline best_bet with a NULL
    best_ml_team and can never be graded.
    """
    scorer = _scorer_with_markets([BOOK_A, BOOK_B])  # non-qualifying book LAST
    prediction = _prediction()

    await scorer._calculate_market_values(prediction, FakeGame("g1", "HOM", "AWY"))

    assert prediction.best_bet is not None
    assert prediction.best_bet.market_type == "moneyline"
    # The actual bug: this was None because only BOOK_B was considered.
    assert prediction.best_ml is not None, (
        "best_ml was dropped because the last book had no qualifying price — "
        "this is what strands the snapshot as ungradeable"
    )
    # And it must be the SAME pick best_bet chose, at the same price.
    assert prediction.best_ml.team == prediction.best_bet.team
    assert prediction.best_ml.odds_decimal == prediction.best_bet.odds_decimal


@pytest.mark.asyncio
async def test_best_ml_takes_the_best_price_across_books_not_the_last():
    """Even when the last book qualifies, best_ml must be the best price."""
    # Both books qualify on HOM, but BOOK_A pays more (2.20 vs 2.05).
    book_c = FakeMarket("moneyline", None, 2.05, 1.85)
    scorer = _scorer_with_markets([BOOK_A, book_c])  # worse-priced book LAST
    prediction = _prediction()

    await scorer._calculate_market_values(prediction, FakeGame("g1", "HOM", "AWY"))

    assert prediction.best_ml is not None
    assert prediction.best_ml.team == "HOM"
    # Pre-fix this returned 2.05 (the last row); the best available is 2.20.
    assert prediction.best_ml.odds_decimal == 2.20
    assert prediction.best_ml.odds_decimal == prediction.best_bet.odds_decimal


@pytest.mark.asyncio
async def test_best_ml_is_none_when_no_book_qualifies():
    """Guard against over-correcting: if nothing qualifies anywhere, best_ml is
    still None (and best_bet is too, so nothing is stranded)."""
    scorer = _scorer_with_markets([BOOK_B, FakeMarket("moneyline", None, 1.70, 2.20)])
    prediction = _prediction()

    await scorer._calculate_market_values(prediction, FakeGame("g1", "HOM", "AWY"))

    assert prediction.best_ml is None
    assert prediction.best_bet is None
