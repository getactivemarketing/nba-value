# tests/unit/test_nfl_nflverse_lines.py
"""Fallback line source: nflverse schedules when The Odds API is unavailable.

Written 2026-09-22, after the Odds API key sat deactivated for 15 days and
Weeks 1-2 were collected with zero snapshots (every game skipped on stale
lines). nflverse publishes a spread, total and both moneylines per game for
free, so collection no longer depends on one paid key being healthy.
"""
from datetime import datetime, timedelta, timezone

import pandas as pd

from src.services.nfl import nflverse_lines as nv


def _sched():
    return pd.DataFrame([
        # nflverse spread_line is the HOME expected margin: +6 = home favoured by 6.
        {"game_id": "2026_03_ATL_GB", "game_type": "REG", "season": 2026, "week": 3,
         "home_team": "GB", "away_team": "ATL", "spread_line": 6.0, "total_line": 43.5,
         "home_moneyline": -290.0, "away_moneyline": 235.0},
        {"game_id": "2026_03_CAR_CLE", "game_type": "REG", "season": 2026, "week": 3,
         "home_team": "CLE", "away_team": "CAR", "spread_line": -2.5, "total_line": 42.5,
         "home_moneyline": 124.0, "away_moneyline": -148.0},
        # no lines posted yet -> not a market
        {"game_id": "2026_04_NYJ_MIA", "game_type": "REG", "season": 2026, "week": 4,
         "home_team": "MIA", "away_team": "NYJ", "spread_line": None, "total_line": None,
         "home_moneyline": None, "away_moneyline": None},
        # postseason/preseason rows are not part of the tracked board
        {"game_id": "2026_00_X_Y", "game_type": "PRE", "season": 2026, "week": 0,
         "home_team": "MIA", "away_team": "NYJ", "spread_line": 3.0, "total_line": 40.0,
         "home_moneyline": -150.0, "away_moneyline": 130.0},
    ])


def test_american_to_decimal_both_signs():
    assert nv.american_to_decimal(-110) == 1.9091
    assert nv.american_to_decimal(235) == 3.35
    assert nv.american_to_decimal(None) is None


def test_lines_from_schedule_keeps_only_priced_regular_season_games():
    out = {l["game_id"]: l for l in nv.lines_from_schedule(_sched())}
    assert set(out) == {"2026_03_ATL_GB", "2026_03_CAR_CLE"}
    gb = out["2026_03_ATL_GB"]
    assert gb["spread_line"] == 6.0 and gb["total_line"] == 43.5
    assert gb["ml_home"] == nv.american_to_decimal(-290)
    assert gb["ml_away"] == nv.american_to_decimal(235)


def test_market_rows_carry_the_house_price_and_are_labelled_nflverse():
    """nflverse posts no juice on spread/total, so those rows carry a standard
    -110. The book label is what lets any CLV analysis exclude them: a constant
    fabricated price cannot support price CLV, only line CLV."""
    rows = nv.market_rows(nv.lines_from_schedule(_sched())[0])
    by_type = {r["market_type"]: r for r in rows}

    assert set(by_type) == {"spread", "total", "moneyline"}
    assert all(r["book"] == nv.BOOK for r in rows)
    assert nv.BOOK == "nflverse"

    # NFLMarket.line convention: spread positive = home favoured (same as nflverse)
    assert by_type["spread"]["line"] == 6.0
    assert by_type["spread"]["home_odds"] == nv.STANDARD_PRICE_DECIMAL
    assert by_type["spread"]["away_odds"] == nv.STANDARD_PRICE_DECIMAL
    assert by_type["total"]["line"] == 43.5
    assert by_type["total"]["over_odds"] == nv.STANDARD_PRICE_DECIMAL
    # real prices where nflverse actually has them
    assert by_type["moneyline"]["home_odds"] == nv.american_to_decimal(-290)


def test_quote_signs_the_spread_the_way_odds_snapshots_do():
    """nfl_odds_snapshots.spread_line is NEGATIVE when home is favoured
    (settlement.py), the opposite of nflverse's convention. Getting this
    backwards would flip every shadow spread candidate."""
    q = nv.quote_from_line(nv.lines_from_schedule(_sched())[0])
    assert q.spread_line == -6.0
    assert q.total_line == 43.5
    assert q.ml_home == nv.american_to_decimal(-290)

    dog = nv.quote_from_line(nv.lines_from_schedule(_sched())[1])
    assert dog.spread_line == 2.5  # home underdog by 2.5


def test_upcoming_only_filter_excludes_kicked_off_games():
    """Lines for played games are closing values; writing them as a fresh
    capture would backfill history that was never observed pre-kick."""
    now = datetime(2026, 9, 24, 18, 0, tzinfo=timezone.utc)
    kickoffs = {
        "2026_03_ATL_GB": now + timedelta(hours=6),
        "2026_03_CAR_CLE": now - timedelta(hours=1),
    }
    lines = nv.lines_from_schedule(_sched())

    kept = [l["game_id"] for l in nv.upcoming_only(lines, kickoffs, now=now)]

    assert kept == ["2026_03_ATL_GB"]


def test_scorer_produces_candidates_from_nflverse_rows():
    """Contract check: the scorer skips any market missing BOTH side prices
    (`if mt == "spread" and m.get("home_odds") and m.get("away_odds")`), so
    fallback rows that left spread/total unpriced would score nothing at all
    and the fallback would collect exactly as little as no fallback."""
    import numpy as np

    from src.services.nfl.scorer import score_game
    from src.services.nfl.training_data import MOV_FEATURES, TOTALS_FEATURES

    class _Booster:
        def __init__(self, val): self.val = val
        def predict(self, frame, num_iteration=None):
            return np.array([self.val] * len(frame))

    mov = {"model": _Booster(3.0), "feature_cols": MOV_FEATURES, "resid_std": 12.5,
           "calibrator": None}
    totals = {"model": _Booster(47.0), "feature_cols": TOTALS_FEATURES, "resid_std": 13.7,
              "calibrator": None}
    feature_row = {f: 0.0 for f in set(MOV_FEATURES) | set(TOTALS_FEATURES)}

    line = nv.lines_from_schedule(_sched())[0]
    scored = score_game(feature_row, nv.market_rows(line), mov, totals)

    assert scored["predicted_margin"] == 3.0 and scored["predicted_total"] == 47.0
    # model says home by 3 while the board says home by 6 -> an away-side read,
    # and 47 against a 43.5 total -> an over read. Both markets must be live.
    assert scored["best_spread"] is not None or scored["best_total"] is not None
    assert scored["best_total"] is not None
