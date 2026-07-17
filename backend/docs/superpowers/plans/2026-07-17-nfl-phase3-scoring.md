# NFL Model — Phase 3: Scorer + Value Calculator + Snapshots Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the Phase 2 models into a scoring engine — a value calculator (mirroring the MLB machinery, with NFL-calibrated gating), a scorer that values live markets, and graded `nfl_prediction_snapshots` — with the whole pipeline validated by a backtest run *through the real gate* before it ships. Totals go in `best_bet`; spread/ML are shadow-recorded.

**Architecture:** `nfl_markets` holds per-game odds rows. The scorer loads the calibrated MOV + totals bundles, projects each game, converts model probabilities + market odds into per-bet value via `value_calculator`, selects `best_spread`/`best_ml`/`best_total`, and picks `best_bet` from the config-enabled markets (totals only, at launch). Results are frozen into `nfl_prediction_snapshots` and graded post-game at flat $100 units. Live odds ingestion and the weekly scheduler are Phase 4; Phase 3 validates end-to-end against historical nflverse lines.

**Tech Stack:** Python 3.11+, LightGBM, scikit-learn (IsotonicRegression), scipy, pandas, joblib, SQLAlchemy 2.0 async, structlog, pytest.

## Global Constraints

- **Naming:** NFL tables/models/files/config prefixed `nfl_` / `NFL`.
- **Gate is NFL-calibrated, NOT MLB-copied (design decision, 2026-07-17):** The MLB gate's `raw_edge ≥ 0.10` floor selects NFL totals' *losing* high-edge band (P2 reliability: 0.05–0.10 → 56.2%, 0.15+ → 49.1%). NFL reuses the value_calculator *machinery* (`edge_pct`, `gate_score` formula shape, `sort_score`, tanh display `value_score`, `MAX_EDGE_PCT` blowup cap) **verbatim in structure**, but the NFL thresholds (`MIN_EDGE`, an edge **ceiling**, and probability calibration) are **fit empirically** in Task 4 and only shipped if the backtest-through-the-gate is profitable. Do not hardcode MLB's 0.10 floor for NFL.
- **Market gating (inverted vs MLB):** totals → `best_bet`; spread + ML → shadow (recorded in `best_spread`/`best_ml`, excluded from `best_bet`). Config: `nfl_totals_in_best_bet=True`, `nfl_spread_in_best_bet=False`, `nfl_ml_in_best_bet=False`.
- **Probabilities:** reuse `services/ml/probability.py` (residual→prob) and `services/ml/calibration.py::CalibrationLayer` (isotonic). No new norm.cdf math.
- **No leakage in validation:** the Task-4 backtest stays walk-forward (train/calibrate on `season < S`, evaluate `season == S`).
- **Odds convention:** decimal odds internally; `market_prob = 1/decimal`, devigged two-way via `MLBValueCalculator.devig_odds`-equivalent. nflverse `spread_line` = home favored by N (positive); `total_line` = posted total; moneylines are American.
- **Git staging:** stage specific files only. Never `git add -A` / `git add .`.
- **Push:** commit locally; do not push unless explicitly asked.
- **Commit trailer:** `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- **DB:** no local Postgres — read/write the prod DB via `export DATABASE_URL=$(grep -oE "postgresql://[^\"']+" src/tasks/prediction_tracker.py | head -1)`. The `nfl_markets`/`nfl_prediction_snapshots` tables are new and isolated.

## File Structure

**Create:**
- `src/models/nfl_market.py`, `src/models/nfl_prediction_snapshot.py` — the two new tables.
- `src/services/nfl/value_calculator.py` — per-bet value math (mirrors MLB structure; NFL market types + config gating).
- `src/services/nfl/scorer.py` — project a game, value its markets, select best per market + best_bet.
- `src/services/nfl/calibration_fit.py` — fit isotonic calibrators on walk-forward OOS predictions; store in the model bundles.
- `src/services/nfl/snapshot.py` — build + grade `nfl_prediction_snapshots` rows.
- `src/tasks/nfl_score_backtest.py` — the gate-tuning / calibration backtest CLI (Task 4, exit gate).
- `src/tasks/nfl_dryrun_score.py` — dry-run: score a historical week, write + grade snapshots (Task 5).
- `tests/unit/test_nfl_value_calculator.py`, `test_nfl_scorer.py`, `test_nfl_snapshot.py`, `test_nfl_calibration_fit.py`.

**Modify:**
- `src/models/__init__.py` — register the two models.
- `src/config.py` — NFL gating + threshold config.
- `src/services/nfl/model_training.py` — extend the bundle to optionally carry a calibrator (Task 2/4).

**Reuse (do not modify):** `services/ml/probability.py`, `services/ml/calibration.py`.

**Out of scope (Phase 4):** The Odds API live ingestion into `nfl_markets`, the weekly scheduler, `api/nfl.py`. Phase 3 validates against nflverse historical lines synthesized into market rows.

---

### Task 1: `nfl_markets` + `nfl_prediction_snapshots` models

**Files:**
- Create: `src/models/nfl_market.py`, `src/models/nfl_prediction_snapshot.py`
- Modify: `src/models/__init__.py`
- Test: `tests/unit/test_nfl_scoring_models_import.py`

**Interfaces:**
- Produces (mirroring `mlb_market` / `mlb_prediction_snapshot`):
  - `NFLMarket(market_id PK, game_id FK->nfl_games, market_type[spread|moneyline|total], line, home_odds, away_odds, over_odds, under_odds, book, captured_at)`
  - `NFLPredictionSnapshot(id PK, game_id unique, snapshot_time, home_team, away_team, kickoff_utc, predicted_margin, predicted_total, best_spread_{team,line,odds,value_score,edge}, best_ml_{team,odds,value_score,edge}, best_total_{direction,line,odds,value_score,edge}, best_bet_{type,team,line,odds,value_score,edge}, actual_winner, home_score, away_score, actual_margin, actual_total, best_spread_{result,profit}, best_ml_{result,profit}, best_total_{result,profit}, best_bet_{result,profit}, game_date)`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_nfl_scoring_models_import.py
def test_nfl_scoring_models_named_and_columns():
    from src.models import NFLMarket, NFLPredictionSnapshot
    assert NFLMarket.__tablename__ == "nfl_markets"
    assert NFLPredictionSnapshot.__tablename__ == "nfl_prediction_snapshots"
    mcols = set(NFLMarket.__table__.columns.keys())
    assert {"game_id", "market_type", "line", "home_odds", "away_odds",
            "over_odds", "under_odds"}.issubset(mcols)
    scols = set(NFLPredictionSnapshot.__table__.columns.keys())
    assert {"best_spread_team", "best_ml_team", "best_total_direction",
            "best_bet_type", "best_bet_profit", "actual_margin", "actual_total"}.issubset(scols)


def test_snapshot_game_id_unique():
    from src.models import NFLPredictionSnapshot
    assert NFLPredictionSnapshot.__table__.columns["game_id"].unique
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/test_nfl_scoring_models_import.py -v`
Expected: FAIL with `ImportError: cannot import name 'NFLMarket'`.

- [ ] **Step 3: Write the models**

```python
# src/models/nfl_market.py
"""NFL betting market odds rows."""
from datetime import datetime
from sqlalchemy import String, Integer, Float, DateTime, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column
from src.database import Base


class NFLMarket(Base):
    __tablename__ = "nfl_markets"

    market_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    game_id: Mapped[str] = mapped_column(String(20), ForeignKey("nfl_games.game_id"), nullable=False, index=True)
    market_type: Mapped[str] = mapped_column(String(20), nullable=False)  # spread|moneyline|total
    line: Mapped[float | None] = mapped_column(Float, nullable=True)      # spread (home fav +) or total
    home_odds: Mapped[float | None] = mapped_column(Float, nullable=True)  # decimal
    away_odds: Mapped[float | None] = mapped_column(Float, nullable=True)
    over_odds: Mapped[float | None] = mapped_column(Float, nullable=True)
    under_odds: Mapped[float | None] = mapped_column(Float, nullable=True)
    book: Mapped[str | None] = mapped_column(String(50), nullable=True)
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
```

```python
# src/models/nfl_prediction_snapshot.py
"""Frozen NFL pre-game predictions + graded results (flat $100 units)."""
from datetime import datetime, date
from sqlalchemy import String, Integer, Float, DateTime, Date
from sqlalchemy.orm import Mapped, mapped_column
from src.database import Base


class NFLPredictionSnapshot(Base):
    __tablename__ = "nfl_prediction_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    game_id: Mapped[str] = mapped_column(String(20), nullable=False, unique=True, index=True)
    snapshot_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    home_team: Mapped[str] = mapped_column(String(5), nullable=False)
    away_team: Mapped[str] = mapped_column(String(5), nullable=False)
    kickoff_utc: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    game_date: Mapped[date | None] = mapped_column(Date, nullable=True, index=True)

    predicted_margin: Mapped[float | None] = mapped_column(Float, nullable=True)
    predicted_total: Mapped[float | None] = mapped_column(Float, nullable=True)

    # best per market (spread + ML are shadow; total is live)
    best_spread_team: Mapped[str | None] = mapped_column(String(5), nullable=True)
    best_spread_line: Mapped[float | None] = mapped_column(Float, nullable=True)
    best_spread_odds: Mapped[float | None] = mapped_column(Float, nullable=True)
    best_spread_value_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    best_spread_edge: Mapped[float | None] = mapped_column(Float, nullable=True)

    best_ml_team: Mapped[str | None] = mapped_column(String(5), nullable=True)
    best_ml_odds: Mapped[float | None] = mapped_column(Float, nullable=True)
    best_ml_value_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    best_ml_edge: Mapped[float | None] = mapped_column(Float, nullable=True)

    best_total_direction: Mapped[str | None] = mapped_column(String(10), nullable=True)  # over|under
    best_total_line: Mapped[float | None] = mapped_column(Float, nullable=True)
    best_total_odds: Mapped[float | None] = mapped_column(Float, nullable=True)
    best_total_value_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    best_total_edge: Mapped[float | None] = mapped_column(Float, nullable=True)

    best_bet_type: Mapped[str | None] = mapped_column(String(20), nullable=True)  # spread|moneyline|total
    best_bet_team: Mapped[str | None] = mapped_column(String(10), nullable=True)
    best_bet_line: Mapped[float | None] = mapped_column(Float, nullable=True)
    best_bet_odds: Mapped[float | None] = mapped_column(Float, nullable=True)
    best_bet_value_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    best_bet_edge: Mapped[float | None] = mapped_column(Float, nullable=True)

    # graded
    actual_winner: Mapped[str | None] = mapped_column(String(5), nullable=True)
    home_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    actual_margin: Mapped[int | None] = mapped_column(Integer, nullable=True)
    actual_total: Mapped[int | None] = mapped_column(Integer, nullable=True)
    best_spread_result: Mapped[str | None] = mapped_column(String(20), nullable=True)
    best_spread_profit: Mapped[float | None] = mapped_column(Float, nullable=True)
    best_ml_result: Mapped[str | None] = mapped_column(String(20), nullable=True)
    best_ml_profit: Mapped[float | None] = mapped_column(Float, nullable=True)
    best_total_result: Mapped[str | None] = mapped_column(String(20), nullable=True)
    best_total_profit: Mapped[float | None] = mapped_column(Float, nullable=True)
    best_bet_result: Mapped[str | None] = mapped_column(String(20), nullable=True)
    best_bet_profit: Mapped[float | None] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
```

- [ ] **Step 4: Register the models**

In `src/models/__init__.py`, after the NFL block, add imports `from src.models.nfl_market import NFLMarket` and `from src.models.nfl_prediction_snapshot import NFLPredictionSnapshot`, plus `"NFLMarket"`, `"NFLPredictionSnapshot"` in `__all__`.

- [ ] **Step 5: Run test to verify it passes**

Run: `python3 -m pytest tests/unit/test_nfl_scoring_models_import.py -v`
Expected: PASS (2 tests).

- [ ] **Step 6: Create the tables on prod (controlled)**

Run:

```bash
export DATABASE_URL=$(grep -oE "postgresql://[^\"']+" src/tasks/prediction_tracker.py | head -1)
python3 -c "
import asyncio
from src.database import engine, Base
from src.models import NFLMarket, NFLPredictionSnapshot
async def main():
    async with engine.begin() as c:
        await c.run_sync(Base.metadata.create_all, tables=[NFLMarket.__table__, NFLPredictionSnapshot.__table__])
    await engine.dispose()
    print('nfl_markets + nfl_prediction_snapshots created')
asyncio.run(main())
"
```

Expected: prints the confirmation; MLB/NBA tables untouched.

- [ ] **Step 7: Commit**

```bash
git add src/models/nfl_market.py src/models/nfl_prediction_snapshot.py src/models/__init__.py tests/unit/test_nfl_scoring_models_import.py
git commit -m "feat(nfl): nfl_markets + nfl_prediction_snapshots models

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `value_calculator` — per-bet value (MLB machinery, NFL gating)

**Files:**
- Create: `src/services/nfl/value_calculator.py`
- Modify: `src/config.py`
- Test: `tests/unit/test_nfl_value_calculator.py`

**Interfaces:**
- Consumes: config thresholds.
- Produces:
  - `NFLValueResult` dataclass: `market_type, bet_type, team, line, model_prob, market_prob, raw_edge, edge_pct, value_score, confidence, odds_decimal, is_value_bet, sort_score`.
  - `NFLValueCalculator.calculate_value(market_type, bet_type, model_prob, market_prob, odds_decimal, team=None, line=None, model_confidence=0.5) -> NFLValueResult`.
  - `NFLValueCalculator.find_best_value(values) -> NFLValueResult | None` (max by `sort_score` among `is_value_bet`).
  - `NFLValueCalculator.find_best_bet(values, enabled_market_types) -> NFLValueResult | None` (filter to `enabled_market_types`, then `find_best_value`).
  - `devig_two_way(odds1, odds2) -> tuple[float, float]` (multiplicative).
- **Machinery copied verbatim in structure from `MLBValueCalculator`:** `edge_pct = raw_edge/market_prob*100`; `confidence_multiplier = 0.8 + model_confidence*0.4`; `market_multiplier` (moneyline 0.95, total 0.90, else 1.0); `gate_score = edge_pct * EDGE_SCALE_FACTOR(4.0) * conf * market (+5 if model_prob>0.65 and raw_edge>0.03), clamped 0-100`; `sort_score = edge_pct * conf * market`; display `value_score = 100*tanh(edge_pct*(1-0.50)/20)*conf*market (+5 fav bonus), clamped`.
- **NFL-calibrated gate (thresholds from config; defaults are placeholders tuned in Task 4):** `is_value_bet = gate_score >= NFL_MODERATE_THRESHOLD AND raw_edge >= nfl_min_edge AND edge_pct <= NFL_MAX_EDGE_PCT AND raw_edge <= nfl_max_edge` — note the added **edge ceiling** `nfl_max_edge` (NFL totals profit is in a band, not a floor). `NFL_MAX_EDGE_PCT = 80.0` (blowup cap, same as MLB).

- [ ] **Step 1: Add config**

In `src/config.py`, near the model-path settings:

```python
    # NFL value gate (calibrated in Phase 3 Task 4 backtest; placeholders here)
    nfl_min_edge: float = 0.03            # floor
    nfl_max_edge: float = 0.12            # ceiling (NFL totals profit in ~0.05-0.10 band)
    nfl_moderate_threshold: float = 40.0  # gate_score qualification
    nfl_totals_in_best_bet: bool = True
    nfl_spread_in_best_bet: bool = False
    nfl_ml_in_best_bet: bool = False
```

- [ ] **Step 2: Write the failing test**

```python
# tests/unit/test_nfl_value_calculator.py
import math
from src.services.nfl.value_calculator import NFLValueCalculator, NFLValueResult


def calc(market_type="total", model_prob=0.58, market_prob=0.50, odds=1.909, conf=0.6):
    bt = {"total": "over", "spread": "home_spread", "moneyline": "home_ml"}[market_type]
    return NFLValueCalculator.calculate_value(
        market_type=market_type, bet_type=bt, model_prob=model_prob,
        market_prob=market_prob, odds_decimal=odds, model_confidence=conf)


def test_edge_and_scores_follow_mlb_formula():
    r = calc(model_prob=0.58, market_prob=0.50)
    assert round(r.raw_edge, 3) == 0.08
    assert round(r.edge_pct, 1) == 16.0            # 0.08/0.50*100
    # value_score uses tanh of regressed edge -> never pegs at 100
    assert 0 < r.value_score < 100
    # sort_score is unclamped edge_pct*conf*market (total market_mult 0.90)
    conf_mult = 0.8 + 0.6 * 0.4                     # 1.04
    assert round(r.sort_score, 2) == round(16.0 * conf_mult * 0.90, 2)


def test_edge_ceiling_rejects_overconfident_pick():
    # raw_edge 0.20 exceeds the NFL ceiling (nfl_max_edge default 0.12) -> not a value bet
    r = calc(model_prob=0.70, market_prob=0.50)
    assert r.raw_edge >= 0.12
    assert r.is_value_bet is False


def test_in_band_pick_qualifies():
    r = calc(model_prob=0.57, market_prob=0.50)   # raw_edge 0.07, in [0.03, 0.12]
    assert r.is_value_bet is True


def test_find_best_bet_respects_enabled_markets():
    over = calc("total", model_prob=0.57)          # qualifies, total
    spread = NFLValueCalculator.calculate_value(
        "spread", "home_spread", 0.60, 0.50, 1.909, model_confidence=0.6)  # bigger edge
    # Only totals enabled -> best_bet is the total even though spread edge is larger
    best = NFLValueCalculator.find_best_bet([over, spread], enabled_market_types={"total"})
    assert best is not None and best.market_type == "total"
    # Spread enabled too -> the larger-edge spread wins by sort_score
    best2 = NFLValueCalculator.find_best_bet([over, spread], enabled_market_types={"total", "spread"})
    assert best2.market_type == "spread"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/test_nfl_value_calculator.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.services.nfl.value_calculator'`.

- [ ] **Step 4: Write minimal implementation**

```python
# src/services/nfl/value_calculator.py
"""NFL per-bet value. Machinery mirrors MLBValueCalculator verbatim; the
qualification gate is NFL-calibrated (a floor AND a ceiling, plus a blowup cap)
because NFL totals profit lives in a mid edge band, not above a floor. Thresholds
come from config (tuned in the Phase-3 backtest)."""
import math
from dataclasses import dataclass

from src.config import settings

EDGE_SCALE_FACTOR = 4.0
MARKET_REGRESSION_WEIGHT = 0.50
DISPLAY_TANH_SCALE = 20.0
NFL_MAX_EDGE_PCT = 80.0  # blowup cap (same as MLB)


@dataclass
class NFLValueResult:
    market_type: str
    bet_type: str
    team: str | None
    line: float | None
    model_prob: float
    market_prob: float
    raw_edge: float
    edge_pct: float
    value_score: float
    confidence: str
    odds_decimal: float
    is_value_bet: bool
    sort_score: float = 0.0


class NFLValueCalculator:
    @classmethod
    def calculate_value(cls, market_type, bet_type, model_prob, market_prob,
                        odds_decimal, team=None, line=None, model_confidence=0.5):
        raw_edge = model_prob - market_prob
        edge_pct = (raw_edge / market_prob) * 100 if market_prob > 0 else 0.0
        confidence_multiplier = 0.8 + (model_confidence * 0.4)
        market_multiplier = 1.0
        if market_type == "moneyline":
            market_multiplier = 0.95
        elif market_type == "total":
            market_multiplier = 0.90

        gate_score = edge_pct * EDGE_SCALE_FACTOR * confidence_multiplier * market_multiplier
        if model_prob > 0.65 and raw_edge > 0.03:
            gate_score += 5
        gate_score = max(0, min(100, gate_score))

        sort_score = edge_pct * confidence_multiplier * market_multiplier

        blended_edge_pct = edge_pct * (1.0 - MARKET_REGRESSION_WEIGHT)
        value_score = 100.0 * math.tanh(blended_edge_pct / DISPLAY_TANH_SCALE) \
            * confidence_multiplier * market_multiplier
        adjusted_model_prob = market_prob + raw_edge * (1.0 - MARKET_REGRESSION_WEIGHT)
        if adjusted_model_prob > 0.65 and raw_edge > 0.03:
            value_score += 5
        value_score = max(0, min(100, value_score))

        if raw_edge >= 0.08 and model_confidence >= 0.6:
            confidence = "high"
        elif raw_edge >= 0.04 and model_confidence >= 0.4:
            confidence = "medium"
        else:
            confidence = "low"

        is_value = (
            gate_score >= settings.nfl_moderate_threshold
            and raw_edge >= settings.nfl_min_edge
            and raw_edge <= settings.nfl_max_edge   # NFL edge CEILING
            and edge_pct <= NFL_MAX_EDGE_PCT
        )
        return NFLValueResult(
            market_type=market_type, bet_type=bet_type, team=team, line=line,
            model_prob=model_prob, market_prob=market_prob, raw_edge=raw_edge,
            edge_pct=edge_pct, value_score=round(value_score, 1), confidence=confidence,
            odds_decimal=odds_decimal, is_value_bet=is_value, sort_score=round(sort_score, 2))

    @classmethod
    def find_best_value(cls, values):
        vb = [v for v in values if v.is_value_bet]
        return max(vb, key=lambda v: v.sort_score) if vb else None

    @classmethod
    def find_best_bet(cls, values, enabled_market_types):
        return cls.find_best_value([v for v in values if v.market_type in enabled_market_types])

    @staticmethod
    def devig_two_way(odds1, odds2):
        p1, p2 = 1 / odds1, 1 / odds2
        total = p1 + p2
        return p1 / total, p2 / total
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python3 -m pytest tests/unit/test_nfl_value_calculator.py -v`
Expected: PASS (4 tests).

- [ ] **Step 6: Commit**

```bash
git add src/services/nfl/value_calculator.py src/config.py tests/unit/test_nfl_value_calculator.py
git commit -m "feat(nfl): value calculator (MLB machinery, NFL-calibrated gate)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: `scorer` — project a game and value its markets

**Files:**
- Create: `src/services/nfl/scorer.py`
- Test: `tests/unit/test_nfl_scorer.py`

**Interfaces:**
- Consumes: `model_training.{load_bundle, predict_mov}`, `services/ml/probability.py`, `value_calculator.*`, `config`.
- Produces:
  - `score_game(feature_row, market_rows, mov_bundle, totals_bundle) -> dict` — returns `{predicted_margin, predicted_total, best_spread, best_ml, best_total, best_bet}` (each best_* is an `NFLValueResult | None`). `feature_row` is one row (dict/Series) with the model feature columns; `market_rows` is a list of dicts `{market_type, line, home_odds, away_odds, over_odds, under_odds}`.
  - Internally: predict margin (mov_bundle) + total (totals_bundle); for each market row build the candidate bets:
    - spread: `mov_to_spread_prob(pred_margin, -line, mov_resid_std)` → home cover prob; devig `home_odds`/`away_odds`; two `NFLValueResult`s (home_spread/away_spread).
    - moneyline: `mov_to_moneyline_prob(pred_margin, mov_resid_std)`; devig home/away ML.
    - total: `mov_to_total_prob(pred_total, 0.0, line, totals_resid_std)` → over prob; devig `over_odds`/`under_odds`; over/under results.
  - `best_bet = find_best_bet(all_results, enabled)` where `enabled` is built from config (`nfl_totals_in_best_bet` → "total", etc.).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_nfl_scorer.py
from src.services.nfl.scorer import score_game


class _Booster:
    def __init__(self, val): self.val = val
    best_iteration = 1
    def predict(self, X, num_iteration=None):
        return [self.val] * len(X)


def _bundle(pred, cols, std):
    return {"model": _Booster(pred), "feature_cols": cols, "resid_std": std}


def test_score_game_picks_total_as_best_bet_when_only_totals_enabled(monkeypatch):
    from src.services.nfl import value_calculator as vc
    # loosen gate for the test so the in-band total qualifies
    monkeypatch.setattr(vc.settings, "nfl_min_edge", 0.02, raising=False)
    monkeypatch.setattr(vc.settings, "nfl_max_edge", 0.20, raising=False)
    monkeypatch.setattr(vc.settings, "nfl_moderate_threshold", 5.0, raising=False)
    monkeypatch.setattr(vc.settings, "nfl_totals_in_best_bet", True, raising=False)
    monkeypatch.setattr(vc.settings, "nfl_spread_in_best_bet", False, raising=False)
    monkeypatch.setattr(vc.settings, "nfl_ml_in_best_bet", False, raising=False)

    feat = {c: 0.0 for c in ["off_epa_diff", "def_epa_diff", "pass_epa_diff",
            "rush_epa_diff", "success_rate_diff", "pace_diff", "power_diff",
            "rest_diff", "is_divisional", "is_primetime", "spread_line",
            "off_epa_sum", "pace_sum", "is_dome", "wind_mph", "temp_f", "total_line"]}
    feat["spread_line"] = 3.0
    feat["total_line"] = 44.0
    # model predicts total 52 (well over 44) -> strong over edge; margin ~ line (no spread edge)
    mov = _bundle(3.0, [c for c in feat if c in (
        "off_epa_diff","def_epa_diff","pass_epa_diff","rush_epa_diff","success_rate_diff",
        "pace_diff","power_diff","rest_diff","is_divisional","is_primetime","spread_line")], 13.0)
    tot = _bundle(52.0, ["off_epa_sum","pace_sum","pass_epa_diff","is_dome",
                         "wind_mph","temp_f","total_line"], 13.7)
    markets = [
        {"market_type": "spread", "line": 3.0, "home_odds": 1.909, "away_odds": 1.909},
        {"market_type": "moneyline", "line": None, "home_odds": 1.6, "away_odds": 2.5},
        {"market_type": "total", "line": 44.0, "over_odds": 1.909, "under_odds": 1.909},
    ]
    out = score_game(feat, markets, mov, tot)
    assert round(out["predicted_total"]) == 52
    assert out["best_total"] is not None and out["best_total"].bet_type == "over"
    assert out["best_bet"] is not None and out["best_bet"].market_type == "total"
    # spread edge ~0 -> best_spread may be None or not the best_bet
    assert out["best_bet"].market_type == "total"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/test_nfl_scorer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.services.nfl.scorer'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/services/nfl/scorer.py
"""Score an NFL game: project margin + total, value each market, select bests."""
import pandas as pd
import structlog

from src.config import settings
from src.services.nfl.model_training import predict_mov
from src.services.nfl.value_calculator import NFLValueCalculator
from src.services.ml.probability import (
    mov_to_spread_prob, mov_to_moneyline_prob, mov_to_total_prob)

logger = structlog.get_logger()


def _enabled_markets() -> set[str]:
    e = set()
    if settings.nfl_totals_in_best_bet:
        e.add("total")
    if settings.nfl_spread_in_best_bet:
        e.add("spread")
    if settings.nfl_ml_in_best_bet:
        e.add("moneyline")
    return e


def score_game(feature_row, market_rows, mov_bundle, totals_bundle) -> dict:
    frame = pd.DataFrame([feature_row])
    pred_margin = float(predict_mov(mov_bundle, frame)[0])
    pred_total = float(predict_mov(totals_bundle, frame)[0])
    mstd, tstd = mov_bundle["resid_std"], totals_bundle["resid_std"]
    calc = NFLValueCalculator
    results = []

    for m in market_rows:
        mt = m["market_type"]
        if mt == "spread" and m.get("home_odds") and m.get("away_odds"):
            p_home = mov_to_spread_prob(pred_margin, -float(m["line"]), mstd)
            mh, ma = calc.devig_two_way(m["home_odds"], m["away_odds"])
            results.append(calc.calculate_value("spread", "home_spread", p_home, mh,
                           m["home_odds"], team="home", line=m["line"], model_confidence=0.6))
            results.append(calc.calculate_value("spread", "away_spread", 1 - p_home, ma,
                           m["away_odds"], team="away", line=-float(m["line"]), model_confidence=0.6))
        elif mt == "moneyline" and m.get("home_odds") and m.get("away_odds"):
            p_home = mov_to_moneyline_prob(pred_margin, mstd)
            mh, ma = calc.devig_two_way(m["home_odds"], m["away_odds"])
            results.append(calc.calculate_value("moneyline", "home_ml", p_home, mh,
                           m["home_odds"], team="home", model_confidence=0.6))
            results.append(calc.calculate_value("moneyline", "away_ml", 1 - p_home, ma,
                           m["away_odds"], team="away", model_confidence=0.6))
        elif mt == "total" and m.get("over_odds") and m.get("under_odds"):
            p_over = mov_to_total_prob(pred_total, 0.0, float(m["line"]), tstd)
            mo, mu = calc.devig_two_way(m["over_odds"], m["under_odds"])
            results.append(calc.calculate_value("total", "over", p_over, mo,
                           m["over_odds"], line=m["line"], model_confidence=0.6))
            results.append(calc.calculate_value("total", "under", 1 - p_over, mu,
                           m["under_odds"], line=m["line"], model_confidence=0.6))

    def best_of(mtype):
        return calc.find_best_value([r for r in results if r.market_type == mtype])

    return {
        "predicted_margin": pred_margin,
        "predicted_total": pred_total,
        "best_spread": best_of("spread"),
        "best_ml": best_of("moneyline"),
        "best_total": best_of("total"),
        "best_bet": calc.find_best_bet(results, _enabled_markets()),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/unit/test_nfl_scorer.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/services/nfl/scorer.py tests/unit/test_nfl_scorer.py
git commit -m "feat(nfl): scorer projects game + values markets + selects best_bet

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Calibration fit + gate-tuning backtest (EXIT GATE)

**Files:**
- Create: `src/services/nfl/calibration_fit.py`, `src/tasks/nfl_score_backtest.py`
- Test: `tests/unit/test_nfl_calibration_fit.py`

**Interfaces:**
- Produces:
  - `fit_isotonic(raw_probs, outcomes) -> IsotonicRegression` — thin wrapper (reuse `sklearn.isotonic.IsotonicRegression(out_of_bounds="clip")`), returns a fitted calibrator; `apply(cal, probs) -> np.ndarray`.
  - `src/tasks/nfl_score_backtest.py`: walk-forward over 2019-2024 that, per test season, trains MOV+totals on prior seasons, **fits isotonic calibrators on the prior seasons' out-of-sample predictions**, scores the test season's games *through the real `NFLValueCalculator` gate* (building synthetic market rows from nflverse lines at -110 / actual moneylines), and reports per-market qualified-pick record + units + reliability. It then **sweeps `nfl_min_edge`/`nfl_max_edge`** over a small grid and prints the totals record at each, so the profitable band is chosen from data.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_nfl_calibration_fit.py
import numpy as np
from src.services.nfl.calibration_fit import fit_isotonic, apply_calibration


def test_isotonic_monotonic_and_corrects_overconfidence():
    # raw probs are overconfident: high raw prob doesn't win more.
    rng = np.random.default_rng(0)
    raw = np.concatenate([np.full(200, 0.8), np.full(200, 0.55)])
    # 0.8-bucket actually wins 50%, 0.55-bucket wins 58% (inverted, like NFL totals)
    outcomes = np.concatenate([
        (rng.random(200) < 0.50).astype(int),
        (rng.random(200) < 0.58).astype(int)])
    cal = fit_isotonic(raw, outcomes)
    p = apply_calibration(cal, np.array([0.55, 0.80]))
    # calibration pulls the overconfident 0.80 down toward its true ~0.50
    assert p[1] <= 0.62
    # outputs stay in [0,1]
    assert (p >= 0).all() and (p <= 1).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/test_nfl_calibration_fit.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/services/nfl/calibration_fit.py
"""Isotonic probability calibration for the NFL models (reuses sklearn)."""
import numpy as np
from sklearn.isotonic import IsotonicRegression


def fit_isotonic(raw_probs, outcomes) -> IsotonicRegression:
    cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    cal.fit(np.asarray(raw_probs, dtype=float), np.asarray(outcomes, dtype=float))
    return cal


def apply_calibration(cal: IsotonicRegression, probs) -> np.ndarray:
    return np.clip(cal.predict(np.asarray(probs, dtype=float)), 0.0, 1.0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/unit/test_nfl_calibration_fit.py -v`
Expected: PASS.

- [ ] **Step 5: Write the gate-tuning backtest CLI**

Write `src/tasks/nfl_score_backtest.py` that: builds the 2010-2024 feature frame (`training_data`); for each test season 2019-2024 trains MOV+totals on prior seasons, generates prior-season OOS totals-over probabilities (via an inner split) and fits an isotonic calibrator, then for the test season computes calibrated over-probabilities, devigs the -110 total market, and runs each game's total through `NFLValueCalculator.calculate_value` — collecting qualified picks and grading vs `actual_total`. Report totals record/units/reliability at the **current config band**, then sweep `(min_edge, max_edge)` over `[(0.02,0.08),(0.03,0.10),(0.03,0.12),(0.05,0.10),(0.05,999)]` and print the totals record+units for each. (Spread/ML are computed + reported as shadow, never gated into best_bet.)

Use the same `_WIN_110` flat-unit grading as Phase 2's backtest.

- [ ] **Step 6: Run it against prod (ACCEPTANCE GATE)**

```bash
export DATABASE_URL=$(grep -oE "postgresql://[^\"']+" src/tasks/prediction_tracker.py | head -1)
export DEBUG=false
python3 -m src.tasks.nfl_score_backtest
```

Expected: prints the totals record/units through the real gate at each `(min_edge, max_edge)` band and a reliability table. **Review with the user and set `nfl_min_edge`/`nfl_max_edge` in config to the best profitable band. This is the Phase 3 exit gate — totals must clear break-even (>52.4% or positive units) through the actual gate, or we do not ship totals live.** Report the honest numbers; do not pick a band that only looks good by overfitting a tiny slice — prefer a band with a meaningful sample (n ≥ ~150 over 6 seasons).

- [ ] **Step 7: Commit**

```bash
git add src/services/nfl/calibration_fit.py src/tasks/nfl_score_backtest.py tests/unit/test_nfl_calibration_fit.py
git commit -m "feat(nfl): isotonic calibration + gate-tuning backtest

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Snapshot build + grade (dry-run on a historical week)

**Files:**
- Create: `src/services/nfl/snapshot.py`, `src/tasks/nfl_dryrun_score.py`
- Test: `tests/unit/test_nfl_snapshot.py`

**Interfaces:**
- Produces:
  - `build_snapshot(game, scored) -> dict` — maps a `score_game` result + game identity into an `NFLPredictionSnapshot` kwargs dict (best_spread/ml/total/best_bet fields; NULL where a best_* is None).
  - `grade_snapshot(snap, home_score, away_score, spread_line, total_line) -> dict` — computes `actual_margin`/`actual_total`, and per-market `result`/`profit` (flat $100, -110 for spread/total, actual odds for ML), including `best_bet_*`. Push → result "push", profit 0.
  - `src/tasks/nfl_dryrun_score.py`: pick a completed season+week, build synthetic `nfl_markets` rows from that week's nflverse lines, score each game with the trained bundles, write `nfl_prediction_snapshots`, then grade against the final scores. Prints per-game best_bet + the week's graded record, and asserts (a) no `best_bet_value_score` at 100 (saturation), (b) every `best_bet_type` is `total` (spread/ML shadow), (c) spread/ML recorded in their own columns.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_nfl_snapshot.py
from src.services.nfl.snapshot import grade_snapshot


def test_grade_total_over_win_and_push():
    snap = {"best_total_direction": "over", "best_total_line": 44.0, "best_total_odds": 1.909,
            "best_bet_type": "total", "best_bet_line": 44.0, "best_bet_odds": 1.909,
            "best_bet_team": None}
    g = grade_snapshot(snap, home_score=30, away_score=20, spread_line=3.0, total_line=44.0)
    assert g["actual_total"] == 50 and g["actual_margin"] == 10
    assert g["best_total_result"] == "win"
    assert round(g["best_total_profit"], 1) == 90.9
    assert g["best_bet_result"] == "win"
    # push case
    g2 = grade_snapshot(snap, home_score=22, away_score=22, spread_line=3.0, total_line=44.0)
    assert g2["best_total_result"] == "push" and g2["best_total_profit"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/test_nfl_snapshot.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/services/nfl/snapshot.py
"""Build and grade nfl_prediction_snapshots (flat $100 units)."""
from datetime import datetime, timezone

_WIN_110 = 100 * (100 / 110)


def build_snapshot(game: dict, scored: dict) -> dict:
    def f(res, *attrs):
        return {a: (getattr(res, a) if res else None) for a in attrs}
    bs, bm, bt, bb = scored["best_spread"], scored["best_ml"], scored["best_total"], scored["best_bet"]
    row = {
        "game_id": game["game_id"], "snapshot_time": game.get("snapshot_time") or datetime.now(timezone.utc),
        "home_team": game["home_team"], "away_team": game["away_team"],
        "kickoff_utc": game.get("kickoff_utc"), "game_date": game.get("game_date"),
        "predicted_margin": scored["predicted_margin"], "predicted_total": scored["predicted_total"],
        "best_spread_team": bs.team if bs else None, "best_spread_line": bs.line if bs else None,
        "best_spread_odds": bs.odds_decimal if bs else None,
        "best_spread_value_score": bs.value_score if bs else None, "best_spread_edge": bs.raw_edge if bs else None,
        "best_ml_team": bm.team if bm else None, "best_ml_odds": bm.odds_decimal if bm else None,
        "best_ml_value_score": bm.value_score if bm else None, "best_ml_edge": bm.raw_edge if bm else None,
        "best_total_direction": (bt.bet_type if bt else None), "best_total_line": bt.line if bt else None,
        "best_total_odds": bt.odds_decimal if bt else None,
        "best_total_value_score": bt.value_score if bt else None, "best_total_edge": bt.raw_edge if bt else None,
        "best_bet_type": bb.market_type if bb else None, "best_bet_team": bb.team if bb else None,
        "best_bet_line": bb.line if bb else None, "best_bet_odds": bb.odds_decimal if bb else None,
        "best_bet_value_score": bb.value_score if bb else None, "best_bet_edge": bb.raw_edge if bb else None,
    }
    return row


def _grade_total(direction, line, odds, actual_total):
    if actual_total == line:
        return "push", 0.0
    over_hit = actual_total > line
    won = over_hit if direction == "over" else not over_hit
    return ("win", _WIN_110) if won else ("loss", -100.0)


def _grade_spread(team, line, actual_margin):
    # team is "home"/"away"; line is that side's spread. Home covers iff margin > home_line_abs.
    if team is None or line is None:
        return None, None
    # snapshot stores the picked side's line; home covers iff actual_margin > home_spread_line.
    # For a home pick, line = home spread (home favored positive). For away, line is away's.
    covered = (actual_margin > line) if team == "home" else (actual_margin < line)
    if (team == "home" and actual_margin == line) or (team == "away" and actual_margin == line):
        return "push", 0.0
    return ("win", _WIN_110) if covered else ("loss", -100.0)


def grade_snapshot(snap, home_score, away_score, spread_line, total_line) -> dict:
    actual_margin = home_score - away_score
    actual_total = home_score + away_score
    out = {"actual_margin": actual_margin, "actual_total": actual_total,
           "home_score": home_score, "away_score": away_score}

    if snap.get("best_total_direction"):
        r, p = _grade_total(snap["best_total_direction"], snap["best_total_line"],
                            snap["best_total_odds"], actual_total)
        out["best_total_result"], out["best_total_profit"] = r, p
    if snap.get("best_bet_type") == "total" and snap.get("best_total_direction"):
        out["best_bet_result"], out["best_bet_profit"] = out["best_total_result"], out["best_total_profit"]
    # spread/ML grading (shadow) omitted here for brevity of the failing test path;
    # implement symmetrically using _grade_spread + moneyline profit from odds.
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/unit/test_nfl_snapshot.py -v`
Expected: PASS.

- [ ] **Step 5: Complete spread/ML grading + the dry-run CLI**

Fill in `grade_snapshot` for the shadow markets: spread via `_grade_spread(snap["best_spread_team"], snap["best_spread_line"], actual_margin)`; moneyline via `snap["best_ml_team"]` won iff that side won outright, profit = `(odds_decimal-1)*100` on win else -100. Then write `src/tasks/nfl_dryrun_score.py` per the interface (build the feature frame for one completed season+week via `training_data`, synthesize `nfl_markets` rows from nflverse `spread_line`/`total_line`/moneylines at -110 for spread/total, score with the Task-2 bundles, write `NFLPredictionSnapshot` rows, grade vs final scores).

- [ ] **Step 6: Run the dry-run (validation)**

```bash
export DATABASE_URL=$(grep -oE "postgresql://[^\"']+" src/tasks/prediction_tracker.py | head -1)
export DEBUG=false
python3 -m src.tasks.nfl_dryrun_score 2024 10
```

Expected: writes snapshots for that week, grades them, prints per-game best_bet + the week's total record, and confirms: 0 saturated `best_bet_value_score`, every `best_bet_type == "total"`, spread/ML populated in their shadow columns. **Review with the user.**

- [ ] **Step 7: Commit**

```bash
git add src/services/nfl/snapshot.py src/tasks/nfl_dryrun_score.py tests/unit/test_nfl_snapshot.py
git commit -m "feat(nfl): snapshot build + grade + historical dry-run

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Phase 3 Exit Criteria (review with user before Phase 4)

1. `pytest tests/unit -k nfl` fully green.
2. Task 4 gate-tuning backtest run: totals clears break-even **through the real `NFLValueCalculator` gate** at a data-chosen `(min_edge, max_edge)` band with a meaningful sample; config set accordingly. If totals cannot clear the gate, STOP and report — do not ship a losing market live.
3. Task 5 dry-run on a historical week: snapshots written + graded, 0 saturated best-bet scores, every `best_bet` is a total, spread/ML shadow-recorded.
4. `nfl_markets` + `nfl_prediction_snapshots` tables live on prod, isolated from MLB/NBA.

## Notes / deferred (Phase 4)

- **Live odds ingestion:** The Odds API `americanfootball_nfl` client (subclass `OddsAPIClient`, `SPORT="americanfootball_nfl"`) → `nfl_markets`; the weekly per-game pre-kick snapshot scheduler; `api/nfl.py` (`/picks`, `/games`). Phase 3 uses nflverse historical lines instead.
- **spread_line as a live input:** the P3 scorer needs the current market spread fed as the `spread_line` feature (home-favored positive) — the P4 odds client must map The Odds API spread to this convention.
- If Task 4 shows the isotonic calibration materially changes the band, persist the calibrator into the model bundles (extend `save_bundle`) so live scoring applies it.
