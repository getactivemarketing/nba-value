# MLB Value Retune + Gated Totals Retrain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retune the MLB value algorithm — unclamped best-bet selection, tanh display scores, blowup cap, totals out of best_bet — then retrain the totals model on 2026 data behind a shadow-mode gate.

**Architecture:** All pick-selection logic lives in `MLBValueCalculator` (pure classmethods, no DB). The scorer (`src/services/mlb/scorer.py`) builds candidates and asks the calculator for best bets; the scheduler (`src/tasks/mlb_scheduler.py`) snapshots them. We change the calculator's scoring/selection, add one config flag, add a defense-in-depth check in the scheduler, and add a standalone retrain script for the totals model.

**Tech Stack:** Python 3.11, FastAPI backend on Railway, PostgreSQL (prod on Railway), LightGBM via `MLBModelTrainer`, pytest (`asyncio_mode=auto`, tests in `backend/tests/`).

**Spec:** `backend/docs/superpowers/specs/2026-07-06-mlb-value-retune-design.md` (read it first — the "Revision 2026-07-06" paragraph explains why selection and display are decoupled).

## Global Constraints

- Working directory for all commands: `/Applications/XAMPP/xamppfiles/htdocs/Sites/Truline/backend`
- The repo has UNCOMMITTED NBA changes (`src/services/scoring/scorer.py`, `src/services/social/content.py`, `.gitignore`, top-level `CLAUDE.md`). NEVER `git add -A` or `git add .` — always add explicit file paths. Do not touch those files.
- Pick-qualification gate must be exactly today's formula (clamped `edge_pct * 4.0 * confidence_multiplier * market_multiplier + bonus >= 55` and `raw_edge >= 0.10`) plus the new `edge_pct <= 80` cap. The +160u season pick set must be preserved.
- Market multipliers stay 0.95 (moneyline) / 0.90 (total). The 0.80 ML penalty was tested and rejected (see spec).
- `MIN_EDGE` stays `0.10`. Do not raise it.
- Prod DB connection for verification queries: import `DB_URL` from `src.tasks.prediction_tracker` (existing repo pattern).
- Commits: descriptive message + `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`. Push only in the deploy tasks (5 and 9).

---

### Task 1: Retune MLBValueCalculator — gate cap, sort_score, tanh display

**Files:**
- Modify: `src/services/mlb/value_calculator.py` (constants block lines 56-71, `calculate_value` lines 73-163, `find_best_value` lines 165-183, `MLBValueResult` dataclass lines 10-37)
- Test: `tests/unit/test_mlb_value_calculator.py` (create; `tests/unit/` exists and is empty)

**Interfaces:**
- Consumes: nothing new.
- Produces: `MLBValueResult.sort_score: float` (unclamped selection metric); constants `MAX_EDGE_PCT = 80.0`, `MARKET_REGRESSION_WEIGHT = 0.50`, `DISPLAY_TANH_SCALE = 20.0`, `EDGE_SCALE_FACTOR = 4.0`; `find_best_value(values)` now ranks by `sort_score`. Signature of `calculate_value` is unchanged. Task 2 relies on `sort_score` and `find_best_value`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/__init__.py` as an empty file if imports fail without it (try without first). Create `tests/unit/test_mlb_value_calculator.py`:

```python
"""Unit tests for the retuned MLBValueCalculator (2026-07-06 spec).

Key invariants:
- Qualification gate identical to pre-retune formula, plus MAX_EDGE_PCT blowup cap.
- value_score (display) uses tanh of the market-regressed edge -> no saturation at 100.
- sort_score (selection) is the unclamped edge_pct * confidence * market multipliers.
"""

import math

import pytest

from src.services.mlb.value_calculator import MLBValueCalculator, MLBValueResult


def calc(
    market_type="runline",
    model_prob=0.60,
    market_prob=0.50,
    odds=2.0,
    conf=0.5,
):
    bet_type = {"runline": "home_rl", "moneyline": "home_ml", "total": "over"}[market_type]
    return MLBValueCalculator.calculate_value(
        market_type=market_type,
        bet_type=bet_type,
        model_prob=model_prob,
        market_prob=market_prob,
        odds_decimal=odds,
        team="NYY" if market_type != "total" else None,
        line=1.5 if market_type == "runline" else None,
        model_confidence=conf,
    )


class TestQualificationGate:
    def test_min_edge_pick_still_qualifies(self):
        # raw_edge 0.10, edge_pct 20 -> legacy gate 20*4*1.0 = 80 >= 55
        result = calc(model_prob=0.60, market_prob=0.50)
        assert result.is_value_bet is True

    def test_below_min_edge_rejected(self):
        # raw_edge 0.09 -> gate score 72 passes but MIN_EDGE filter rejects
        result = calc(model_prob=0.59, market_prob=0.50)
        assert result.is_value_bet is False

    def test_blowup_capped(self):
        # raw_edge 0.50, edge_pct 111 > MAX_EDGE_PCT 80 -> rejected
        result = calc(model_prob=0.95, market_prob=0.45, odds=2.22)
        assert result.edge_pct > MLBValueCalculator.MAX_EDGE_PCT
        assert result.is_value_bet is False

    def test_edge_pct_at_cap_boundary_qualifies(self):
        # edge_pct exactly 80 (raw 0.40 / market 0.50) passes (<=)
        result = calc(model_prob=0.90, market_prob=0.50)
        assert result.edge_pct == pytest.approx(80.0)
        assert result.is_value_bet is True


class TestDisplayScore:
    def test_moderate_edge_not_saturated(self):
        # edge_pct 20 -> blended 10 -> 100*tanh(0.5) = 46.2 (conf 0.5 -> mult 1.0)
        result = calc(model_prob=0.60, market_prob=0.50)
        assert result.value_score == pytest.approx(100 * math.tanh(0.5), abs=0.1)
        assert result.value_score < 100

    def test_huge_edge_stays_below_100_without_bonus(self):
        # edge_pct 60 -> blended 30 -> 100*tanh(1.5) = 90.5; adjusted prob
        # 0.50 + 0.30*0.5 = 0.65, not > 0.65 -> no favorite bonus
        result = calc(model_prob=0.80, market_prob=0.50)
        assert result.value_score == pytest.approx(100 * math.tanh(1.5), abs=0.1)

    def test_ml_multiplier_applies_to_display(self):
        rl = calc(market_type="runline", model_prob=0.60, market_prob=0.50)
        ml = calc(market_type="moneyline", model_prob=0.60, market_prob=0.50)
        assert ml.value_score == pytest.approx(rl.value_score * 0.95, abs=0.1)


class TestSortScore:
    def test_sort_score_is_unclamped(self):
        # edge_pct 60 with conf 0.5 (mult 1.0) -> sort_score 60, far above
        # what the old clamped score could express
        result = calc(model_prob=0.80, market_prob=0.50)
        assert result.sort_score == pytest.approx(60.0, abs=0.01)

    def test_find_best_value_ranks_by_sort_score_not_display(self):
        # Both would have clamped to 100 under the old formula; the bigger
        # edge must win regardless of list order.
        smaller = calc(model_prob=0.75, market_prob=0.50)   # edge_pct 50
        bigger = calc(model_prob=0.80, market_prob=0.50)    # edge_pct 60
        best = MLBValueCalculator.find_best_value([smaller, bigger])
        assert best is bigger
        best = MLBValueCalculator.find_best_value([bigger, smaller])
        assert best is bigger

    def test_find_best_value_still_requires_is_value_bet(self):
        blowup = calc(model_prob=0.95, market_prob=0.45, odds=2.22)  # capped
        ok = calc(model_prob=0.60, market_prob=0.50)
        best = MLBValueCalculator.find_best_value([blowup, ok])
        assert best is ok
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_mlb_value_calculator.py -v`
Expected: FAIL — `AttributeError: MAX_EDGE_PCT` / `sort_score` missing / saturation assertions failing.

- [ ] **Step 3: Implement the calculator changes**

In `src/services/mlb/value_calculator.py`:

3a. Add `import math` after the existing imports (top of file).

3b. Add `sort_score` to the dataclass, after the `is_value_bet` field:

```python
    # Is this a recommended bet?
    is_value_bet: bool

    # Unclamped selection metric: edge_pct * confidence * market multipliers.
    # Used to rank candidates; value_score is display-only.
    sort_score: float = 0.0
```

3c. Replace the constants block (currently lines 56-71, from the `MIN_EDGE` comment through `EDGE_SCALE_FACTOR = 400`) with:

```python
    # Minimum edge required to consider a bet.
    # Tightened from 0.02 -> 0.10 on 2026-04-28 after backtest showed bets with
    # raw_edge < 0.10 had a 20% win rate over 15 graded bets.
    # 2026-07-06: stays 0.10 — the 0.10-0.15 bucket is the best performer
    # since May 23 (56.9% WR, +17.9u over 72 bets). Do not raise.
    MIN_EDGE = 0.10

    # Sanity cap on edge_pct. Real sports-betting edges rarely exceed ~20%;
    # anything above 80% means the model is wildly miscalibrated for that
    # game ("model blowup") — skip the bet.
    MAX_EDGE_PCT = 80.0

    # Maximum decimal odds for a runline pick. Backtest (Apr 3-27) showed RL bets
    # with odds in 2.5-3.0 had 40% win rate over 35 bets (-2.5u). Filter applied
    # in scorer._calculate_market_values.
    MAX_RUNLINE_ODDS = 2.5

    # Value score thresholds
    STRONG_VALUE_THRESHOLD = 65
    MODERATE_VALUE_THRESHOLD = 55

    # Gate scale: the pre-2026-07-06 score formula (edge_pct * 4.0 * multipliers,
    # clamped) is kept verbatim as the QUALIFICATION gate so the proven +160u
    # season pick set is preserved exactly. (Was a dead `= 400` constant; the
    # formula hardcoded 4.0.)
    EDGE_SCALE_FACTOR = 4.0

    # Display score: market-regress the edge 50% toward the market, then
    # compress with tanh so scores spread 30-97 instead of piling up at 100
    # (65% of picks scored exactly 100 before this change). Display only —
    # selection uses sort_score, qualification uses the legacy gate.
    MARKET_REGRESSION_WEIGHT = 0.50
    DISPLAY_TANH_SCALE = 20.0
```

3d. In `calculate_value`, replace the body from `# Base value score from edge` (line ~110) through `value_score = max(0, min(100, value_score))` (line ~133) with:

```python
        # Confidence multiplier (0.8 - 1.2)
        confidence_multiplier = 0.8 + (model_confidence * 0.4)

        # Market type adjustment. 0.95/0.90 retained; an 0.80 ML penalty was
        # backtested 2026-07-06 and REJECTED (-8u vs 0.95 on 586 picks).
        market_multiplier = 1.0
        if market_type == "moneyline":
            market_multiplier = 0.95
        elif market_type == "total":
            market_multiplier = 0.90

        # --- Qualification gate: legacy formula, kept verbatim -------------
        gate_score = edge_pct * cls.EDGE_SCALE_FACTOR * confidence_multiplier * market_multiplier
        if model_prob > 0.65 and raw_edge > 0.03:
            gate_score += 5
        gate_score = max(0, min(100, gate_score))

        # --- Selection metric: unclamped ------------------------------------
        # The clamp made large edges tie at 100 and max() silently preferred
        # whichever market was added to the candidate list first (ML).
        sort_score = edge_pct * confidence_multiplier * market_multiplier

        # --- Display score: regressed + tanh-compressed ----------------------
        blended_edge_pct = edge_pct * (1.0 - cls.MARKET_REGRESSION_WEIGHT)
        value_score = (
            100.0
            * math.tanh(blended_edge_pct / cls.DISPLAY_TANH_SCALE)
            * confidence_multiplier
            * market_multiplier
        )
        # Favorite bonus uses the regressed probability so it tracks the
        # displayed edge (mirrors b004564).
        adjusted_model_prob = market_prob + raw_edge * (1.0 - cls.MARKET_REGRESSION_WEIGHT)
        if adjusted_model_prob > 0.65 and raw_edge > 0.03:
            value_score += 5
        value_score = max(0, min(100, value_score))
```

(The existing `raw_edge` / `edge_pct` computation above this block stays as is. Delete the old `base_score`, old multiplier block, and old bonus block — they are replaced by the above.)

3e. Replace the `is_value` line:

```python
        # Qualification: legacy gate + MIN_EDGE + blowup cap.
        is_value = (
            gate_score >= cls.MODERATE_VALUE_THRESHOLD
            and raw_edge >= cls.MIN_EDGE
            and edge_pct <= cls.MAX_EDGE_PCT
        )
```

3f. Add `sort_score=round(sort_score, 2),` to the `MLBValueResult(...)` constructor call at the end of `calculate_value`.

3g. In `find_best_value`, change the ranking line:

```python
        return max(value_bets, key=lambda v: v.sort_score)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/test_mlb_value_calculator.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/services/mlb/value_calculator.py tests/unit/test_mlb_value_calculator.py
git commit -m "tune(mlb): unclamped best-bet selection, tanh display score, blowup cap

Gate formula unchanged (preserves +160u season pick set) plus edge_pct<=80
cap. Selection now ranks by unclamped sort_score, fixing the tie-at-100
bias toward ML. Display score = tanh of 50%-market-regressed edge: 0%
saturation vs 65% before. Spec: docs/superpowers/specs/2026-07-06-mlb-value-retune-design.md

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Totals out of best_bet — config flag, find_best_bet, scorer wiring

**Files:**
- Modify: `src/config.py` (Model Flags section, ~line 70)
- Modify: `src/services/mlb/value_calculator.py` (add `find_best_bet` classmethod after `find_best_value`)
- Modify: `src/services/mlb/scorer.py:488` (best_bet selection)
- Test: `tests/unit/test_mlb_value_calculator.py` (extend)

**Interfaces:**
- Consumes: `MLBValueResult.sort_score`, `find_best_value` from Task 1.
- Produces: `settings.totals_in_best_bet: bool` (default False); `MLBValueCalculator.find_best_bet(values: list[MLBValueResult], include_totals: bool = False) -> MLBValueResult | None`. Task 3 uses both.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_mlb_value_calculator.py`:

```python
class TestFindBestBet:
    def _candidates(self):
        rl = calc(market_type="runline", model_prob=0.62, market_prob=0.50)    # edge_pct 24
        ml = calc(market_type="moneyline", model_prob=0.61, market_prob=0.50)  # edge_pct 22
        total = calc(market_type="total", model_prob=0.70, market_prob=0.50)   # edge_pct 40 (highest)
        return rl, ml, total

    def test_totals_excluded_by_default(self):
        rl, ml, total = self._candidates()
        best = MLBValueCalculator.find_best_bet([ml, rl, total])
        assert best is rl  # highest non-total sort_score

    def test_totals_included_when_flagged(self):
        rl, ml, total = self._candidates()
        best = MLBValueCalculator.find_best_bet([ml, rl, total], include_totals=True)
        assert best is total

    def test_returns_none_when_only_totals_qualify(self):
        total = calc(market_type="total", model_prob=0.70, market_prob=0.50)
        assert MLBValueCalculator.find_best_bet([total]) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_mlb_value_calculator.py::TestFindBestBet -v`
Expected: FAIL — `AttributeError: find_best_bet`.

- [ ] **Step 3: Implement**

3a. In `src/services/mlb/value_calculator.py`, add after `find_best_value`:

```python
    @classmethod
    def find_best_bet(
        cls,
        values: list[MLBValueResult],
        include_totals: bool = False,
    ) -> MLBValueResult | None:
        """
        Find the overall best bet across markets.

        Totals are excluded unless include_totals — they ran -29u / 48.7%
        as best bets over Apr-Jul 2026. best_total is still tracked
        separately as the shadow record for the retrained model
        (re-entry gate: >=100 graded picks, >=53% WR, positive units).
        """
        if not include_totals:
            values = [v for v in values if v.market_type != "total"]
        return cls.find_best_value(values)
```

3b. In `src/config.py`, after the `suppress_totals` setting (~line 73):

```python
    # Allow totals (over/under) to be chosen as the overall best_bet.
    # Independent of suppress_totals (which stops totals being scored at all).
    # Totals as best bets ran -29u / 48.7% Apr-Jul 2026. Flip to True only via
    # the re-entry gate: >=100 graded best_total picks under the retrained
    # model with >=53% WR and positive cumulative units.
    totals_in_best_bet: bool = False
```

3c. In `src/services/mlb/scorer.py` line 488, replace:

```python
        prediction.best_bet = MLBValueCalculator.find_best_value(all_values)
```

with:

```python
        prediction.best_bet = MLBValueCalculator.find_best_bet(
            all_values, include_totals=settings.totals_in_best_bet
        )
```

(`settings` is already imported in scorer.py line 14.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/test_mlb_value_calculator.py -v`
Expected: all PASS (including Task 1 tests).

- [ ] **Step 5: Commit**

```bash
git add src/config.py src/services/mlb/value_calculator.py src/services/mlb/scorer.py tests/unit/test_mlb_value_calculator.py
git commit -m "tune(mlb): exclude totals from best_bet behind totals_in_best_bet flag

best_total is still scored, recorded, and graded — that shadow record is
the Phase 2 re-entry gate for the retrained totals model.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Scheduler defense-in-depth — never snapshot a total as best_bet

**Files:**
- Modify: `src/tasks/mlb_scheduler.py` (add module-level helper near the top of the file after imports; use it in `snapshot_predictions_async` best_bet block, ~line 376)
- Test: `tests/unit/test_mlb_scheduler_best_bet.py` (create)

**Interfaces:**
- Consumes: `settings.totals_in_best_bet`, `MLBValueResult` (with `sort_score`, `is_value_bet`).
- Produces: `resolve_best_bet(best_bet, best_ml, best_rl, totals_allowed: bool) -> MLBValueResult | None` — pure function, importable from `src.tasks.mlb_scheduler`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_mlb_scheduler_best_bet.py`:

```python
"""Defense-in-depth: a total must never be written to snapshot best_bet_*
fields while totals_in_best_bet is off, even if the scorer produced one
(e.g. stale prediction object or env drift)."""

from src.services.mlb.value_calculator import MLBValueCalculator
from src.tasks.mlb_scheduler import resolve_best_bet


def calc(market_type, model_prob, market_prob=0.50):
    bet_type = {"runline": "home_rl", "moneyline": "home_ml", "total": "over"}[market_type]
    return MLBValueCalculator.calculate_value(
        market_type=market_type,
        bet_type=bet_type,
        model_prob=model_prob,
        market_prob=market_prob,
        odds_decimal=2.0,
        team="NYY" if market_type != "total" else None,
        line=1.5 if market_type == "runline" else None,
        model_confidence=0.5,
    )


def test_non_total_passes_through():
    rl = calc("runline", 0.62)
    assert resolve_best_bet(rl, None, rl, totals_allowed=False) is rl


def test_total_replaced_by_best_of_ml_rl():
    total = calc("total", 0.70)
    ml = calc("moneyline", 0.61)
    rl = calc("runline", 0.62)
    assert resolve_best_bet(total, ml, rl, totals_allowed=False) is rl


def test_total_kept_when_allowed():
    total = calc("total", 0.70)
    ml = calc("moneyline", 0.61)
    assert resolve_best_bet(total, ml, None, totals_allowed=True) is total


def test_total_with_no_qualifying_fallback_returns_none():
    total = calc("total", 0.70)
    weak_ml = calc("moneyline", 0.55)  # raw_edge 0.05 < MIN_EDGE
    assert resolve_best_bet(total, weak_ml, None, totals_allowed=False) is None


def test_none_passes_through():
    assert resolve_best_bet(None, None, None, totals_allowed=False) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/test_mlb_scheduler_best_bet.py -v`
Expected: FAIL — `ImportError: cannot import name 'resolve_best_bet'`.

- [ ] **Step 3: Implement**

3a. In `src/tasks/mlb_scheduler.py`, add after the imports (module level, before the first task function). Also add `from src.services.mlb.value_calculator import MLBValueResult` to the imports if not already imported:

```python
def resolve_best_bet(
    best_bet: "MLBValueResult | None",
    best_ml: "MLBValueResult | None",
    best_rl: "MLBValueResult | None",
    totals_allowed: bool,
) -> "MLBValueResult | None":
    """Defense-in-depth for snapshots: if the scorer handed us a total as
    best_bet while totals_in_best_bet is off, fall back to the better of
    ML/RL so a total never lands in best_bet_* fields."""
    if best_bet is None or totals_allowed or best_bet.market_type != "total":
        return best_bet
    candidates = [v for v in (best_ml, best_rl) if v is not None and v.is_value_bet]
    if not candidates:
        return None
    return max(candidates, key=lambda v: v.sort_score)
```

3b. In `snapshot_predictions_async`, replace the best_bet block (~line 376):

```python
                if prediction.best_bet:
                    snapshot.best_bet_type = prediction.best_bet.market_type
                    ...
```

with:

```python
                best_bet = resolve_best_bet(
                    prediction.best_bet,
                    prediction.best_ml,
                    prediction.best_rl,
                    totals_allowed=settings.totals_in_best_bet,
                )
                if best_bet:
                    snapshot.best_bet_type = best_bet.market_type
                    snapshot.best_bet_team = best_bet.team
                    snapshot.best_bet_line = best_bet.line
                    snapshot.best_bet_odds = best_bet.odds_decimal
                    snapshot.best_bet_value_score = int(best_bet.value_score)
                    snapshot.best_bet_edge = best_bet.raw_edge
```

(`settings` is already imported at line 32.)

- [ ] **Step 4: Run all unit tests**

Run: `python3 -m pytest tests/unit/ -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tasks/mlb_scheduler.py tests/unit/test_mlb_scheduler_best_bet.py
git commit -m "tune(mlb): snapshot-level guard against totals in best_bet

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Offline validation sim — DEPLOY GATE

**Files:**
- Create: `<scratchpad>/validate_retune.py` (scratchpad dir from the session env — NOT committed)

**Interfaces:**
- Consumes: the actual retuned code (`MLBValueCalculator` from Tasks 1-2), prod DB via `DB_URL` from `src.tasks.prediction_tracker`.
- Produces: pass/fail verdict. **If any assertion fails, STOP — do not proceed to Task 5. Report the numbers.**

- [ ] **Step 1: Write the sim script**

```python
"""Replay graded season snapshots through the retuned calculator.

Gate asserts (spec):
  (a) no totals in simulated best_bet
  (b) zero simulated best_bet picks score exactly 100
  (c) simulated P&L >= +150u over the season window
"""

import sys

sys.path.insert(0, "/Applications/XAMPP/xamppfiles/htdocs/Sites/Truline/backend")

import psycopg2

from src.services.mlb.value_calculator import MLBValueCalculator
from src.tasks.prediction_tracker import DB_URL


def candidate(market_type, odds, edge, direction=None):
    if odds is None or edge is None:
        return None
    market_prob = 1.0 / float(odds)
    bet_type = {"runline": "home_rl", "moneyline": "home_ml", "total": direction or "over"}[market_type]
    return MLBValueCalculator.calculate_value(
        market_type=market_type,
        bet_type=bet_type,
        model_prob=market_prob + float(edge),
        market_prob=market_prob,
        odds_decimal=float(odds),
        team="X" if market_type != "total" else None,
        line=1.5 if market_type == "runline" else None,
        model_confidence=0.5,
    )


conn = psycopg2.connect(DB_URL)
cur = conn.cursor()
cur.execute("""
    SELECT best_ml_odds, best_ml_edge, best_ml_result, best_ml_profit,
           best_rl_odds, best_rl_edge, best_rl_result, best_rl_profit,
           best_total_odds, best_total_edge, best_total_result, best_total_profit,
           best_total_direction
    FROM mlb_prediction_snapshots
    WHERE best_bet_result IS NOT NULL
""")

picks, totals_chosen, saturated = [], 0, 0
for (mlo, mle, mlr, mlp, rlo, rle, rlr, rlp, to, te, tr, tp, tdir) in cur.fetchall():
    cands = []
    if mlr in ("win", "loss"):
        c = candidate("moneyline", mlo, mle)
        if c:
            cands.append((c, float(mlp)))
    if rlr in ("win", "loss"):
        c = candidate("runline", rlo, rle)
        if c:
            cands.append((c, float(rlp)))
    if tr in ("win", "loss"):
        c = candidate("total", to, te, tdir)
        if c:
            cands.append((c, float(tp)))
    best = MLBValueCalculator.find_best_bet([c for c, _ in cands])
    if best is None:
        continue
    profit = next(p for c, p in cands if c is best)
    picks.append(profit)
    totals_chosen += best.market_type == "total"
    saturated += best.value_score >= 100

n = len(picks)
pnl = sum(picks) / 100
wr = 100 * sum(1 for p in picks if p > 0) / n
print(f"picks={n} wr={wr:.1f}% pnl={pnl:+.1f}u totals={totals_chosen} saturated={saturated}")

assert totals_chosen == 0, f"GATE FAIL: {totals_chosen} totals chosen as best_bet"
assert saturated == 0, f"GATE FAIL: {saturated} picks at value_score 100"
assert pnl >= 150, f"GATE FAIL: simulated P&L {pnl:+.1f}u < +150u"
print("GATE PASS")
```

- [ ] **Step 2: Run it**

Run: `cd /Applications/XAMPP/xamppfiles/htdocs/Sites/Truline/backend && python3 <scratchpad>/validate_retune.py`
Expected: `GATE PASS` with pnl ≈ +155u ± 10 (planning-phase clean sim was +157.8u; small drift from the confidence multiplier and favorite bonus is fine). Caveat: `model_confidence=0.5` is an approximation — prod used real confidences.

- [ ] **Step 3: If GATE FAIL — stop and report**

Do not deploy. Post the printed numbers and stop the plan. If GATE PASS, continue.

---

### Task 5: Deploy Phase 1 and verify in prod

**Files:** none new — push and observe.

- [ ] **Step 1: Full test suite + push**

```bash
python3 -m pytest tests/ -v
git log --oneline main -5   # confirm only the retune commits + spec/plan docs are new
git push origin main
```
Expected: tests pass; push triggers Railway build.

- [ ] **Step 2: Verify service health after deploy (~3-5 min)**

```bash
curl -s https://nba-value-production.up.railway.app/health
```
Expected: healthy JSON response. If Railway CLI is authed (`railway login` may be needed — ask the user to run `! railway login` if not), also check `railway logs | tail -50` for startup errors.

- [ ] **Step 3: Verify next pick slate (after the next snapshot run, i.e. within ~1h of games going live, or next morning)**

```python
# via python3 with psycopg2, DB_URL from src.tasks.prediction_tracker
SELECT game_date, best_bet_type, best_bet_value_score, best_bet_edge
FROM mlb_prediction_snapshots
WHERE created_at > NOW() - INTERVAL '1 day'
ORDER BY created_at DESC LIMIT 20;
```
Expected: zero rows with `best_bet_type = 'total'`; `best_bet_value_score` values spread below 100 (roughly 45-97), not piled at 100.

- [ ] **Step 4: Report Phase 1 done to the user with the verification numbers.**

---

### Task 6: Configurable totals model path with v1 fallback

**Files:**
- Modify: `src/config.py` (Model Flags section)
- Modify: `src/services/mlb/scorer.py:82,127-137` (model loading)
- Test: `tests/unit/test_mlb_scorer_totals_model.py` (create)

**Interfaces:**
- Consumes: nothing from other tasks (independent of Tasks 1-3).
- Produces: `settings.mlb_totals_model_path: str` (default `"models/mlb_totals_v1.joblib"`); `MLBScorer` loads the configured path and falls back to `DEFAULT_TOTALS_MODEL` (v1) if the configured file is missing. Task 9 flips the env var `MLB_TOTALS_MODEL_PATH` to v2.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_mlb_scorer_totals_model.py`:

```python
"""Totals model path is configurable with fallback to v1 then heuristic."""

from src.services.mlb.scorer import MLBScorer


def test_explicit_path_loads():
    scorer = MLBScorer(None, totals_model_path="models/mlb_totals_v1.joblib")
    assert scorer.totals_model is not None


def test_missing_path_falls_back_to_v1():
    scorer = MLBScorer(None, totals_model_path="models/mlb_totals_v99_missing.joblib")
    assert scorer.totals_model is not None  # fell back to v1


def test_default_uses_configured_setting():
    from src.config import settings
    scorer = MLBScorer(None)
    # default setting points at v1, so the model loads
    assert settings.mlb_totals_model_path == "models/mlb_totals_v1.joblib"
    assert scorer.totals_model is not None
```

Note: `MLBScorer.__init__` only stores the session and loads model files, so `session=None` is safe for these tests. If `__init__` turns out to touch the session, wrap it in a minimal stub object instead — do not skip the tests.

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/test_mlb_scorer_totals_model.py -v`
Expected: `test_missing_path_falls_back_to_v1` FAILS (totals_model is None today when the path is missing); `test_default_uses_configured_setting` FAILS (`AttributeError: mlb_totals_model_path`).

- [ ] **Step 3: Implement**

3a. `src/config.py`, next to the other model flags:

```python
    # Path to the MLB totals model. Point at mlb_totals_v2.joblib once the
    # retrained model passes holdout eval (see retrain_mlb_totals task).
    # Missing file -> scorer falls back to v1, then to the heuristic.
    mlb_totals_model_path: str = "models/mlb_totals_v1.joblib"
```

3b. `src/services/mlb/scorer.py` — in `__init__`, replace:

```python
        totals_path = Path(totals_model_path or self.DEFAULT_TOTALS_MODEL)
```

with:

```python
        totals_path = Path(totals_model_path or settings.mlb_totals_model_path)
        if not totals_path.exists() and totals_path != Path(self.DEFAULT_TOTALS_MODEL):
            logger.warning(
                "Configured totals model missing, falling back to v1",
                configured=str(totals_path),
                fallback=self.DEFAULT_TOTALS_MODEL,
            )
            totals_path = Path(self.DEFAULT_TOTALS_MODEL)
```

(The existing `if totals_path.exists(): ... joblib.load ...` block below stays — if even v1 is missing, `totals_model` stays None and the heuristic is used, as today.)

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/unit/ -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/config.py src/services/mlb/scorer.py tests/unit/test_mlb_scorer_totals_model.py
git commit -m "feat(mlb): configurable totals model path with v1 fallback

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: Trainer — game_date column + explicit holdout

**Files:**
- Modify: `src/services/mlb/model_training.py` (`collect_training_data` ~line 322, `train_run_diff_model` ~line 348, `train_totals_model` ~line 410)
- Test: `tests/unit/test_mlb_model_training.py` (create)

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: training DataFrames carry a `game_date` string column (ISO, from MLB API `gameDate`); `train_totals_model(df, test_df=None)` — when `test_df` is given, train on all of `df` and evaluate on `test_df` (no internal split). Task 8 relies on both.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_mlb_model_training.py`:

```python
"""Trainer accepts an explicit holdout and excludes game_date from features."""

import numpy as np
import pandas as pd
import pytest

from src.services.mlb.model_training import HAS_LIGHTGBM, MLBModelTrainer

pytestmark = pytest.mark.skipif(not HAS_LIGHTGBM, reason="lightgbm not installed")


def synthetic_df(n, seed):
    rng = np.random.default_rng(seed)
    f1 = rng.normal(4.5, 1.0, n)
    f2 = rng.normal(4.5, 1.0, n)
    return pd.DataFrame({
        "home_runs_per_game": f1,
        "away_runs_per_game": f2,
        "total_runs": (f1 + f2 + rng.normal(0, 1.5, n)).round(),
        "run_diff": rng.integers(-5, 6, n),
        "home_win": rng.integers(0, 2, n),
        "season": 2026,
        "game_id": np.arange(n),
        "game_date": pd.date_range("2026-04-01", periods=n).strftime("%Y-%m-%d"),
    })


def test_explicit_holdout_used_for_eval():
    trainer = MLBModelTrainer(model_dir="/tmp")
    train_df = synthetic_df(400, seed=1)
    test_df = synthetic_df(100, seed=2)
    model, feature_cols, metrics = trainer.train_totals_model(train_df, test_df=test_df)
    assert "game_date" not in feature_cols
    assert metrics["mae"] > 0


def test_internal_split_still_works():
    trainer = MLBModelTrainer(model_dir="/tmp")
    model, feature_cols, metrics = trainer.train_totals_model(synthetic_df(500, seed=3))
    assert "game_date" not in feature_cols
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/test_mlb_model_training.py -v`
Expected: FAIL — `train_totals_model() got an unexpected keyword argument 'test_df'` and/or `game_date` leaking into feature_cols.

- [ ] **Step 3: Implement**

3a. In `collect_training_data`, next to the existing `features["season"] = season` line:

```python
                    features["season"] = season
                    features["game_id"] = game.get("gamePk")
                    features["game_date"] = (game.get("gameDate") or "")[:10]
```

3b. In BOTH `train_run_diff_model` and `train_totals_model`, add `"game_date"` to `exclude_cols`:

```python
        exclude_cols = ["run_diff", "total_runs", "home_win", "season", "game_id", "game_date"]
```

3c. Change `train_totals_model` signature and split block:

```python
    def train_totals_model(self, df: pd.DataFrame, test_df: pd.DataFrame | None = None) -> tuple:
        """Train the total runs prediction model.

        Args:
            df: Training data.
            test_df: Optional explicit holdout. When given, train on all of
                df and evaluate on test_df (time-based holdout). Otherwise
                fall back to the legacy positional 80/20 split.
        """
```

and replace the split lines:

```python
        X = df[feature_cols].fillna(0)
        y = df["total_runs"]

        if test_df is not None:
            X_train, y_train = X, y
            X_test = test_df[feature_cols].fillna(0)
            y_test = test_df["total_runs"]
        else:
            split_idx = int(len(df) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
```

(The rest of the method — lgb.Dataset, params, train, evaluate — is unchanged and works off `X_train/X_test/y_train/y_test`.)

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/unit/ -v`
Expected: all PASS (model_training tests skip if lightgbm missing — if they skip, run `pip3 install lightgbm` and re-run).

- [ ] **Step 5: Commit**

```bash
git add src/services/mlb/model_training.py tests/unit/test_mlb_model_training.py
git commit -m "feat(mlb): game_date in training data + explicit holdout for totals trainer

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: Retrain script — retrain_mlb_totals.py

**Files:**
- Create: `src/tasks/retrain_mlb_totals.py`

**Interfaces:**
- Consumes: `MLBModelTrainer` with `test_df` support (Task 7), `DB_URL` from `src.tasks.prediction_tracker`.
- Produces: `models/mlb_totals_v2.joblib` in the same dict format as v1 (`{"model", "feature_cols", "metrics"}` — matches what `MLBScorer` loads). Exit code 1 if v2 does not beat v1.

- [ ] **Step 1: Write the script**

Create `src/tasks/retrain_mlb_totals.py`:

```python
"""Retrain the MLB totals model on 2024-2026 data with a time-based holdout.

The current mlb_totals_v1.joblib was trained 2026-02-09 and has never seen
the 2026 season. Gate: v2 must beat v1 on holdout MAE (Jun 1 - Jul 5 2026)
or it is not saved. Over/under hit rate vs recorded snapshot lines is
reported as informational (only ~177 holdout games have a recorded line).

Usage:
    python3 -m src.tasks.retrain_mlb_totals              # eval, then save v2 if it wins
    python3 -m src.tasks.retrain_mlb_totals --eval-only  # eval only, never save
"""

import asyncio
import sys
from pathlib import Path

import joblib
import psycopg2
from sklearn.metrics import mean_absolute_error

from src.services.mlb.model_training import MLBModelTrainer
from src.tasks.prediction_tracker import DB_URL

SEASONS = [2024, 2025, 2026]
HOLDOUT_START = "2026-06-01"
V1_PATH = Path("models/mlb_totals_v1.joblib")
V2_PATH = Path("models/mlb_totals_v2.joblib")


def hit_rate(pred_by_game_id: dict) -> str:
    """Over/under accuracy vs snapshot best_total_line where recorded."""
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    cur.execute(
        """SELECT game_id, best_total_line, home_score + away_score
           FROM mlb_prediction_snapshots
           WHERE game_date >= %s AND home_score IS NOT NULL
             AND best_total_line IS NOT NULL""",
        (HOLDOUT_START,),
    )
    hits = misses = 0
    for game_id, line, actual_total in cur.fetchall():
        pred = pred_by_game_id.get(str(game_id))
        if pred is None or float(actual_total) == float(line):  # unmatched or push
            continue
        predicted_over = pred > float(line)
        actual_over = float(actual_total) > float(line)
        hits += predicted_over == actual_over
        misses += predicted_over != actual_over
    conn.close()
    n = hits + misses
    return f"{hits}/{n} = {100 * hits / n:.1f}%" if n else "no matchable games"


async def main() -> int:
    eval_only = "--eval-only" in sys.argv
    trainer = MLBModelTrainer()

    df = await trainer.collect_training_data(SEASONS)
    df = df[df["game_date"] != ""].sort_values("game_date").reset_index(drop=True)
    train_df = df[df["game_date"] < HOLDOUT_START]
    holdout_df = df[df["game_date"] >= HOLDOUT_START]
    print(f"train={len(train_df)} holdout={len(holdout_df)} (holdout from {HOLDOUT_START})")

    # v2 candidate: train pre-holdout, evaluate on holdout
    model, feature_cols, metrics = trainer.train_totals_model(train_df, test_df=holdout_df)
    v2_mae = metrics["mae"]

    # v1 baseline on the same holdout
    v1 = joblib.load(V1_PATH)
    X_hold = holdout_df.reindex(columns=v1["feature_cols"]).fillna(0)
    v1_pred = v1["model"].predict(X_hold)
    v1_mae = mean_absolute_error(holdout_df["total_runs"], v1_pred)

    v2_pred = model.predict(holdout_df.reindex(columns=feature_cols).fillna(0))
    game_ids = holdout_df["game_id"].astype(str).tolist()
    print(f"v1 holdout MAE={v1_mae:.3f}  hit-rate {hit_rate(dict(zip(game_ids, v1_pred)))}")
    print(f"v2 holdout MAE={v2_mae:.3f}  hit-rate {hit_rate(dict(zip(game_ids, v2_pred)))}")

    if v2_mae >= v1_mae:
        print("GATE FAIL: v2 does not beat v1 on holdout MAE — not saving.")
        return 1
    if eval_only:
        print("GATE PASS (eval-only, not saving).")
        return 0

    # Final model: retrain on ALL data (legacy internal split for early stopping)
    final_model, final_cols, final_metrics = trainer.train_totals_model(df)
    joblib.dump(
        {"model": final_model, "feature_cols": final_cols, "metrics": final_metrics},
        V2_PATH,
    )
    print(f"GATE PASS: saved {V2_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
```

- [ ] **Step 2: Syntax check + dry import**

Run: `python3 -c "import src.tasks.retrain_mlb_totals"`
Expected: no output (imports cleanly).

- [ ] **Step 3: Commit**

```bash
git add src/tasks/retrain_mlb_totals.py
git commit -m "feat(mlb): totals retrain script with time-based holdout vs v1 gate

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 9: Run the retrain, shadow-deploy v2, document the re-entry gate

**Files:**
- Possibly create: `models/mlb_totals_v2.joblib` (committed only if the gate passes — check `git ls-files models/` first to confirm model files are tracked; v1 is)

- [ ] **Step 1: Run the retrain (long-running — MLB API fetch for 3 seasons, expect 20-60 min due to rate limiting)**

Run in background: `python3 -m src.tasks.retrain_mlb_totals`
Expected output: train/holdout counts, v1 vs v2 MAE + hit rates, `GATE PASS: saved models/mlb_totals_v2.joblib` — or `GATE FAIL`.

- [ ] **Step 2: If GATE FAIL — stop Phase 2, report numbers**

Totals stay on v1 and stay out of best_bet. Report the MAE comparison to the user; the retrain can be revisited with better features later. Phase 1 remains fully deployed. Skip remaining steps.

- [ ] **Step 3: If GATE PASS — deploy v2 in shadow mode**

```bash
git ls-files models/ | head          # confirm model files are git-tracked
git add models/mlb_totals_v2.joblib
git commit -m "feat(mlb): totals model v2 trained through 2026-07 (shadow mode)

Holdout (Jun 1 - Jul 5): v2 MAE <fill from output> vs v1 MAE <fill>.
Serving best_total only; excluded from best_bet until the re-entry gate
passes (>=100 graded picks, >=53% WR, positive units).

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
git push origin main
```

Then flip the model path on Railway — either `railway variables --set MLB_TOTALS_MODEL_PATH=models/mlb_totals_v2.joblib` (if CLI is authed) or ask the user to set it in the Railway dashboard. Railway redeploys on variable change; verify with the health endpoint and the "Loaded totals model" log line.

- [ ] **Step 4: Document the re-entry gate check**

Add to the spec's Phase 2 section (edit `docs/superpowers/specs/2026-07-06-mlb-value-retune-design.md`, append under "Re-entry gate"):

```markdown
Check (run every ~2 weeks after v2 shadow deploy of <deploy date>):

    SELECT count(*) AS picks,
           count(*) FILTER (WHERE best_total_result = 'win') AS wins,
           round(100.0 * count(*) FILTER (WHERE best_total_result = 'win')
                 / nullif(count(*) FILTER (WHERE best_total_result IN ('win','loss')), 0), 1) AS wr,
           round(sum(best_total_profit)::numeric / 100, 1) AS units
    FROM mlb_prediction_snapshots
    WHERE game_date >= '<v2 deploy date>' AND best_total_result IS NOT NULL;

Gate: picks >= 100 AND wr >= 53 AND units > 0 → set TOTALS_IN_BEST_BET=true
on Railway. Otherwise totals stay out.
```

Commit the spec edit:

```bash
git add docs/superpowers/specs/2026-07-06-mlb-value-retune-design.md
git commit -m "docs: record totals re-entry gate check query

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
git push origin main
```

- [ ] **Step 5: Final report to user**

Summarize: Phase 1 deployed + verified numbers, v2 holdout metrics, shadow-mode status, and the date the re-entry gate can first be evaluated (~2-3 weeks of graded picks).
