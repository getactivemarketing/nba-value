# Runline Sign-Pairing Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the MLB scorer so each runline side is paired with its own cover probability, real price, and signed line — killing the phantom edge — then re-measure the true runline edge from history.

**Architecture:** Extract a pure, DB-free helper that turns one runline market row into the two side-`MLBValueResult`s with correct probability↔odds↔signed-line pairing; the async `score_game` loop calls it per row and picks the best. A separate read-only script re-derives the corrected pick per historical game from frozen `predicted_run_diff` + historical `mlb_markets` prices and grades it, producing the corrected edge report. History is not mutated.

**Tech Stack:** Python 3.11, SQLAlchemy async, psycopg2 (scripts), pytest. Run tests from `backend/`.

**Spec:** `docs/superpowers/specs/2026-07-21-runline-signfix-design.md`

## Global Constraints

- Work in worktree `/private/tmp/claude-501/-Applications-XAMPP-xamppfiles-htdocs-Sites/8910339d-4857-421c-bc4e-8d32294c9d28/scratchpad/tl-fix` on branch `runline-full-fix` (off origin/main `57c45d9`). Do NOT touch the primary checkout at `Sites/Truline`.
- `runline_in_best_bet` stays `False` (paused) — this change does NOT un-pause runline.
- Cover-prob formulas (verbatim): `p_home_minus = cover(rd, 1.5)`, `p_away_minus = cover(-rd, 1.5)`, `p_home_plus = 1 - p_away_minus`, `p_away_plus = 1 - p_home_minus`, where `rd = predicted_run_diff` (positive = home favored) and `cover(run_diff, spread) = clamp(logistic(0.5*(run_diff - spread)), 0.05, 0.95)`.
- Row→side mapping (verbatim): `line == -1.5` → home side (home −1.5, `p_home_minus`, `home_odds`, signed line **−1.5**) + away side (away +1.5, `p_away_plus`, `away_odds`, signed line **+1.5**). `line == +1.5` → home side (home +1.5, `p_home_plus`, `home_odds`, signed line **+1.5**) + away side (away −1.5, `p_away_minus`, `away_odds`, signed line **−1.5**).
- Only process standard runline rows: `abs(line) == 1.5`. Skip alt lines.
- Drop the `MAX_RUNLINE_ODDS` odds filter from the scorer.
- Signed line flows into `MLBValueResult.line` unchanged; grading and formatters are already sign-aware — do NOT change them.
- Commit messages end with: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
- Test command: `cd backend && python3 -m pytest tests/unit/<file> -v`.

---

### Task 1: Pure runline 4-side helper

**Files:**
- Modify: `backend/src/services/mlb/scorer.py` (make `_run_diff_to_cover_prob` a `@staticmethod`; add `_runline_side_values` static helper)
- Test: `backend/tests/unit/test_runline_sidevalues.py` (create)

**Interfaces:**
- Consumes: `MLBValueCalculator.calculate_value(market_type, bet_type, model_prob, market_prob, odds_decimal, team, line)` and `MLBValueCalculator.devig_odds(home_odds, away_odds) -> (home_prob, away_prob)`.
- Produces: `MLBScorer._runline_side_values(predicted_run_diff: float, home_team: str, away_team: str, line: float, home_odds: float, away_odds: float) -> list[MLBValueResult]` — the two side-values for one standard (`abs(line)==1.5`) runline row, each with correct model_prob, market_prob (devigged), odds_decimal, team, and **signed** line. And `MLBScorer._run_diff_to_cover_prob` becomes a staticmethod (same body).

- [ ] **Step 1: Write the failing tests**

`backend/tests/unit/test_runline_sidevalues.py`:

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `cd backend && python3 -m pytest tests/unit/test_runline_sidevalues.py -v`
Expected: FAIL — `AttributeError: ... _runline_side_values` (and `_run_diff_to_cover_prob` not callable as staticmethod).

- [ ] **Step 3: Implement**

In `backend/src/services/mlb/scorer.py`, change the `_run_diff_to_cover_prob` method to a staticmethod (remove `self`; body unchanged):

```python
    @staticmethod
    def _run_diff_to_cover_prob(run_diff: float, spread: float) -> float:
        """Probability the (home) team covers `spread` runs. Pure/static."""
        import math
        adjusted_diff = run_diff - spread
        k = 0.5
        p = 1 / (1 + math.exp(-k * adjusted_diff))
        return max(0.05, min(0.95, p))
```

Update the existing internal call `self._run_diff_to_cover_prob(predicted_run_diff, self.RUNLINE)` to `MLBScorer._run_diff_to_cover_prob(predicted_run_diff, self.RUNLINE)` (or leave `self.` — staticmethod is callable both ways; prefer `self.` to minimize diff). Keep it working.

Add the helper as a staticmethod on `MLBScorer` (place it directly below `_run_diff_to_cover_prob`):

```python
    @staticmethod
    def _runline_side_values(
        predicted_run_diff: float,
        home_team: str,
        away_team: str,
        line: float,
        home_odds: float,
        away_odds: float,
    ) -> list["MLBValueResult"]:
        """Two side-values for ONE standard runline row (abs(line)==1.5), each
        paired with its own cover probability, real price, and SIGNED line."""
        cover = MLBScorer._run_diff_to_cover_prob
        p_home_minus = cover(predicted_run_diff, 1.5)    # home -1.5
        p_away_minus = cover(-predicted_run_diff, 1.5)   # away -1.5
        p_home_plus = 1 - p_away_minus                    # home +1.5
        p_away_plus = 1 - p_home_minus                    # away +1.5

        home_prob, away_prob = MLBValueCalculator.devig_odds(home_odds, away_odds)

        if line == -1.5:
            sides = [
                (p_home_minus, home_odds, -1.5, home_team, "home_rl", home_prob),
                (p_away_plus, away_odds, 1.5, away_team, "away_rl", away_prob),
            ]
        else:  # line == 1.5
            sides = [
                (p_home_plus, home_odds, 1.5, home_team, "home_rl", home_prob),
                (p_away_minus, away_odds, -1.5, away_team, "away_rl", away_prob),
            ]

        return [
            MLBValueCalculator.calculate_value(
                market_type="runline", bet_type=bt, model_prob=prob,
                market_prob=mprob, odds_decimal=odds, team=team, line=sline,
            )
            for prob, odds, sline, team, bt, mprob in sides
        ]
```

Ensure `MLBValueCalculator` is imported at module top (it already is — it's used elsewhere in scorer.py).

- [ ] **Step 4: Run to verify pass**

Run: `cd backend && python3 -m pytest tests/unit/test_runline_sidevalues.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/src/services/mlb/scorer.py backend/tests/unit/test_runline_sidevalues.py
git commit -m "feat(mlb): pure runline side-value helper with correct sign pairing

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Wire the helper into score_game; drop MAX_RUNLINE_ODDS

**Files:**
- Modify: `backend/src/services/mlb/scorer.py` (the `elif market.market_type == "runline":` block in `score_game`)
- Test: `backend/tests/unit/test_runline_scorer_integration.py` (create)

**Interfaces:**
- Consumes: `MLBScorer._runline_side_values(...)` from Task 1; `MLBValueCalculator.find_best_value(values)`.
- Produces: `score_game` sets `prediction.best_rl` to the best correctly-paired runline side; `all_values` includes every processed side. No `MAX_RUNLINE_ODDS` reference remains in the runline block.

- [ ] **Step 1: Write the failing test**

`backend/tests/unit/test_runline_scorer_integration.py`:

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `cd backend && python3 -m pytest tests/unit/test_runline_scorer_integration.py -v`
Expected: FAIL — the test imports fine but currently `_runline_side_values` exists (Task 1), so these should actually PASS already for the helper-level asserts. If they pass, that's fine — this test locks selection behavior. Proceed to Step 3 to change `score_game` itself. (If any assert fails, it reveals a real pairing error to fix before touching score_game.)

- [ ] **Step 3: Rewire the score_game runline block**

Replace the entire `elif market.market_type == "runline":` block (the one containing `home_odds`, `away_odds`, `MLBValueCalculator.devig_odds`, the two `if ... <= MLBValueCalculator.MAX_RUNLINE_ODDS` guards, and `prediction.best_rl = MLBValueCalculator.find_best_value(rl_values)`) with:

```python
            elif market.market_type == "runline":
                # Standard runline only; each side paired with its own cover
                # probability, real price, and SIGNED line (2026-07-21 fix).
                if (
                    market.home_odds and market.away_odds and market.line
                    and abs(float(market.line)) == 1.5
                ):
                    side_values = MLBScorer._runline_side_values(
                        predicted_run_diff,
                        game.home_team,
                        game.away_team,
                        float(market.line),
                        float(market.home_odds),
                        float(market.away_odds),
                    )
                    all_values.extend(side_values)
                    rl_values.extend(side_values)
```

Notes for the implementer:
- `rl_values` must be initialized to `[]` before the market loop if it isn't already accumulating across rows. Check the current code: `rl_values: list[MLBValueResult] = []` was previously declared INSIDE the runline block (per-row). Move its declaration to just before the `for market in markets` loop (alongside where `all_values` is declared) so sides accumulate across all rows, and set `prediction.best_rl = MLBValueCalculator.find_best_value(rl_values)` ONCE after the loop (next to where `prediction.best_bet` / `best_ml` finalization happens). Verify `predicted_run_diff` is in scope in the loop (it is a local computed earlier in `score_game`).
- Remove the now-unused `MAX_RUNLINE_ODDS` references in this block. Leave the constant defined in `value_calculator.py` (harmless) unless it has no other references — if `grep -rn MAX_RUNLINE_ODDS backend/src` shows only the definition after this edit, delete the definition too.

- [ ] **Step 4: Run tests + full suite**

Run: `cd backend && python3 -m pytest tests/unit/test_runline_scorer_integration.py tests/unit/test_runline_sidevalues.py -v`
Expected: all pass.

Run: `cd backend && python3 -c "from src.services.mlb.scorer import MLBScorer; print('import ok')"` → `import ok`
Run: `cd backend && python3 -m pytest tests/unit -q`
Expected: all pass (existing suite + new tests). If any prior runline test asserted the old abs-line behavior, update it to the signed-line reality and note it.

- [ ] **Step 5: Commit**

```bash
git add backend/src/services/mlb/scorer.py backend/tests/unit/test_runline_scorer_integration.py
git commit -m "fix(mlb): rewire runline scoring to signed-side helper, drop odds cap

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Corrected-report / re-validation script

**Files:**
- Create: `backend/scripts/runline_revalidation.py`
- Test: `backend/tests/unit/test_runline_revalidation.py`

**Interfaces:**
- Consumes: `MLBScorer._runline_side_values`, `MLBValueCalculator.find_best_value` (Tasks 1–2).
- Produces: pure helper `grade_runline(bet_team: str, signed_line: float, home_team: str, home_score: int, away_score: int, odds_decimal: float) -> tuple[str, float]` returning `(result, profit_units)` where result ∈ {"win","loss","push"} and profit is at flat 1.0-unit stake (win → `odds_decimal - 1`, loss → `-1.0`, push → `0.0`); and a `main()` that prints the corrected runline report. Grading rule (matches production): `adjusted = bet_score + signed_line; win if adjusted > opp_score, push if ==, else loss`, where `bet_score`/`opp_score` are chosen by whether `bet_team == home_team`.

- [ ] **Step 1: Write the failing test**

`backend/tests/unit/test_runline_revalidation.py`:

```python
"""Corrected runline grading helper (flat 1u): sign-aware, matches prod rule."""

import pytest
from scripts.runline_revalidation import grade_runline


def test_plus15_covers_when_team_loses_by_one():
    # away DET +1.5; DET loses by 1 (LAA 3, DET 2) -> +1.5 covers (win)
    res, profit = grade_runline("DET", 1.5, home_team="LAA", home_score=3, away_score=2, odds_decimal=1.5)
    assert res == "win" and profit == pytest.approx(0.5)


def test_minus15_loses_when_favorite_wins_by_one():
    # DET -1.5 needs win by 2+. DET 2 - LAA 1 = win by 1 -> loss.
    res, profit = grade_runline("DET", -1.5, home_team="LAA", home_score=1, away_score=2, odds_decimal=2.5)
    assert res == "loss" and profit == pytest.approx(-1.0)


def test_minus15_wins_when_favorite_wins_by_two():
    res, profit = grade_runline("DET", -1.5, home_team="LAA", home_score=1, away_score=3, odds_decimal=2.5)
    assert res == "win" and profit == pytest.approx(1.5)


def test_home_team_plus15_perspective():
    # home LAA +1.5, LAA loses by 1 (LAA 2, DET 3) -> covers (win)
    res, profit = grade_runline("LAA", 1.5, home_team="LAA", home_score=2, away_score=3, odds_decimal=1.8)
    assert res == "win" and profit == pytest.approx(0.8)
```

- [ ] **Step 2: Run to verify failure**

Run: `cd backend && python3 -m pytest tests/unit/test_runline_revalidation.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.runline_revalidation'`.

- [ ] **Step 3: Implement the script**

`backend/scripts/runline_revalidation.py`:

```python
#!/usr/bin/env python3
"""Corrected runline re-validation (read-only).

Re-derives the correctly-paired runline pick for each graded MLB snapshot from
the frozen predicted_run_diff + historical mlb_markets prices (via the fixed
MLBScorer._runline_side_values), grades it against the actual margin, and reports
the real runline edge vs the tracker's inflated figure. Does NOT mutate the DB.

Run from backend/: python3 scripts/runline_revalidation.py
"""
import os
import re
import sys
from pathlib import Path

BASELINE = "2026-07-08"


def grade_runline(bet_team, signed_line, home_team, home_score, away_score, odds_decimal):
    if bet_team == home_team:
        bet_score, opp_score = home_score, away_score
    else:
        bet_score, opp_score = away_score, home_score
    adjusted = bet_score + signed_line
    if adjusted > opp_score:
        return "win", odds_decimal - 1.0
    if adjusted == opp_score:
        return "push", 0.0
    return "loss", -1.0


def _db_url():
    tracker = Path(__file__).resolve().parent.parent / "src" / "tasks" / "prediction_tracker.py"
    m = re.search(r"postgresql://[^\"'\s]+", tracker.read_text())
    if not m:
        sys.exit("prod DB URL not found")
    return m.group(0)


def corrected_pick(predicted_run_diff, market_rows):
    """market_rows: list of (line, home_odds, away_odds, home_team, away_team).
    Returns the best correctly-paired runline MLBValueResult or None."""
    from src.services.mlb.scorer import MLBScorer
    from src.services.mlb.value_calculator import MLBValueCalculator
    vals = []
    for line, ho, ao, h, a in market_rows:
        if line is None or abs(float(line)) != 1.5 or ho is None or ao is None:
            continue
        vals += MLBScorer._runline_side_values(
            float(predicted_run_diff), h, a, float(line), float(ho), float(ao)
        )
    return MLBValueCalculator.find_best_value(vals)


def main():
    import psycopg2
    conn = psycopg2.connect(os.environ.get("PGURL") or _db_url())
    cur = conn.cursor()
    cur.execute(
        """
        SELECT game_id, game_date, home_team, away_team, predicted_run_diff,
               home_score, away_score,
               best_bet_type, best_bet_team, best_bet_value_score
        FROM mlb_prediction_snapshots
        WHERE predicted_run_diff IS NOT NULL AND actual_winner IS NOT NULL
          AND home_score IS NOT NULL AND away_score IS NOT NULL
        ORDER BY game_date
        """
    )
    rows = cur.fetchall()
    if not rows:
        print("no graded snapshots found")
        return

    def market_rows(gid):
        cur.execute(
            """SELECT line, home_odds, away_odds FROM mlb_markets
               WHERE game_id=%s AND market_type='runline'
                 AND home_odds IS NOT NULL AND away_odds IS NOT NULL""",
            (gid,),
        )
        return cur.fetchall()

    windows = {"since_baseline": [], "all_history": []}
    dates = []
    for (gid, d, home, away, rd, hs, as_, bt, bteam, bscore) in rows:
        dates.append(d)
        mrows = [(l, ho, ao, home, away) for (l, ho, ao) in market_rows(gid)]
        pick = corrected_pick(rd, mrows)
        if pick is None:
            continue
        res, profit = grade_runline(pick.team, float(pick.line), home, hs, as_, float(pick.odds_decimal))
        rec = (res, profit)
        windows["all_history"].append(rec)
        if str(d) >= BASELINE:
            windows["since_baseline"].append(rec)

    def summarize(name, recs):
        n = len(recs)
        w = sum(1 for r, _ in recs if r == "win")
        l = sum(1 for r, _ in recs if r == "loss")
        u = sum(p for _, p in recs)
        wr = w / (w + l) * 100 if (w + l) else 0.0
        print(f"  {name}: {w}-{l} ({wr:.1f}% WR over {n}) -> {u:+.1f}u (flat)")

    print(f"=== corrected runline re-validation | snapshots {min(dates)}..{max(dates)} ===")
    print("(re-derived from frozen predicted_run_diff + historical odds; snapshots NOT modified)")
    summarize("since 2026-07-08 baseline", windows["since_baseline"])
    summarize("all snapshot history", windows["all_history"])
    if min(str(d) for d in dates) > "2026-04-05":
        print(f"  NOTE: snapshots start {min(dates)} — earlier season (Apr start) not covered; "
              f"full-season backtest number cannot be reproduced from snapshots.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify pass**

Run: `cd backend && python3 -m pytest tests/unit/test_runline_revalidation.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/scripts/runline_revalidation.py backend/tests/unit/test_runline_revalidation.py
git commit -m "feat(mlb): corrected runline re-validation script (read-only)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Run re-validation; deploy the fix

**Files:** none (operational).

- [ ] **Step 1: Full suite green**

Run: `cd backend && python3 -m pytest tests/unit -q`
Expected: all pass.

- [ ] **Step 2: Run the corrected report against prod (read-only)**

Run: `cd backend && python3 scripts/runline_revalidation.py`
Capture the corrected runline record (since baseline + all history) and the covered date range. This is the number that informs the un-pause decision — report it to the controller; do NOT flip `runline_in_best_bet` in this plan.

- [ ] **Step 3: Merge + deploy the scorer fix**

The scorer fix is safe to deploy even while runline stays paused (it only changes how `best_rl` is computed; `best_rl` is excluded from best_bet). From the primary worktree (`Sites/Truline`, on `main`):

```bash
git merge --no-ff runline-full-fix -m "Merge runline-full-fix: correct runline sign pairing + re-validation

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
git push origin main
```

- [ ] **Step 4: Verify deploy**

`railway status --json` commitHash matches the merge; after next scoring cycle, spot-check a game's `best_rl` in prod shows a signed line consistent with the market (a favorite's `best_rl` at `+1.5` carries minus-money odds; a `-1.5` carries plus-money). Confirm no runline reaches `best_bet` (still paused).
