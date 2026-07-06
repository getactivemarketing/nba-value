# MLB Value Algorithm Retune + Gated Totals Retrain

**Date:** 2026-07-06
**Status:** Approved
**Goal:** Maximize best-bet profitability by re-applying the data-supported parts of reverted commit `b004564`, fixing value-score saturation, and giving totals a monitored path back into `best_bet` via a retrained model.

## Background

Commit `b004564` (May 18) tuned the MLB value calculator; it was reverted May 22 (`fc59656`) for reasons nobody remembers. Season data through Jul 5 (889 graded best bets, flat $100 stakes):

| Bet type | Record | Win % | P&L |
|---|---|---|---|
| Run line | 411-344 | 54.4% | +127u |
| Moneyline | 251-281 | 47.2% | +29u |
| Totals | 203-214 | 48.7% | -29u |
| Best bet | 459-442 | 50.9% | +160u |

Findings that drive this design:
- The tuned window (May 18-22) performed fine (+12.2u, 54.2%); the revert was not performance-driven.
- Since the revert, best-bet win rate dropped 54.1% → 50.5% and totals crept back into `best_bet` (98 of 445 picks).
- 65% of picks since the revert score exactly 100 — the value score barely differentiates picks.
- The 10-15% raw-edge bucket, which `b004564` filtered out by raising MIN_EDGE to 0.12, has been the **best** bucket since the revert (56.9%, +17.9u). That change is not re-applied.
- 93% of RL picks are +1.5 underdogs at avg +110 winning 56.1% (breakeven 47.6%). Caveat: `best_rl_odds` is the best line at snapshot time and may not always be gettable.
- The MLB totals model (`models/mlb_totals_v1.joblib`) was trained Feb 9 — it has never seen 2026 data.

## Phase 1 — Value-layer retune

### `src/services/mlb/value_calculator.py`
1. **`MARKET_REGRESSION_WEIGHT = 0.50`** — blend `model_prob` 50/50 toward `market_prob` before computing the edge used for `value_score`. The raw (unblended) edge remains the MIN_EDGE / MAX_EDGE_PCT filter signal. Port the implementation from `b004564` (`git show b004564 -- backend/src/services/mlb/value_calculator.py`).
2. **`MAX_EDGE_PCT = 80.0`** — if raw `edge_pct` exceeds 80, the bet is marked `is_value_bet = False` (model blowup).
3. **Market multipliers** — moneyline 0.95 → 0.80; total 0.90 → 0.85.
4. **`MIN_EDGE` stays 0.10** — update its comment: as of 2026-07-06 the 10-15% edge bucket is the best performer since May 23 (56.9% WR, +17.9u over 72 bets); do not raise to 0.12.
5. **Cleanup** — `EDGE_SCALE_FACTOR = 400` is dead code; the formula hardcodes `edge_pct * 4.0`. Set the constant to 4.0 and use it in the formula.

### `src/config.py`
6. New setting **`totals_in_best_bet: bool = False`** (env-overridable like `suppress_totals`). Independent of `suppress_totals`, which controls whether totals are scored at all.

### `src/services/mlb/scorer.py`
7. Where `best_bet` is chosen (`find_best_value(all_values)`, ~line 488): when `totals_in_best_bet` is false, exclude total-market values from the `best_bet` candidate list. `best_total` is still computed, recorded, and graded (shadow track record for Phase 2).

### `src/tasks/mlb_scheduler.py`
8. Defense-in-depth (port from `b004564`): in `snapshot_predictions_async`, if the chosen best bet is a total while `totals_in_best_bet` is false, fall back to the better-scored of `best_ml` / `best_rl` before writing snapshot `best_bet_*` fields.

### Validation (before deploy)
Offline re-simulation script (scratchpad, not committed):
- Replay graded season snapshots through the new scoring. Reconstruct `market_prob = 1/odds` and `model_prob = market_prob + edge` from stored fields for each of `best_ml` / `best_rl` / `best_total`.
- Assert: (a) no totals in simulated best_bet; (b) share of picks scoring exactly 100 drops below 30% (currently 65%); (c) simulated P&L on the surviving pick set is not materially worse than actuals (target: ≥ +160u equivalent on the same window).
- If simulation shows degradation, stop and reassess before deploying.

### Non-goals / unaffected
- NRFI pipeline (separate content + grading path) — untouched.
- Social recap posts read `best_bet_*` from snapshots, so totals suppression flows through automatically — no social code changes.
- The uncommitted NBA working-tree changes (`scoring/scorer.py`, `content.py`) are not part of these commits.

## Phase 2 — Totals model retrain (gated)

1. **Retrain** with existing `MLBModelTrainer.train_totals_model()` on seasons 2024 + 2025 + 2026-to-date. Hold-out evaluation: train on games before June 1, test on June 1 – Jul 5; compare MAE and over/under hit rate vs `mlb_totals_v1`.
2. **Save** as `models/mlb_totals_v2.joblib`. Totals model path becomes a config setting with fallback to v1 if the file is missing.
3. **Shadow mode**: v2 powers `best_total` picks, which are recorded and graded nightly but never enter `best_bet` (Phase 1 flag).
4. **Re-entry gate**: after ≥100 graded `best_total` picks under v2, if win rate ≥53% AND cumulative profit > 0, flip `totals_in_best_bet = true`. Otherwise totals stay out of best_bet.

## Rollout & risk

- Phase 1 commits to `main`; Railway auto-deploys. Verify via startup logs and next morning's pick slate (no totals as best bet, value scores spread below 100).
- Rollback = single-commit revert. `best_total` keeps recording either way, so no data loss.
- Railway prod DB is the source of truth for all evaluation queries (connection per `prediction_tracker.py`).

## Testing

- Unit tests for `calculate_value`: regression blend math, MAX_EDGE_PCT rejection, new multipliers, MIN_EDGE boundary, saturation (a 20% raw edge no longer scores 100).
- Unit test for best_bet candidate filtering: totals excluded when flag off, included when on.
- Scheduler fallback test: total-as-best-bet falls back to better of ML/RL.
- Validation sim doubles as an end-to-end check against real data.
