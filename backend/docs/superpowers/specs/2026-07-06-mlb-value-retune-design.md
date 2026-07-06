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

**Revision 2026-07-06 (approved):** a clean replay of the 586 qualifying graded picks overturned two details of the first draft: the ML 0.80 penalty costs profit (+150.0u at 0.80 vs +157.8u at 1.0), and market regression cannot change candidate ranking (a uniform 50% blend rescales every candidate identically) — its only legitimate role is display. The revised mechanics below keep the proven pick set and fix the two real defects: score saturation (65% of picks score exactly 100) and the resulting tie-break bias where `max()` silently prefers ML because it is added to the candidate list first.

### `src/services/mlb/value_calculator.py`
1. **Gate unchanged + blowup cap** — a bet qualifies (`is_value_bet`) exactly as today (raw-edge score `raw_edge_pct * 4.0 >= 55` and `raw_edge >= MIN_EDGE`), **plus** new **`MAX_EDGE_PCT = 80.0`**: raw `edge_pct` above 80 marks the bet `is_value_bet = False` (model blowup). This preserves the +160u season pick set.
2. **Selection metric** — new field `sort_score` on `MLBValueResult`: unclamped `edge_pct * market_multiplier`. `find_best_value` ranks by `sort_score`, not the clamped display score. Fixes the ML tie bias.
3. **Display score** — `value_score = min(100, 100 * tanh(blended_edge_pct / 20) * market_multiplier + favorite_bonus)` where `blended_edge_pct = edge_pct * (1 - MARKET_REGRESSION_WEIGHT)` with `MARKET_REGRESSION_WEIGHT = 0.50`. Backtest distribution: 0% saturation, min 33 / median 77 / max 96. UI thresholds (65/70/80) remain meaningful.
4. **Market multipliers unchanged** — moneyline 0.95, total 0.90 (the 0.80 ML penalty is explicitly rejected; comment this in code).
5. **`MIN_EDGE` stays 0.10** — the 0.10-0.15 absolute-edge bucket is the best performer since May 23 (56.9% WR, +17.9u over 72 bets); do not raise to 0.12.
6. **Cleanup** — `EDGE_SCALE_FACTOR = 400` is dead code (formula hardcodes `* 4.0`); set it to 4.0 and use it in the gate formula.

### `src/config.py`
6. New setting **`totals_in_best_bet: bool = False`** (env-overridable like `suppress_totals`). Independent of `suppress_totals`, which controls whether totals are scored at all.

### `src/services/mlb/scorer.py`
7. Where `best_bet` is chosen (`find_best_value(all_values)`, ~line 488): when `totals_in_best_bet` is false, exclude total-market values from the `best_bet` candidate list. `best_total` is still computed, recorded, and graded (shadow track record for Phase 2).

### `src/tasks/mlb_scheduler.py`
8. Defense-in-depth (port from `b004564`): in `snapshot_predictions_async`, if the chosen best bet is a total while `totals_in_best_bet` is false, fall back to the better-scored of `best_ml` / `best_rl` before writing snapshot `best_bet_*` fields.

### Validation (before deploy)
Offline re-simulation script (scratchpad, not committed) replaying graded season snapshots through the actual new code (`market_prob = 1/odds`, `model_prob = market_prob + stored raw_edge`, confidence 0.5):
- Assert: (a) no totals in simulated best_bet; (b) zero picks score exactly 100; (c) simulated P&L ≥ +150u on the season window (clean-sim baseline +157.8u vs +160u actual).
- If simulation shows degradation, stop and reassess before deploying.

Planning-phase sims already run (2026-07-06, basis for the revision): linear-scale sweep contaminated by a profit tie-break leak and discarded; clean unclamped-selection sim +157.8u (ML mult 1.0) / +150.0u (0.80); tanh display distribution min 33 / med 77 / max 96, 0% saturation. The deploy-gate sim re-runs against the real implementation as a regression check.

### Non-goals / unaffected
- NRFI pipeline (separate content + grading path) — untouched.
- Social recap posts read `best_bet_*` from snapshots, so totals suppression flows through automatically — no social code changes.
- The uncommitted NBA working-tree changes (`scoring/scorer.py`, `content.py`) are not part of these commits.

## Phase 2 — Totals model retrain (gated)

1. **Retrain** with existing `MLBModelTrainer.train_totals_model()` on seasons 2024 + 2025 + 2026-to-date. Hold-out evaluation: train on games before June 1, test on June 1 – Jul 5; compare MAE and over/under hit rate vs `mlb_totals_v1`.
2. **Save** as `models/mlb_totals_v2.joblib`. Totals model path becomes a config setting with fallback to v1 if the file is missing.
3. **Shadow mode**: v2 powers `best_total` picks, which are recorded and graded nightly but never enter `best_bet` (Phase 1 flag).
4. **Re-entry gate**: after ≥100 graded `best_total` picks under v2, if win rate ≥53% AND cumulative profit > 0, flip `totals_in_best_bet = true`. Otherwise totals stay out of best_bet.

**Shadow deploy record (2026-07-06):** v2 trained on 6,299 games (2024-2026); holdout gate PASSED — v2 MAE 3.538 vs v1 3.564, line hit-rate 55.6% both (Jun 1 - Jul 5, 468 games, 169 with recorded lines). Served via `mlb_totals_model_path` code default (Railway CLI unauthenticated; env var flip not used).

Re-entry gate check (run every ~2 weeks after 2026-07-06):

    SELECT count(*) AS picks,
           count(*) FILTER (WHERE best_total_result = 'win') AS wins,
           round(100.0 * count(*) FILTER (WHERE best_total_result = 'win')
                 / nullif(count(*) FILTER (WHERE best_total_result IN ('win','loss')), 0), 1) AS wr,
           round(sum(best_total_profit)::numeric / 100, 1) AS units
    FROM mlb_prediction_snapshots
    WHERE game_date >= '2026-07-07' AND best_total_result IS NOT NULL;

Gate: picks >= 100 AND wr >= 53 AND units > 0 → set `totals_in_best_bet = True` (config default or `TOTALS_IN_BEST_BET=true` on Railway). Otherwise totals stay out.

## Rollout & risk

- Phase 1 commits to `main`; Railway auto-deploys. Verify via startup logs and next morning's pick slate (no totals as best bet, value scores spread below 100).
- Rollback = single-commit revert. `best_total` keeps recording either way, so no data loss.
- Railway prod DB is the source of truth for all evaluation queries (connection per `prediction_tracker.py`).

## Testing

- Unit tests for `calculate_value`: regression blend math, MAX_EDGE_PCT rejection, new multipliers, MIN_EDGE boundary, saturation (a 20% raw edge no longer scores 100).
- Unit test for best_bet candidate filtering: totals excluded when flag off, included when on.
- Scheduler fallback test: total-as-best-bet falls back to better of ML/RL.
- Validation sim doubles as an end-to-end check against real data.
