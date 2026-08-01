# Model Governance Handbook

**Purpose:** how models are versioned, approved, rolled out, monitored, and rolled back.
**Last verified against code:** 2026-07-31

---

## 1. The governance failure this exists to prevent

```
mlb_run_diff_v1.joblib   trained_at: 2026-02-09T21:32:48   RMSE 4.431
```

**The only model making live MLB picks was trained before the season started and has never been retrained.** Team stats refresh every two hours; the function mapping those stats to a prediction is frozen at preseason. `retrain_mlb_v2.py` exists and has never produced a deployed artifact.

Nothing in the system flagged this, because nothing owns model freshness. That is what this document assigns.

For context on how much is at stake: run-differential SD is 4.64, the model's RMSE is 4.431, so it explains roughly **9% of variance**. That is near the realistic ceiling for single-game baseball — but it means the model is a small edge on a noisy signal, and small edges degrade quietly.

---

## 2. Model registry

| Artifact | Vertical | Status | Trained | Holdout |
|---|---|---|---|---|
| `mlb_run_diff_v1.joblib` | MLB | **SERVING** | 2026-02-09 | RMSE 4.431 / MAE 3.477 |
| `mlb_totals_v2.joblib` | MLB | SHADOW | ~2026-07-06 | RMSE 4.487 / MAE 3.521 |
| `mlb_totals_v1.joblib` | MLB | fallback | 2026-02-09 | RMSE 4.504 |
| `nfl_mov_v1.joblib` | NFL | SERVING (dark) | Phase 2 | see backtest report |
| `nfl_totals_v1.joblib` | NFL | SERVING (dark) | Phase 2 | totals +10.2u / 53.9% |
| `mov_model.pkl`, `spread_model_v2.pkl`, `totals_model_v3.pkl`, `calibration.pkl` | NBA | mixed | — | see `MODEL_CHANGES_2026-01-13.md` |

**Required artifact metadata** (already the convention — enforce it):
```python
{"model": ..., "feature_cols": [...], "metrics": {...},
 "trained_at": isoformat, "version": "1.0"}
```
`mlb_totals_v2.joblib` has `trained_at: None`. That is a defect; backfill it at next retrain.

---

## 3. Model states

| State | Meaning | Can affect a bet? |
|---|---|---|
| **EXPERIMENTAL** | Local only, not committed as an artifact | No |
| **SHADOW** | Deployed, predictions stored, excluded from `best_bet` | No |
| **SERVING** | Selected by config, produces live picks | Yes |
| **SERVING (dark)** | Deployed and wired, but the scheduler is disabled | No |
| **FALLBACK** | Loaded only if the configured path is missing | Yes, silently |
| **DEPRECATED** | Superseded, retained for rollback | No |

**FALLBACK deserves scrutiny.** `MLBScorer.__init__` silently degrades: configured totals path missing → v1 → and if the run-diff model file is absent entirely, `_estimate_run_diff()` takes over with hardcoded coefficients (`+0.3` home advantage, `× 0.5` ERA weighting). A deploy that loses `models/` would keep serving picks from a hand-written heuristic and log only a warning.

**Required:** a startup assertion that the SERVING model actually loaded, and a `/health` field exposing `run_diff_model_version` and `trained_at`. Silent heuristic fallback in production is not acceptable.

---

## 4. Promotion gates

A model or market is promoted only against a **pre-registered** gate — written down before the data is looked at.

### The reference example (follow this pattern)

MLB totals re-entry gate, defined in `config.py` *before* evaluation:
> ≥100 graded `best_total` picks under the retrained model, ≥53% win rate, positive cumulative units.

Outcome (2026-07-29): the all-games shadow cut reached 125 picks at **52.1% WR, −0.91u** — below break-even. **Gate not met. Totals stayed suppressed.** The decision was mechanical because the threshold predated the result.

This is the single most valuable practice in the repo. Preserve it.

### Gate template

```
MODEL/MARKET:      what is being promoted
FROM -> TO:        SHADOW -> SERVING
SAMPLE:            minimum graded decisions (state it, don't discover it)
PRIMARY METRIC:    with threshold
GUARDRAIL:         what would abort even if primary passes
EVALUATION WINDOW: dates, fixed in advance
DECIDED BY:        named person
ROLLBACK:          exact config change that reverses it
```

### Standing thresholds

| Promotion | Requirement |
|---|---|
| New model → SHADOW | Holdout beats incumbent on MAE **and** hit rate; artifact metadata complete |
| SHADOW → SERVING | ≥100 graded decisions; beats incumbent on realized units; no calibration regression |
| Market → `best_bet` | Its own pre-registered re-entry gate |
| Any calibration constant change | Derived from measured data, **not** tuned on P&L (see §6) |

---

## 5. Rollout and rollback

Rollout is **config, not code**. Every SERVING model is selected by a settings path:

```python
mlb_totals_model_path = "models/mlb_totals_v2.joblib"
nfl_mov_model_path    = "models/nfl_mov_v1.joblib"
```

`mlb_run_diff` is the exception — hardcoded as `DEFAULT_RUN_DIFF_MODEL`. **Move it to config** so it is rollback-able the same way.

### Rollback procedure

1. Revert the config path (or flip the market flag to `False`).
2. Redeploy. No model code changes.
3. Record the incident in this file's §8 log.

Because artifacts are committed files, rollback never requires retraining. **Never delete a superseded artifact.**

### Kill switches

```python
suppress_totals        # stop scoring a market entirely
totals_in_best_bet     # score + store, never bet
runline_in_best_bet    # score + store, never bet
nfl_scheduler_enabled  # generate nothing at all
```

The `*_in_best_bet` pattern — keep computing, stop betting — is what makes shadow evaluation and evidence-based re-entry possible. Preserve it on every new market.

---

## 6. Calibration constants are governed artifacts

Constants converting model output to probability are **model parameters**, and were historically the weakest-governed part of the system.

```python
RUN_DIFF_LOGISTIC_K = 0.391   # was 0.5, hand-picked: "gives reasonable spread"
```

The old value implied a margin SD of 3.63 runs against a measured 4.64 — a curve ~28% too steep, producing systematic overconfidence. Measured on the model's own graded picks, it overstated edge by **+9 points on underdogs and +23 points on favorites**.

Rules:

1. **Every probability constant must be derived from measured data**, with the derivation in a comment. `0.391 = (π/√3)/4.64` where 4.64 is the observed SD over 1,463 games.
2. **A constant must never be tuned on P&L.** Fit it to the distribution, *then* check P&L as corroboration. For k this order mattered: the value came from margin SD independently, and separately landed on the backtest optimum.
3. **Pin it with a test.** `tests/unit/test_mlb_run_diff_calibration.py` asserts k matches the observed SD and that the curve is flatter than the old guess.
4. **Re-derive each offseason.** Margin SD is a league-season property.

Still outstanding: `_total_to_over_prob` uses `k = 0.4` ("more conservative"). Its implied SD (4.53) happens to match reality (4.54), so it is correct by luck rather than derivation. Document or re-derive it.

### Known mis-specified curve
`_run_diff_to_cover_prob` shifts the win curve by the spread, which assumes fixed-scale logistic margins. **27.2% of MLB games are decided by exactly one run**, and the ±1.5 runline sits on that spike. Empirical vs modelled P(margin ≥ 2) by bucket: **+0.039 / +0.059 / −0.043 — the sign flips.** Runline stays paused until this is rebuilt as an empirical cover curve. This is documented in-code as a CAVEAT; do not "fix" it by adjusting k.

---

## 7. Monitoring

### Required, currently missing

| Monitor | Alert | Why |
|---|---|---|
| Model age | `trained_at` > 45 days during season | Would have caught the Feb-9 model in March |
| Fallback engaged | any heuristic/v1 fallback in prod | Silent degradation |
| Feature coverage | default rate > 20% on any served feature | Stats pipeline broken upstream |
| Calibration drift | rolling predicted vs realized win rate, 100-pick window | The k problem, detected automatically |
| Pick volume anomaly | zero picks with odds present | Distinguishes "no edge" from "no data" |
| CLV | rolling closing-line value | See `04-research-experimentation.md` §5 |

### Retrain cadence (adopt)

| Vertical | Cadence |
|---|---|
| MLB | Monthly in-season + preseason full rebuild |
| NFL | Weekly in-season (already implemented via `season_update`) |
| NBA | Monthly in-season |

Each retrain is a SHADOW → gate → SERVING promotion, never a direct overwrite.

---

## 8. Change log

Append one row per model/constant change reaching production.

| Date | Change | Rationale | Result |
|---|---|---|---|
| 2026-07-06 | MLB totals v1 → v2 (SHADOW) | Holdout MAE 3.538 vs 3.564, 55.6% hit | Never promoted; re-entry gate failed |
| 2026-07-21 | Runline paused from `best_bet` | Sign-pairing bug inflated tracked record | Corrected record: −128u. Stays paused |
| 2026-07-22 | Runline sign fix + re-validation | Grade from frozen columns; legacy guard | No real edge found |
| 2026-07-28/29 | Grading rewritten to frozen columns | `best_ml` per-row overwrite stranded 109 snapshots as ungradeable | Clean record: ML 343 graded / +24.50u |
| 2026-07-29 | Totals re-entry gate evaluated | 125 picks, 52.1%, −0.91u | **Gate failed. Stays suppressed** |
| 2026-07-30 | `RUN_DIFF_LOGISTIC_K` 0.5 → 0.391 | Fitted to measured margin SD 4.64 | Deployed `de196dc`. Backtest +38.90u vs +32.87u |
| 2026-07-31 | Populate 24 constant-serving features | Train/serve skew: model trained on real variance, served league-average constants | Deployed `481463e`. **Worth +10.0pts of holdout winner accuracy — see below** |
| 2026-08-01 | Feature vector built by name; 28 → 33 features | Twelve serving names had no counterpart in training output | Deployed `2ccb803`. Behaviour-preserving for the incumbent |
| 2026-08-01 | **2026 retrain evaluated → NO-GO** | 1,405 point-in-time games, 33 features | **Rejected. Zero skill — see below** |

### 2026 retrain: NO-GO (2026-08-01)

Chronological holdout, 281 games (2026-07-06 → 07-30), both models scored on
the same games.

| model | MAE | RMSE | hit% |
|---|---|---|---|
| challenger — 2026, 1,124 games, 33 features | 3.564 | 4.733 | 45.2 |
| incumbent — 2024-25, 4,931 games, 28 features | **3.436** | **4.646** | **59.8** |
| baseline — predict the training mean | 3.565 | 4.725 | 47.7 |

The challenger is **indistinguishable from predicting the mean** (MAE −0.000,
RMSE +0.008 against baseline) and its winner accuracy is *below* baseline.
1,124 games cannot support 33 features at `num_leaves=31`; the incumbent had
4.4× the data. **Keep the incumbent.** Revisit when a second season of
point-in-time history exists — the pipeline to build it now works.

Of the five recovered features only two carried any gain
(`away_last_10_win_pct` rank 4, `weather_factor` rank 14); `temperature`,
`is_dome` and `home_last_10_win_pct` scored exactly zero. That is evidence
about this sample size, not proof the features are worthless.

### The model was never the problem — the features were

Same incumbent model, same 281-game holdout, only the inputs differ:

| inputs | MAE | RMSE | hit% |
|---|---|---|---|
| real features (production since `481463e`) | 3.436 | 4.646 | **59.8** |
| constants (what production actually served all season) | 3.536 | 4.744 | **49.8** |

**+10.0 percentage points of winner accuracy from the feature fix alone**, on a
model nobody retrained. Blind, it was a coin flip — worse than always backing
the home team, who won 52.3% of these games.

Caveat: 281 games, roughly 2–3 standard errors. Real, but confirm forward.
Note also that sign accuracy is not betting profit — the market prices these
games too, and CLV remains the instrument that decides whether this converts.

---

## 9. Open governance items

1. **Retrain the MLB run-diff model.** Six months stale. Highest-value single action.
2. **Fix the v2 feature-vector contract** in the same change — see `02-feature-engineering-playbook.md` §2. Deploying v2 against the hardcoded 28-slot builder misaligns features.
3. Move `DEFAULT_RUN_DIFF_MODEL` into config for rollback parity.
4. Expose model version + `trained_at` on `/health`.
5. Backfill `trained_at` on `mlb_totals_v2.joblib`.
6. Re-derive or document the totals `k = 0.4`.
7. Build the runline empirical cover curve; keep paused until validated on clean forward data.
