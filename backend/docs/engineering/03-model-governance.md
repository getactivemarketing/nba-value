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

### 2024-26 retrain with bullpen + FIP: NO-GO (2026-08-03)

The first NO-GO was reasoned from too little. "1,405 games cannot support 33
features" was really "not at `num_leaves=31`" — the hyperparameters of a
4,931-game model applied to a quarter of the data. Prior-season history was
also fetchable all along (gameLog and schedule both serve 2024/2025), so
"wait until April" was wrong too.

Rebuilt from the API: **6,012 games** (2024: 2,231 / 2025: 2,247 / 2026: 1,534),
41 features at 100% coverage, all point-in-time. Chronological
train/validate/holdout; tuning swept leaf counts on validation only; the
holdout (902 games, 2026-05-22 → 08-02) was touched once.

| model | MAE | RMSE | hit% |
|---|---|---|---|
| challenger — 41 features incl. bullpen + FIP | 3.626 | 4.720 | 54.0 |
| same, minus bullpen/FIP (33 features) | 3.630 | 4.723 | 54.8 |
| **incumbent — 2024-25, 28 features** | **3.578** | **4.709** | **55.5** |
| baseline — predict the training mean | 3.663 | 4.743 | 48.3 |

**Two findings, and they point opposite ways.**

The tuning fix was real: `num_leaves=7` won on validation, and the challenger
now has genuine skill (RMSE 4.720 vs 4.743 baseline, 54.0% vs 48.3%) where the
2026-only model had none. The earlier conclusion was too broad.

But **the new features do not earn their place**: removing bullpen and FIP
entirely changes RMSE by −0.003 and *improves* hit rate by 0.8 points. And the
incumbent still wins outright, despite the challenger having seen early 2026
that the incumbent never did.

Feature gains are split rather than uniformly flat: `home_bullpen_whip` ranks
8/40 — real signal — while `away_bullpen_era` ranks 39/40 and
`home_bullpen_ip_l3` dead last at 40/40. FIP ranks 23-35/40, plausibly because
it is highly correlated with the ERA columns already present.

**KEEP THE INCUMBENT.** Bullpen and FIP stay collected but unwired. Why the
incumbent still wins is not established — its original training set may be
constructed differently from this reconstruction — and that gap should be
closed before the next attempt rather than assumed away.

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

> **CONFIRMED FORWARD 2026-08-13: IT DID NOT HOLD. See §10.**

---

## 10. The +10pt did not generalise, and why (2026-08-13)

The caveat above was the operative sentence. Forward data arrived and the
result reversed. The backtest itself reproduces exactly — same model, same
production feature path, scored offline today:

| window | n | model hit% | always bet home | model edge | MAE | RMSE |
|---|---|---|---|---|---|---|
| 2026-07-06..07-30 (the holdout) | 288 | **60.1%** | 52.1% | **+8.0** | 3.434 | 4.649 |
| 2026-07-31..08-13 (forward) | 177 | **47.5%** | 57.1% | **−9.6** | 3.529 | 4.383 |

60.1 / 3.434 / 4.649 against the doc's 59.8 / 3.436 / 4.646 — the original
evaluation was computed correctly. A 17.6-point swing between adjacent windows.
Live production over the forward window hit 49.2%, agreeing with the 47.5%
scored offline, so serving does not diverge from evaluation.

**The fix did land.** Default substitution fell from 10.6% of feature slots in
the holdout window to **1.8%** live; team stats went from ~21% defaulted to
zero; `mlb_pitcher_stats` went from ~55 to ~330 rows/day on 07-31. "The fix
never reached production" was checked and is false.

### Why hit% swings that hard

Predictions have SD ≈ 0.97 runs against outcomes with SD ≈ 4.6 — roughly 3% of
variance, not the 9% quoted in §1 (that was the February holdout). With
predictions clustered near zero the SIGN is close to a coin flip, so
directional accuracy swings ±10 points on noise. The tell: RMSE stayed flat
(4.649 -> 4.383) while hit% collapsed. A real regression moves both.

**A single-window holdout hit% cannot promote a change on this model. The
metric's window-to-window noise exceeds any effect being measured.**

### The model carries no signal the market has not already priced

177 games with a validated pre-game consensus (>=3 books, devigged per book,
closest pre-game quote, rebuilt from `mlb_odds_snapshots`):

| fit | logloss |
|---|---|
| base rate only | 0.68314 |
| **model only** | **0.68302** |
| market only | 0.67579 |
| market + model | 0.67222 |

`logit(model)` alone: β = −0.084, t = −0.20. Added to the market:
β = −0.544, t = −1.11. Zero either way, negative point estimate both times.
`corr(model, market) = +0.478`, so about half of what it knows is the market's
own information rediscovered. At n=177 this cannot rule out a *small* true
edge; it does rule out one large enough to clear a 4.6% vig.

### The selection rule makes it worse than neutral

- model probability SD = **0.0889**
- market probability SD = **0.0877**

The model is as *confident* as a market that actually knows something, while
carrying no information. So `edge = model − market` is essentially noise with
SD ≈ 0.090, and `MIN_EDGE = 0.10` bets the ~27% tail of it — **selecting
precisely the games where the model's own error is largest.** This is adverse
selection, not a small edge lost to vig, and it explains the rest of the
picture: negative CLV despite positive `market_move`, inverted confidence
buckets (predicted 0.659 -> actual 0.486), and best_bet underperforming the
model's average game.

The corollary is uncomfortable and worth stating plainly: the correct
adjustment for zero skill is to shrink predictions toward the market, which
yields no bets. More features and more retuning widen the spread, which makes
adverse selection worse, not better. Two retrains already returned NO-GO.
Making this work requires information the market has not priced — timing-based
edges (lineups, late scratches, bullpen availability, weather) rather than
better modelling of the same public inputs.

### Bullpen and FIP: NO signal in the incumbent's residual (2026-08-13)

`home_bullpen_whip` ranking 8/40 in the 2026-08-03 retrain was read as "real
signal, discarded because the aggregate lost". It is not. Gain-based importance
measures how much a fitted model *used* a feature, not whether the feature
helps out of sample — a nearly-random column still earns splits.

Direct test instead. Freeze the incumbent (trained 2024-25, so every 2026 game
is out of sample), take its residual, and ask whether bullpen/FIP explains any
of it. n=1,534, 100% feature coverage:

| term (standardised) | t |
|---|---|
| home_bullpen_era | +0.24 |
| away_bullpen_era | +1.11 |
| home_bullpen_whip | −0.57 |
| away_bullpen_whip | −0.85 |
| home_bullpen_ip_l3 | +0.55 |
| away_bullpen_ip_l3 | −0.07 |
| home_starter_fip | +1.12 |
| away_starter_fip | −0.68 |

R² = 0.00235, F(8,1525) = 0.448 (needs 1.94 for p<0.05). Assumption-free check:
a permutation test puts **p = 0.90 — 1,807 of 2,000 random shuffles of the
residual explained as much as the real features.** Nothing here.

(`starter_fip_diff` is exactly `home_starter_fip − away_starter_fip`, so the
design matrix is singular with it included; dropped. The joint test is
unaffected — same fitted subspace.)

Keep collecting both; the acquisition rationale in `bullpen.py` still holds and
a second season may change this. But bullpen is not the missing edge, and no
further tuning of it is warranted on current data.

### The incumbent is worse than a constant out of sample

Same 1,534 games:

    incumbent RMSE        4.6434
    predict-the-mean RMSE 4.6280
    out-of-sample R²      -0.00665

Not "small edge". Negative. §1 of this document estimates ~9% of variance
explained from the February holdout; on 2026 games out of sample it is below
zero. Any statement about this model's accuracy that predates 2026-08-13
should be re-derived before it is relied on.

### Two data-integrity defects found in the course of this

1. **`entry_novig_prob` was corrupt for the whole pre-August season.** `9f278a9`
   froze the column because deriving it was unsafe, then backfilled 964 rows
   using that same derivation. Bucketed against outcomes those rows read
   implied 0.231 vs actual 0.490 (n=577, 61% of them) — a price that cannot
   exist. `market_move` and `vig_paid` derive from it. From 2026-07-31 the
   stored values match a rebuilt consensus to a median of 0.0068 (93.9% within
   two points), so only the backfill is affected. Retracted to NULL by
   `src/tasks/repair_entry_novig_backfill.py` (947 rows); unrecoverable, since
   no pre-game price was ever recorded for those games. `clv` is unaffected —
   it never used the entry price.

2. **`mlb_markets` holds post-game settled odds.** Updated in place and never
   stops updating, so completed games devig to implied 0.774 -> actual 0.995.
   Any historical join to it shows a spectacular edge that does not exist. Use
   `mlb_odds_snapshots`. Note also that odds there are DECIMAL, not American.

---

## 9. Open governance items

0. **Promotion gate: CLV on shadow picks, never holdout hit%.** §10 — hit% noise
   between adjacent windows was 17.6 points, larger than any effect measured.
   No model change ships on a single-window hit% again.
0b. **Decide whether best_bet should fire at all** while `logit(model)` sits at
   β = −0.084 (t = −0.20) and `MIN_EDGE` is selecting the model's own noise.
   This is a product decision, not a modelling one.
1. **Retrain the MLB run-diff model.** Six months stale — but note §10: two
   retrains already returned NO-GO, and retuning widens a spread that is
   already wider than the model's skill. Retraining alone will not fix this.
2. **Fix the v2 feature-vector contract** in the same change — see `02-feature-engineering-playbook.md` §2. Deploying v2 against the hardcoded 28-slot builder misaligns features.
3. Move `DEFAULT_RUN_DIFF_MODEL` into config for rollback parity.
4. Expose model version + `trained_at` on `/health`.
5. Backfill `trained_at` on `mlb_totals_v2.joblib`.
6. Re-derive or document the totals `k = 0.4`.
7. Build the runline empirical cover curve; keep paused until validated on clean forward data.
