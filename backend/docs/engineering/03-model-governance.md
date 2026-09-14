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

## 11. The CLV gate resolved: FAILED, permanently (2026-08-24)

The pause shipped `9cc4f90` (2026-08-14) with a pre-registered re-entry gate:
**>=100 CLV-measured picks, mean CLV > 0, lower 95% bound above zero.** It was
written that way so it could not be moved once the data got interesting.

It has resolved against re-entry, at n=90, by a margin the remaining 10 picks
cannot touch:

| metric | value |
|---|---|
| n | 90 |
| mean CLV | **-0.00612** |
| SD | 0.00874 |
| SE | 0.00092 |
| 95% CI | **[-0.00792, -0.00431]** — entirely below zero |
| t | **-6.64** |
| beat the close | **18 / 90 (20.0%)** |

`P(<=18 beats | p=0.5) = 4.0e-09`. This is not "no edge" — that looks like CLV
scattered around zero. The model is **systematically on the wrong side of the
closing line**: it takes sides the market subsequently moves against.

### It is not the longshot artifact

The obvious confound is mechanical. The selection rule steers to underdogs
(§10), dogs drift out as late money lands on favourites, so negative CLV could
be a pricing artifact rather than a statement about the model. It is not.
CLV by price bucket:

| bucket | n | mean CLV | t | beat close |
|---|---|---|---|---|
| fav <2.00 | 14 | **-0.01321** | -5.32 | 1/14 (7%) |
| 2.00-2.40 | 29 | -0.00888 | -5.12 | 5/29 (17%) |
| 2.40-2.80 | 31 | -0.00277 | -2.91 | 6/31 (19%) |
| dog 2.80+ | 16 | -0.00138 | -0.78 | 6/16 (38%) |

Negative in every bucket, and **worst on favourites** — the opposite of what
the drift artifact predicts. The effect is diffuse, which also disposes of
fading the signal: there is no concentrated pocket to harvest, and a -0.006
mean sits inside the vig.

### The W/L record over the same period

Moneyline, by window, to 2026-08-23 (these are PAPER — see below):

| window | record | win% | units | avg odds | implied breakeven |
|---|---|---|---|---|---|
| last 4 days | **1-16** | 5.9% | -14.38u | 2.78 | 36.0% |
| last 7 days | 5-23 | 17.9% | -16.27u | 2.66 | 37.6% |
| last 14 days | 17-36 | 32.1% | -14.40u | 2.48 | 40.3% |
| last 30 days | 43-72 | 37.4% | -15.59u | 2.41 | 41.5% |
| last 54 days | 73-105 | 41.0% | -9.79u | 2.40 | 41.7% |

The 1-16 window is 2026-08-20..08-23: 17 picks, market-implied 6.3 expected
wins, 1 actual. Note that window was selected *because* it was bad, so treat it
as the tail of the decline the CLV series already describes, not as an
independent shock. The honest summary is the 54-day line: 41.0% against a
41.7% breakeven, i.e. paying the vig with no edge, with avg price drifting
2.40 -> 2.78 as the selection rule reached for longer numbers.

### The pause held — none of this was published

Verified 2026-08-24: `/api/v1/mlb/picks/top` returns
`{"picks":[],"total":0,"paused":true}`, and `sms_alert_sent` is 0 for every day
from 08-15 onward (it was still firing 08-10..08-14). Scoring, freezing,
grading and CLV all continued as designed. The -14.38u is a paper record of
what would have been published.

### Decision

**best_bet stays paused permanently. This closes open governance item 0b.** The
gate was the mechanism for reversing that call and it has returned -6.6 sigma
in the wrong direction. Do not re-open it on a W/L run, a retrain, or a good
week; a new gate would have to be argued from a different instrument entirely.

Consequence for §9 item 1: retraining cannot be justified as a route back to
best_bet. Two retrains already returned NO-GO, and CLV now says the failure is
in what the model knows relative to the market, which a retrain of the same
features on the same target does not address.

---

## 12. The end-of-July regime change: revelation, not regression (2026-08-24)

Raised by the founder, who noticed the algo "seems messed up since the end of
July" and pointed at 2026-07-20..07-31. He was right that something changed.
It was not a bug.

### The W/L swing he saw is not significant

Moneyline picks against what the market implied for those same picks:

| period | n | W | expected | diff | z |
|---|---|---|---|---|---|
| Apr 03-Jun 30 | 263 | 119 | 110.6 | +8.4 | +1.06 |
| Jul 01-Jul 19 | 45 | 18 | 19.5 | -1.5 | -0.44 |
| **Jul 20-Jul 31** | 46 | 24 | 20.1 | **+3.9** | **+1.17** |
| Aug 01-Aug 12 | 43 | 16 | 18.7 | -2.7 | -0.84 |
| Aug 13-Aug 23 | 45 | 15 | 18.5 | -3.5 | -1.08 |

Everything inside +/-1.2 SD. The good stretch and the collapse are both ~1
sigma on ~45-pick samples. Consistent with §10: hit%-style metrics on this
model cannot resolve anything at these sample sizes.

### But the model's OUTPUT changed, and that is not noise

`481463e` (07-31, populate the 24 constant features) and `2ccb803` (08-01,
build feature vectors by name) materially changed what the model emits:

| | PRE-FIX (Apr 3-Jul 31) | POST-FIX (Aug 1-Aug 23) |
|---|---|---|
| n games | 1,484 | 314 |
| model output mean | +0.190 | **-0.006** |
| model output SD | 0.569 | **0.932** (+64%, F=2.68) |
| outcome mean | +0.041 | +0.268 |
| corr(pred, actual) | +0.0649 [+0.014,+0.115] | +0.0521 [-0.059,+0.162] |
| mean error (bias) | +0.149 (z=+1.25) | -0.273 (z=-1.06) |

**Predictive correlation did NOT change: Fisher z = +0.21, not significant.**
The fix did not destroy signal, because there was none either side. The
apparent loss of the home-field tilt (model picks home fell 53.7% -> 48.2%
while actual home win% ROSE to 56.9%) is a real change in output but its
effect on accuracy is z = -1.06 — not significant, and not evidence of a
wiring defect. A gross home/away swap would have driven correlation to zero or
negative; it did not move at all.

### The mechanism

Pre-fix the model was, in effect, a **constant predictor** — output SD 0.569
against outcome SD 4.6. A constant scores R^2 ~ 0: harmless. It also rarely
disagreed with the market enough to cross `MIN_EDGE = 0.10` by much.

Post-fix, real features widened output spread 64% while correlation stayed at
~0.05. **All of the added variance is noise.** So `edge = model - market` is now
drawn from a much wider noise distribution, crosses `MIN_EDGE` more often and
more extremely, and each crossing selects the game where the model's own error
is largest. Adverse selection got materially worse — which is what both the W/L
decline and the -6.64 t on CLV (§11) record.

**The blindfolded model was accidentally safer.** A model that cannot disagree
with the market cannot bet badly. The Apr-Jun +18u was a near-constant
predictor riding variance; the fix removed the accidental protection. This is
§10's warning made literal: "retuning widens a spread that is already wider
than the model's skill."

### Consequences

1. **Supersedes how §10 framed the +10pt reversal.** §10 reads it as "hit% is
   too noisy to promote on" — true, but the sharper statement is that the fix
   worked as engineering, widened a signal-less spread, and thereby made bet
   selection worse.
2. **`MIN_EDGE = 0.10` is now mis-specified.** It was implicitly set against a
   0.569-SD predictor and is applied to a 0.932-SD one, so it admits a far
   fatter tail than intended. Re-deriving it reduces volume and damage; it does
   NOT create an edge. Do not mistake it for a fix.
3. **This closes the "was the pre-August model better" question left open by
   §11.** CLV cannot answer it (no odds history before 07-31), but predictive
   correlation is statistically indistinguishable across the boundary. The
   pre-August model was not better — it was quieter about not having an edge.
4. Not fully excluded: a *subtle* misalignment. n=314 makes the bias test ~1
   sigma. There is no positive evidence for one.

**Nothing here is a route back to un-pausing.** The binding constraint is
unchanged and unchanged by any threshold: the features do not predict the
target.

---

## 9. Open governance items

0. **Promotion gate: CLV on shadow picks, never holdout hit%.** §10 — hit% noise
   between adjacent windows was 17.6 points, larger than any effect measured.
   No model change ships on a single-window hit% again.
0b. ~~**Decide whether best_bet should fire at all**~~ — **CLOSED 2026-08-24, §11.**
   The pre-registered CLV gate resolved at n=90: mean CLV −0.00612, t = −6.64,
   95% CI entirely below zero, 18/90 beat the close. best_bet stays paused
   permanently.
1. **Retrain the MLB run-diff model.** Six months stale — but note §10: two
   retrains already returned NO-GO, and retuning widens a spread that is
   already wider than the model's skill. Retraining alone will not fix this.
2. **Fix the v2 feature-vector contract** in the same change — see `02-feature-engineering-playbook.md` §2. Deploying v2 against the hardcoded 28-slot builder misaligns features.
3. Move `DEFAULT_RUN_DIFF_MODEL` into config for rollback parity.
4. Expose model version + `trained_at` on `/health`.
5. Backfill `trained_at` on `mlb_totals_v2.joblib`.
6. Re-derive or document the totals `k = 0.4`.
7. Build the runline empirical cover curve; keep paused until validated on clean forward data.
