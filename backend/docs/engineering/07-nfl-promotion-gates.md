# NFL Promotion Gates — pre-registered 2026-08-03

**Status:** PRE-REGISTERED. Written before the first 2026 regular-season
prediction and before any 2026 forward data exists.
**Amendment rule:** these thresholds may be changed only *before* the window
they govern opens. Changing a threshold after seeing results voids the gate
and restarts the window. Any amendment is recorded in §7 with its date and
reason.

---

## 1. Current state — everything is off

```
nfl_totals_in_best_bet   False    (disabled 2026-08-03)
nfl_spread_in_best_bet   False
nfl_ml_in_best_bet       False
nfl_scheduler_enabled    False
```

Totals was the one live market, enabled on a historical **+10.2u / 53.9%**.
That figure came from a walk-forward using **random-split** residuals. Under
rolling-origin out-of-fold validation on 1,537 games:

| market | OOF MAE | market-only MAE | side% | 95% CI | verdict |
|---|---|---|---|---|---|
| MOV | 10.942 | 9.917 | 49.7 | [47.2, 52.2] | **below break-even** |
| Totals | 11.214 | 10.413 | 52.4 | [49.9, 54.9] | indistinguishable |

Break-even at -110 is 52.38%. **Neither model beats the closing line on MAE.**
Totals by season runs 54.3 / 51.7 / 51.9 / 56.2 / 50.7 / 51.8 / 50.0 — the
apparent edge sits in the older folds and the last three average 50.8%.

`resid_std` was also understated by the random split: **+11.6% (MOV)**, +2.6%
(totals). Since `resid_std` *is* the probability scale, every production MOV
probability was ~12% overconfident.

**Expectation: these gates will not be met in 2026.** They are written to be
failed honestly rather than to justify a launch.

---

## 2. What is measured

Every model scores **every market candidate at every scoring timestamp**, into
the append-only `nfl_shadow_predictions` store:

| model_key | what it is |
|---|---|
| `market_only` | the book's own number as the forecast — the baseline to beat |
| `incumbent` | `nfl_mov_v1` / `nfl_totals_v1`, trained 2010-2023 |
| `challenger` | rolling-origin shadow, trained 2010-2024, OOF resid_std |
| `v2` | reserved |

All four see identical inputs, so differences are attributable to the model
rather than to separately-collected samples.

**Probabilities and uncertainty come only from rolling-origin out-of-fold
residuals.** In-sample or random-split residuals are not permitted as a
calibration source. `resid_std` is stored per row because it *is* the
calibration.

---

## 3. Reporting dimensions

CLV, calibration, and ranking are reported by **model × market**, and sliced by:

- **week** (early-season features are thinner)
- **line range** (favourite / pick'em / dog; totals low / mid / high)
- **weather** (dome, wind bucket, temperature bucket)
- **quarterback continuity** (both starters as expected / one change / both)
- **time to kickoff** (opener, midweek, day-of, final hour)

A signal that exists only in one slice is a slice-specific finding and must be
re-registered as its own hypothesis before it can be promoted. Discovering an
edge in a subgroup after the fact is not the same as predicting one.

---

## 4. Gate — ordered, and the order is binding

A market may enter `best_bet` **only** when every stage passes, in sequence.
Later stages are not examined until earlier ones pass.

### Stage 1 — Coverage (integrity)
- [ ] ≥ 95% of games produce a complete feature row (no default-filled model)
- [ ] ≥ 95% of predictions have a matched closing line
- [ ] Zero silent-default scoring: `build_live_feature_row` returning `None`
      must remain the behaviour on missing inputs

### Stage 2 — Calibration vs the market (the real bar)
Over ≥ 150 forward predictions in that market:
- [ ] Model **log loss < market-only log loss**
- [ ] Model **Brier < market-only Brier**
- [ ] Expected calibration error ≤ 0.05
- [ ] No probability bin off by > 0.10 with n ≥ 30

If a model cannot out-predict the book's own number, nothing downstream matters.

### Stage 3 — Forward CLV
Over ≥ 150 forward predictions:
- [ ] **Median price CLV > 0**
- [ ] **> 50% of predictions with positive price CLV**
- [ ] Median line CLV ≥ 0 for spread/total markets
- [ ] CLV not driven by one slice: positive in ≥ 3 of 4 time-to-kickoff buckets
- [ ] CLV not driven by one book

### Stage 4 — Ranking coherence
- [ ] Higher model confidence → better average CLV (monotone across ≥ 3 bands)
- [ ] Monotone in **CLV**, not in realized win rate. Win rate over a partial
      season is noise at these sample sizes and must not be used here.

### Stage 5 — Stability
- [ ] Stages 2 and 3 hold across ≥ 2 distinct weather regimes
- [ ] No single team, line range, or week driving the result
- [ ] Result survives dropping the best and worst weeks

### Stage 6 — Human decision
- [ ] Named person reviews and signs off
- [ ] Rollback is a config flag flip, verified before enabling

---

## 5. What is explicitly NOT a promotion criterion

- **Win rate.** Not at any sample size reachable in one NFL season. 272 games
  gives ±3 points; break-even sits inside every plausible interval.
- **Units won / P&L.** The MLB moneyline record was +1.06 SE from zero after
  366 bets, and the runline showed +160u while actually being −128u.
- **Backtested performance.** The +10.2u / 53.9% that justified enabling totals
  is precisely the evidence this gate exists to override.
- **A good week.** Or a good month.

P&L may be *reported*. It may not be *cited* as a reason to promote.

---

## 6. Season plan

| Phase | When | Action |
|---|---|---|
| Shadow | Week 1 → season end | Scheduler on, all markets OFF, store every candidate |
| First read | ~Week 6 | Coverage + calibration only. **No promotion possible** |
| Mid read | ~Week 12 | Stages 1-3 if sample allows |
| Decision | Post-season | Full gate, or an explicit "no edge found" |

Enabling `nfl_scheduler_enabled` is **not** a promotion. It starts collection.
Markets stay off regardless of what the numbers do mid-season.

---

## 7. Amendments

*None. Any entry here must record date, reason, and which window it applies to
— and must predate that window opening.*
