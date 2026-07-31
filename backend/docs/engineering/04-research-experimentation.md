# Research & Experimentation Framework

**Purpose:** how a hypothesis becomes evidence, and how evidence becomes a production change.
**Last verified against code:** 2026-07-31

---

## 1. The number that governs everything here

```
graded moneyline bets : 352
record                : 160-192  (45.5%)
net                   : +26.4u  (+0.075/bet)
avg odds              : 2.40  (breakeven 41.6%)
standard error        : ±22.5u
result                : +1.18 SE from zero
```

**A zero-edge strategy betting these prices returns +26u or better roughly one season in eight.** The result is encouraging and directionally right — the realized win rate does beat the implied breakeven — but it is **not statistically distinguishable from luck.**

Two consequences that shape every rule below:

1. **P&L over a few hundred bets is weak evidence.** Any decision justified by "it's up 26 units" is resting on a coin that has not landed.
2. **This configuration is a survivor.** Market multipliers, gate thresholds, edge floors, k values, a QB feature, and a totals bias patch have all been tested against overlapping data. Keeping the winner of many comparisons biases the winner upward. The true edge is likely *below* the point estimate.

---

## 2. Experiment types

| Type | Question | Evidence strength |
|---|---|---|
| **Calibration study** | Do stated probabilities match observed frequency? | **Strongest.** Needs no P&L, uses every game, converges fastest |
| **Holdout model comparison** | Does model B predict better than A? | Strong on prediction; says nothing about profit |
| **Backtest** | Would this have made money? | **Weak.** Small samples, survivorship, limited candidate reconstruction |
| **Shadow tracking** | What would this do on live data? | Strong but slow |
| **Live A/B** | — | Not available; no traffic-splitting infrastructure |

**Prefer calibration studies.** The k recalibration is the model case: it used 1,463 games (not 352 bets), measured a distributional property (margin SD 4.64 vs implied 3.63), and produced a constant derived independently of profit. P&L was then checked as *corroboration*, not as the fitting objective.

That ordering is the whole method. Fit to the distribution, verify against money.

---

## 3. Known backtest limitations — read before trusting any backtest

**`mlb_markets` is overwritten in place.** Historical candidate sets are gone. A backtest can only re-price *picks that were actually made*; it cannot ask "what else would the model have chosen?" This structurally prevents testing any change to **selection** — including whether ranking by `raw_edge` instead of `edge_pct` would fix the 332-dog / 19-favorite skew.

**Pre-2026-07-22 runline lines are sign-corrupted.** No runline backtest before that date is valid.

**Snapshot odds are not fill prices.** Best-of-11-books on a dog moneyline is often the slowest book, and those prices get limited. Backtests assume fills that may not be available.

**Grading before 2026-07-28 used copied components.** 109 snapshots were stranded ungradeable by the `best_ml` overwrite bug. Records spanning that boundary are not comparable.

If a backtest cannot answer the question, **say so and stop** rather than producing a number with hidden caveats. Declaring an experiment unmeasurable is a valid, valuable result.

---

## 4. The experiment protocol

### Pre-registration (before looking at outcome data)

```
HYPOTHESIS:      specific and falsifiable
MECHANISM:       why would this work? why hasn't the market priced it?
PREDICTION:      what number, in which direction, by how much
DATA:            exact source and window, fixed now
METRIC:          primary, chosen before the run
THRESHOLD:       what result counts as success
KILL CRITERION:  what result ends this line of work
CONFOUNDS:       what else could produce this result
```

Written to `docs/superpowers/specs/YYYY-MM-DD-<name>.md` **before** execution.

### Execution
Run once. Record the result whatever it is. Re-running with tweaked parameters starts a **new** experiment and inflates the multiple-comparisons problem — note it explicitly.

### Verdict
**GO / NO-GO / INCONCLUSIVE**, written into the spec.

NO-GO requires **reverting the wiring**, not just declining to enable it. The QB spike is the reference: `qb_delta` made spread ATS worse (49.1% vs 50.2%), and worse still on the 26.3% subset where it was non-zero. Verdict NO-GO; wiring reverted so future retrains cannot silently bake in the rejected feature; `qb_ratings.py` retained as reusable infrastructure.

INCONCLUSIVE is a real verdict. Record it and state what sample would resolve it.

---

## 5. Closing-line value — the missing instrument

**This is the highest-value item in this document.**

Win rate needs thousands of bets to resolve. CLV needs dozens. If a bet is placed at 2.09 and the market closes 2.20, value was lost regardless of the result; if it closes 1.95, the market moved toward the pick and the outcome is noise.

Beating the close is the only fast, honest signal that the model knows something the market did not. Over a few hundred bets, CLV is a far better estimator of true edge than realized P&L — it has a fraction of the variance because it strips out game-outcome randomness entirely.

### Required implementation

1. New append-only table `closing_lines` — `(game_id, market_type, side, closing_odds, captured_at)`.
2. Capture at first pitch / kickoff (the odds ingest already runs every 30m; add a final capture keyed to game start).
3. Store snapshot-vs-close delta per pick.
4. Report rolling CLV alongside win rate on the evaluation pages.
5. **Alert on sustained negative CLV** — that is the trigger worth acting on, far more than a quiet streak with no picks.

Until this exists, every model comparison is limited to slow, noisy P&L evidence.

---

## 6. Statistical standards

**Report a standard error with every P&L claim.** For flat-stake betting at decimal odds *d* with win rate *p*:

```
SE_per_bet   = sqrt(p*(1-p)) * d
SE_total     = SE_per_bet * sqrt(n)
```

**Rules of thumb given current volume (~3 picks/night):**

| Claim | Bets needed |
|---|---|
| Distinguish a 2% edge from zero | thousands — not reachable in one season |
| Distinguish a 5% edge from zero | ~1,000 |
| Detect a calibration error | ~200 (uses all games, not just bets) |
| Detect CLV edge | ~50–100 |

**This is why calibration and CLV are the primary instruments and P&L is the lagging confirmation.**

Additional requirements:
- **Report every comparison run**, not just the winner. A note like "tested 4 market multipliers, kept the best" belongs in the spec.
- **Multiple-comparisons discipline:** more variants tested on one dataset means the winner's true effect is smaller than measured.
- **Never retune on the same window used to discover an effect.**
- **Hold a configuration still.** Each change resets the clock on what can be concluded. ~350 bets across several system changes is not 350 bets of evidence for the current system.

---

## 7. Worked examples from this repo

**Good — k recalibration (GO).** Distributional measurement on 1,463 games, derivation independent of P&L, pinned by a test, deployed with the derivation in a comment. Backtest agreement (+38.90u vs +32.87u) treated as corroboration, with the honest caveat that +6u over 351 picks is within noise.

**Good — totals bias patch (NO-GO).** A real, measured −0.73 run bias. Correcting it made P&L **worse** (−11.22u vs −8.04u) because it admitted more coin-flips through the gate (161 → 292 picks). Rejected on evidence despite a correct-looking premise. Lesson: fixing a real defect at the wrong layer can hurt.

**Good — totals re-entry gate (NO-GO).** Threshold pre-registered in `config.py`; 125 picks at 52.1% / −0.91u; decision mechanical.

**Good — QB spike (NO-GO).** Time-boxed, wiring reverted, infrastructure kept.

**Cautionary — runline record.** A sign-pairing bug made a losing market look profitable for months. Corrected: **−128u.** Lesson: validate that the thing being measured is the thing being bet, before trusting any record.

**Cautionary — "scorer quirk."** The `best_ml` per-row overwrite was logged as "pre-existing, not a retune bug" and left alone. It was in fact stranding 109 snapshots as permanently ungradeable. Lesson: a known-but-unexplained anomaly is an open defect, not a curiosity.

---

## 8. Current research queue

| Priority | Question | Method | Blocked by |
|---|---|---|---|
| 1 | Are we beating the closing line? | CLV instrumentation | Needs §5 built |
| 2 | Does a retrained run-diff model beat the Feb-9 one? | Holdout + shadow | Retrain not run |
| 3 | Do wired-up weather/recency features improve the holdout? | Retrain with/without | Feature wiring |
| 4 | Do process stats (FIP/xERA) beat outcome stats? | Holdout comparison | Data source |
| 5 | Is the underdog skew signal or artifact of `edge_pct`? | Forward shadow only | **Unmeasurable retroactively** — `mlb_markets` overwritten |
| 6 | Can an empirical cover curve make runline viable? | Fit P(margin≥2), forward-validate | Needs clean forward data |
| 7 | Can totals be made viable with real features? | Full rebuild | Depends on 3, 4 |

Item 5 is a standing argument for retaining candidate sets going forward: an append-only odds table would make selection changes testable for the first time.
