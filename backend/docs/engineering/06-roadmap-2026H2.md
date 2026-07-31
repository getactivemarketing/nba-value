# Roadmap — 2026 H2 (Aug 2026 → Jan 2027)

**Status:** Proposed. Supersedes ad-hoc prioritisation.
**Written:** 2026-07-31
**Companion docs:** `01`–`05` in this directory.

---

## 1. The thesis

TruLine v1 was a software problem and it is solved: three verticals, a scheduler, immutable snapshots, honest grading, evidence-based market gating. TruLine v2 is a quantitative research problem, and it is not solved.

The reframe that organises this roadmap:

> **Stop asking "how do I predict games?" Start asking "how do I detect when the market is wrong?"**

These are different problems with different difficulty. The market has already priced everything publicly knowable. Our measured position is consistent with that: the live run-diff model has RMSE 4.431 against a margin SD of 4.64 — **R² ≈ 0.09** — and the moneyline record sits **+1.18 standard errors from zero** (352 bets, 45.5%, +26.4u, ±22.5u SE).

That is not a failing model. It is close to the realistic ceiling for predicting single baseball games. It means edge cannot come from better prediction of *reality*. It has to come from measuring **reality minus market expectation**, which requires information about the market itself — information we currently throw away.

---

## 2. The organising principle

The useful axis is **not** infrastructure vs. modelling. It is:

| | Definition | Examples |
|---|---|---|
| **Information-acquiring** | Adds signal the system did not previously have | Market history, bullpen data, umpire assignment, lineups, weather wiring |
| **Information-reorganising** | Restructures signal already present | Ensembles, feature stores, explainability, automated search |

**We are information-starved, not architecture-starved.** Acquisition wins for at least the next quarter. Reorganisation work is scheduled only after the information it would reorganise exists.

This is the single most common way roadmaps like this go wrong: an ensemble of five models trained on 28 season-aggregate features redistributes the same 9% of variance five ways.

---

## 3. Calendar constraints — these drive the sequencing

| Date | Event | Consequence |
|---|---|---|
| **~2026-09-27** | MLB regular season ends | **Market-history capture must ship in August or the next MLB in-season window is April 2027** |
| **~2026-09-10** | NFL season opens | NFL go-live (scheduler flip + first-week validation) is calendar-locked |
| **~2026-10-21** | NBA season opens | NBA becomes the live testbed over winter |
| Oct 2026 – Mar 2027 | MLB offseason | Model rebuilds, backfills, distributional work — no live MLB feedback |

**The binding constraint is Phase 1.** Market-dynamics features (dispersion, velocity, steam, hold movement) require *months of stored history* before they can be computed at all. Every week without capture is a week that can never be recovered. Nothing else on this roadmap has that property.

---

## 4. Phases

### Phase 1 — Instrument the market (August)

**Objective:** never lose another day of market history, and acquire the ability to evaluate work in weeks instead of seasons.

The infrastructure is already half-built. `odds_snapshots` exists with the right schema:

```
79,453 rows | 644 games | 11 books | 2026-01-05 → 2026-06-14
book_key, minutes_to_tip, home_ml_odds, away_ml_odds, home_spread,
total_line, over/under_odds, home_ml_prob, over_prob, is_closing_line

is_closing_line = true:  0 rows        <- never once set
MLB equivalent:          does not exist
```

**Correction (2026-07-31):** this capture did not fail — the last NBA game was
2026-06-13, so it stopped at the season boundary and the NBA pipeline is
healthy. The real gap is narrower and worse: **MLB and NFL have no odds
history at all**, and `is_closing_line` has never been set true in any
vertical. Phase 1 is therefore *porting a working NBA capability to MLB*, not
repairing a broken one.

**Work**
1. ~~Build MLB odds history~~ **DONE 2026-07-31.** `mlb_odds_snapshots`,
   append-on-change, wired into `ingest_odds`. Live-verified: 14 games × 11
   books captured, 108 unchanged quotes suppressed, genuine moves recorded.
2. ~~Closing-line marking~~ **DONE 2026-07-31.** `mark_closing_lines` task
   selects the last quote at or before first pitch per (game, book) and sets
   `is_closing_line`. Scheduled every 30m; idempotent.
3. Compute and store CLV per pick: snapshot price vs closing price.
4. Surface rolling CLV on the evaluation pages beside win rate.
5. Alert on sustained negative CLV.
6. Extend the same capture to NFL before the September opener.
7. **Retrain the MLB run-diff model** (last trained 2026-02-09) and fix the feature-vector contract in the same commit — see `02` §2, `03` §9.
8. Backfill NBA `odds_snapshots` closing-line marking when the season resumes.

**Exit gate**
```
SAMPLE:    >=30 days of continuous multi-book capture, all three verticals
METRIC:    CLV computable for >=95% of graded picks
GUARDRAIL: zero gaps >6h in capture
ROLLBACK:  capture is additive; nothing downstream depends on it yet
```

**Why first:** cheapest item on the roadmap, it is the only one the calendar forces, and every subsequent phase is judged by the instrument it produces. Without CLV, every later decision waits on P&L evidence that takes seasons to resolve.

---

### Phase 2 — Acquire missing feature categories (September–October)

**Objective:** close the categorical gaps. Not more features — *different kinds* of features.

The model currently sees season-long team rates, starter rates, and park factor. It does not see:

| Category | Status | Why it matters |
|---|---|---|
| **Bullpen** | **Zero coverage** | ~40% of modern innings. Largest single gap |
| **Lineup** | Zero coverage | Season OPS used whether or not the best hitters rest |
| **Umpire** | Zero coverage | Well-documented total-runs effect |
| **Weather** | **Collected, discarded** | `calculate_wind_factor()` exists and is never called |
| **Recency** | Computed, discarded | `last_10_win_pct` never reaches the model |
| **Process stats** | Not used | FIP/xERA/SIERA predict forward better than ERA |

**Work**
1. Wire up what already exists — `wind_factor`, `temperature`, `is_dome`, `last_10`. Lowest effort on the roadmap.
2. Bullpen features: trailing reliever quality, 3-day usage, closer availability.
3. Lineup features: posted lineup quality, handedness matchup, rest.
4. Umpire assignment and zone tendency.
5. Swap outcome stats for process stats on starters.
6. **Point-in-time correctness enforced from the first commit** — every feature reproducible as of snapshot time. This is the genuinely valuable half of "feature store."

**Target: ~28 → 70–90 features across these categories.** Deliberately *not* 200–500. At ~2,430 MLB games per season, 500 mostly-noise columns invite overfitting that `feature_fraction` does not save you from. The gap is categorical coverage, not column count.

**Exit gate**
```
SAMPLE:    full retrain on backfilled history
METRIC:    holdout MAE beats the Phase 1 retrained incumbent
GUARDRAIL: no feature with >20% default rate ships (see 02 §4)
GUARDRAIL: leakage audit passed — every input knowable at T-60min
ROLLBACK:  config path back to the Phase 1 artifact
```

**Risk:** lineup and umpire data need a feed we do not have. If acquisition stalls, ship bullpen + weather + process stats and defer the rest — do not let a blocked sub-item hold the phase.

---

### Phase 3 — Distributional outputs (November–December)

**Objective:** stop predicting means. Predict distributions.

This is scheduled third not because it is least valuable but because it **fixes a defect we have already measured**, which makes it the least speculative modelling change on the list.

The runline cover curve derives P(cover) by shifting the win curve by the spread — which assumes margins are logistic with fixed scale. They are not: **27.2% of MLB games are decided by exactly one run**, and the ±1.5 line sits on that spike. Empirical minus modelled P(margin ≥ 2), by predicted-run-diff bucket: **+0.039 / +0.059 / −0.043 — the sign flips.** Betting the disagreement systematically takes the wrong side. Corrected runline history: **−128u.**

A half-inning Monte Carlo produces the discrete, lumpy margin distribution baseball actually has. Runline, alternate lines, and totals prices then fall out of the same simulation instead of three separately hand-tuned logistics.

**Work**
1. Half-inning simulation producing full joint distribution of (home runs, away runs).
2. Derive moneyline, runline, totals, and alternate lines from simulated distributions.
3. Retire `_run_diff_to_win_prob`, `_run_diff_to_cover_prob`, `_total_to_over_prob` — three hand-fitted logistics replaced by one generative model.
4. Re-derive the runline empirically; re-evaluate the runline pause against its own gate.
5. Re-test totals with Phase 2 features *and* a real distribution — the first honest test totals has had.

**Exit gate**
```
SAMPLE:    full-season backtest + >=100 forward shadow picks
METRIC:    simulated probabilities calibrate better than the logistics they replace
           (reliability curve on all games, not just bets)
GUARDRAIL: moneyline P&L not worse than incumbent
ROLLBACK:  logistic path retained behind a config flag for one full cycle
```

**Note:** runline and totals stay paused through this phase regardless of backtest results. Both were previously un-paused on evidence that later proved wrong; re-entry requires forward validation per `03` §4.

---

### Phase 4 — Market dynamics as features (December–January)

**Objective:** the moat. Model the market, not just its prices.

Only buildable once Phase 1 has accumulated months of history — which is exactly why Phase 1 leads.

**Features to engineer from `odds_snapshots`**
- Dispersion: cross-book disagreement at a point in time
- Velocity: rate and direction of line movement
- Sharp-vs-public divergence: movement at sharp books while soft books stay stale
- Hold: vig expansion/contraction as a confidence signal
- Steam: correlated rapid movement across books
- Staleness: books lagging consensus (this is where a beatable price actually lives)
- Consensus probability vs our model's probability, as an explicit feature

**Work**
1. `MarketIntelligence` service computing these continuously.
2. Store as first-class, point-in-time-correct features.
3. Retrain with market-dynamics features included.
4. **Add MLB market regression** — NBA blends 70% market into the prediction; MLB blends 0%. Cheapest single fix on the roadmap and it belongs in this phase conceptually.

**Exit gate**
```
SAMPLE:    >=90 days of market history; full retrain
METRIC:    CLV improves vs the Phase 2 model  <- the real test, available by now
GUARDRAIL: no degradation in calibration
ROLLBACK:  config path to prior artifact
```

**Why this is the moat:** everyone models teams. Almost nobody models book behaviour. And unlike team quality, market microstructure is *not* fully priced into the closing line — it partially constitutes it.

---

### Phase 5 — Ensemble (January, conditional)

**Objective:** blend distinct signal families once they exist.

**Explicitly gated on Phases 2 and 4.** An ensemble today would train five weak models on one narrow feature set. After Phase 2 and 4 there are genuinely distinct families — statistical, situational, matchup, market, environmental — and blending becomes meaningful.

**Preconditions (all required)**
- [ ] ≥3 signal families with independent, non-trivial holdout contribution
- [ ] Enough training data to learn blend weights without overfitting
- [ ] Per-member calibration and CLV tracked independently
- [ ] Every component prediction stored for explainability and retraining

If the preconditions are unmet in January, **this phase does not run.** That is an acceptable outcome, not a failure.

---

### Phase 6 — Explainability and automated research (January+)

Both are reorganisation work and both are gated on earlier phases.

**Explainability** converts feature attribution into bettor-facing reasoning. It cannot ship before Phase 2, because the factors worth naming — bullpen fatigue, wind, handedness edge — are precisely the ones the model cannot currently see. Generating fluent explanations from a 28-feature season-aggregate model would manufacture authority the system has not earned.

**Automated research** — the highest-risk item on this roadmap, and it ships **only** with statistical controls:

> "Run 500 experiments, keep the best" is a false-discovery engine. At α = 0.05, 500 experiments produce ~25 significant results by construction. Our entire measured edge is +1.18 SE. Automating *search* without automating *correction* would systematically promote noise into production.

**Mandatory if built**
- Nested cross-validation — selection never touches the evaluation fold
- False-discovery-rate control (Benjamini–Hochberg or equivalent)
- Enforced pre-registration per `04` §4
- Every experiment logged, including failures — reporting only winners is the bias
- Automated *recommendations*, never automated *promotion*. A human runs the gate in `03` §4

---

## 5. Dependency graph

```
Phase 1 (market capture + CLV + retrain)
   │
   ├──────────────► Phase 4 (market dynamics)   [needs >=90d history]
   │                      │
Phase 2 (feature categories)                    │
   │                      │                     │
   ├──► Phase 3 (distributional)                │
   │                      │                     │
   └──────────┬───────────┴─────────────────────┘
              ▼
        Phase 5 (ensemble, conditional)
              │
              ▼
        Phase 6 (explainability, automated research)
```

Phase 1 blocks everything. Phases 2 and 3 can overlap. Phase 4 is time-gated, not work-gated — start the clock in August or it slips a quarter.

---

## 6. Running alongside

| Item | When | Note |
|---|---|---|
| **NFL go-live** | ~Sept 10 | Flip `nfl_scheduler_enabled`, validate one real week, wire the nightly tracker. `grade_finals` has never run against a real NFL final |
| Model-age and fallback monitoring | Phase 1 | Would have caught the Feb-9 model in March (`03` §7) |
| "No edge" vs "no data" alerting | Phase 1 | With only moneyline live, silence is now common and a broken feed hides in it |
| `/health` model version + `trained_at` | Phase 1 | |
| `DEFAULT_RUN_DIFF_MODEL` → config | Phase 1 | Rollback parity |

---

## 7. Explicitly deferred, with reasons

| Item | Why not now |
|---|---|
| 200–500 engineered features | Category coverage is the gap, not column count. ~2,430 games/season does not support it |
| Ensemble before Phase 2/4 | Would redistribute one narrow feature set five ways |
| Explainability before Phase 2 | Would name factors the model cannot see |
| Unconstrained automated search | False-discovery engine at our sample size |
| Player props | Needs Phase 3 distributions first |
| Un-pausing runline or totals | Both previously un-paused on evidence that proved wrong. Forward validation only |
| Splitting API from schedulers | Real coupling, not yet a real problem |

---

## 8. How this roadmap gets judged

Not by features shipped. By these, in order:

1. **Is CLV positive and stable?** The primary question. Answerable from ~Sept.
2. **Does calibration hold?** Predicted vs realized frequency on all games, not just bets.
3. **Is realized P&L consistent with measured CLV?** Lagging confirmation only.

If CLV is flat after Phases 1, 2, and 4, the honest conclusion is that the current approach does not beat this market — and that is worth knowing in six months rather than six seasons. **This roadmap is designed to fail fast and legibly**, which is the main thing the current architecture cannot do.

---

## 9. Open questions

1. **Data budget.** Bullpen, lineup, and umpire feeds may cost money. What is the ceiling?
2. **Sport focus.** Three verticals at one-person capacity is thin. Is MLB the priority, or is NFL's Sept go-live the bigger opportunity given totals showed +10.2u / 53.9% in backtest?
3. **Bet sizing.** Everything here assumes flat stakes. Fractional Kelly is out of scope but becomes relevant if CLV confirms real edge.
4. **Execution reality.** Best-of-11-books on a dog moneyline is often the slowest book, and those get limited. Are backtested prices actually available at size?
