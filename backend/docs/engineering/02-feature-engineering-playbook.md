# Feature Engineering Playbook

**Purpose:** how a feature gets proposed, validated, versioned, served, and retired.
**Last verified against code:** 2026-07-31

---

## 1. Why this document exists

Three real failures motivated it:

1. **`wind_factor` is dead code.** Weather is fetched every two hours. `wind_speed` and `wind_direction` are written to `mlb_game_context`. `calculate_wind_factor()` exists in `services/mlb/weather_api.py`. **It is never called from anywhere.** The field is always `None`. Nobody noticed because nothing fails when a feature is silently absent.

2. **`temperature`, `is_dome`, and `last_10_win_pct` are computed and discarded.** They populate `MLBGameFeatures` and appear in `get_feature_names()`, but are absent from `MODEL_FEATURE_NAMES`, so they never reach training or serving.

3. **The V2 feature contract is a live trap.** `retrain_mlb_v2.py` trains on `V2_FEATURE_NAMES` (up to 32 features). `MLBScorer._build_model_feature_vector()` hardcodes **28 values in fixed positional order**. Deploying a v2 model without updating the vector builder produces either a shape error or — worse — silently misaligned features.

Every one of these is a *contract* failure, not a modelling failure.

---

## 2. The feature contract

A model artifact and the code that serves it agree on **an ordered list of feature names**. That agreement is the contract.

Today it is enforced by convention only:

```python
# services/mlb/scorer.py
MODEL_FEATURE_NAMES = [ ...28 names... ]
V2_FEATURE_NAMES    = MODEL_FEATURE_NAMES + [ ...4 first-inning names... ]

def _build_model_feature_vector(self, features) -> np.ndarray:
    vector = [ ...28 positional expressions with inline defaults... ]
```

**Required change (do this before the next retrain):** build the vector *from* the artifact's own `feature_cols`, not from a hardcoded list.

```python
def _build_model_feature_vector(self, features, feature_cols):
    lookup = features.to_feature_dict()          # name -> value, with defaults
    missing = [c for c in feature_cols if c not in lookup]
    if missing:
        raise FeatureContractError(f"model expects unknown features: {missing}")
    return np.array([[lookup[c] for c in feature_cols]])
```

This makes a contract break loud at load time instead of silent at inference time.

### Rules

1. Every artifact stores `feature_cols`, `trained_at`, `version`, `metrics`. (Already true — keep it.)
2. Serving builds the vector **by name from `feature_cols`**, never by position.
3. A feature name is permanent once trained. Renaming = new feature + retire old.
4. Defaults for missing values live in **one** place and are documented per feature.
5. `get_feature_names()` and the training feature list must be **the same list**, or the difference must be explicit and commented.

---

## 3. Feature lifecycle

```
PROPOSED -> SPIKED -> SHADOW -> SERVING -> DEPRECATED -> RETIRED
```

### PROPOSED
Written down before code. One paragraph:
- What real-world effect is it capturing?
- Why would the market not have already priced it?
- What is the cheapest way to be wrong quickly?

That second question is the filter. Season-long OPS is fully priced into every closing line; adding a variant of it cannot generate edge. Features worth building are ones the market prices *lazily* — bullpen fatigue, umpire assignment, wind at specific parks, lineup quality on rest days.

### SPIKED
A time-boxed research spike, **not wired into production**. Output is a written result with a number, and an explicit GO / NO-GO.

The QB-adjustment spike (`docs/superpowers/specs/2026-07-20-nfl-qb-adjustment-design.md`) is the reference example of this done right:
- built `qb_ratings.py` as standalone infra
- measured: spread ATS **49.1% with** the feature vs **50.2% without**, and worse still on the 26.3% subset where the feature was non-zero
- verdict **NO-GO**
- **wiring reverted so retrains cannot silently bake in the rejected feature**; the reusable module was kept

That last step is the discipline that matters. A rejected feature left half-wired becomes a landmine.

### SHADOW
Feature is computed and stored in production but excluded from the decision. Compare shadow vs serving predictions on live data before promoting.

### SERVING
In `feature_cols` of the deployed artifact. Covered by a contract test.

### DEPRECATED / RETIRED
Still computed but excluded from new training; then removal of the computation. **Retiring a feature requires deleting its computation**, otherwise you get another `wind_factor` — code that looks alive and does nothing.

---

## 4. Validation checklist

Before any feature reaches SERVING:

- [ ] **Leakage check.** Is every input knowable at snapshot time (T-60min)? Post-game stats, final lineups posted after first pitch, and any season aggregate that includes the game being predicted are leaks.
- [ ] **Coverage.** What fraction of games have a real (non-default) value? A feature defaulting on 40% of rows is mostly noise. Record the number.
- [ ] **Default sanity.** Is the default a neutral value or does it bias the prediction?
- [ ] **Backfill availability.** Can it be reconstructed for the training window? If not, it cannot be trained on.
- [ ] **Marginal contribution.** Retrain with and without. Report holdout RMSE/MAE both ways. A feature that does not move the holdout does not ship.
- [ ] **Gate impact.** How many picks change? Direction of P&L on the backtest? See `04-research-experimentation.md` for why this number is weak evidence on its own.
- [ ] **Contract test.** A unit test asserting the artifact's `feature_cols` matches what the serving code builds.

---

## 5. Feature inventory — MLB (as served)

The live model `mlb_run_diff_v1.joblib` uses exactly these 28, in this order:

```
home_runs_per_game    away_runs_per_game
home_ops              away_ops
home_avg              away_avg
home_obp              away_obp
home_slg              away_slg
home_era              away_era
home_whip             away_whip
home_starter_era      away_starter_era
home_starter_whip     away_starter_whip
home_starter_k9       away_starter_k9
home_starter_bb9      away_starter_bb9
home_starter_ip       away_starter_ip
park_factor
offense_diff          starter_era_diff      team_era_diff
```

**Computed but not served:** `temperature`, `is_dome`, `wind_factor` (always None), `home/away_last_10_win_pct`, `home/away_first_inning_*`.

---

## 6. Known feature gaps, ranked

Ranked by expected value per unit of work, informed by the calibration audit (2026-07-30).

| # | Feature | Why it matters | Effort |
|---|---|---|---|
| 1 | **Wire up what already exists** — `temperature`, `is_dome`, `wind_factor`, `last_10` | Data is already in the database. `calculate_wind_factor()` is already written. | Low |
| 2 | **Process stats over outcome stats** — FIP / xERA / SIERA instead of ERA; xwOBA instead of AVG | ERA and AVG carry large luck components; process metrics predict forward substantially better. Highest predictive return per hour. | Medium |
| 3 | **Bullpen** — trailing reliever quality, 3-day usage, closer availability | Relievers throw ~40% of modern innings. The model has **zero** visibility. Largest single gap. | High |
| 4 | **Lineup quality** — actual posted lineup, handedness matchup, rest | Team season OPS is used whether or not the three best hitters are resting. | High (feed dependency) |
| 5 | **Umpire** — plate umpire zone tendency | Well-documented total-runs effect. Totals-specific. | Medium |
| 6 | **Defense** — DRS / OAA | Run prevention is pitching *and* fielding; only pitching is modelled. | Medium |
| 7 | **Rest / travel / series position** | Cheap, weak, but nearly free. | Low |

Items 1, 3, and 5 are the ones with a plausible story for *why the market underprices them*. Items 2 and 6 mainly improve the point estimate.

---

## 7. Anti-patterns

**Silent defaults.** `features.home_ops or 0.720` cannot distinguish "average team" from "stats failed to load." Coverage must be measured, not assumed.

**Positional feature vectors.** See §2.

**Collect-then-forget.** Data ingested but never wired in — `wind_factor` is the canonical case. If a feature is not in `feature_cols` within one cycle of being ingested, either wire it or stop ingesting it.

**Season aggregates as recency.** `home_ops` is season-long. A team hot for three weeks looks identical to one that was hot in April.

**Feature added to fix a P&L problem.** Features fix *prediction* problems. The totals bias patch is the cautionary case: correcting a real, measured −0.73 run bias made P&L **worse** (−11.22u vs −8.04u) because it only admitted more coin-flips through the gate. Diagnose which layer is broken before adding inputs.
