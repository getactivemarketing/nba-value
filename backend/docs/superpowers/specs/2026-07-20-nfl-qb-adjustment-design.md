# NFL QB-Adjustment (P2.5a) — Design Spec

**Date:** 2026-07-20
**Status:** Approved (brainstorming) → ready for implementation plan
**Depends on:** NFL Phases 1–4 (merged to main @ `8ccef77`). See [[truline-nfl]].

## Problem

The NFL margin-of-victory (MOV) model drives spread + moneyline, but on walk-forward
backtest it hits only **50.2% ATS** — below the **~52.4%** break-even for -110 vig — so
spread/ML ship as SHADOW (recorded, not bet). Diagnosis (Phase 2): NFL spreads are
efficient (model corr 0.31 vs market 0.46; market line RMSE 12.8 < model 12.5). The
model's team-EPA features price roughly what the market already prices.

The most plausible source of *un-priced* spread edge is **QB changes**. Our trailing-8-game
team-EPA features encode whoever played those games. When the QB **projected to start this
game differs** from the QB who generated that trailing form (injury, benching, a starter
returning), the team features mis-state the team's true strength for this game — and the
market can be slow to fully price a backup. A signal that captures this mismatch is the
lever most likely to move spread toward break-even.

## Goal & scope

**This build (P2.5a) is a research spike with a single decision output:** does a QB-change
feature push spread through the real gate to break-even? Build the feature, retrain, backtest
through the gate on historical **actual** starters.

- **GO** (spread clears ~52.4% ATS with positive units at meaningful sample, 2019–24) →
  proceed to **P2.5b**: live starter-projection feed (nflverse injury/depth at snapshot time)
  + flip `nfl_spread_in_best_bet=True`.
- **NO-GO** → keep `nfl_mov_v1`, spread stays SHADOW, record the negative result
  ("QB-EPA delta insufficient for spread edge").

**Out of scope for P2.5a** (deferred to P2.5b, only if GO): the live starter-projection feed,
any scheduler/live-scoring change, flipping the spread gate flag, any new DB table.

**No leakage of risk to the live product:** `qb_delta` is added to `MOV_FEATURES` **only**.
The totals model (`TOTALS_FEATURES`, the live-gated product) is untouched — worst case is
"qb_delta didn't help, keep v1."

## Design

### Decisions (locked in brainstorming)
1. **Mechanism:** `qb_delta` as a learned **feature** in the MOV model (not a hand-tuned
   post-model point adjustment). The model learns its weight; validated by the same
   walk-forward go/no-go used for totals.
2. **QB rating:** rolling **career-to-date dropback EPA/play**, empirical-Bayes **shrunk toward
   a replacement-level prior** by dropback count. Point-in-time (through the prior week).
3. **Signal shape:** a **delta** (projected-starter quality − trailing-form-QB quality), which
   is ~0 for the stable-QB majority and moves only on a real QB change.

### Component 1 — `services/nfl/qb_ratings.py` (new, pure)

- `qb_game_dropback_epa(pbp) -> DataFrame`: aggregate the pbp we already load into per
  `(passer_player_id, season, week, team)` rows carrying `dropbacks` and `epa_sum`.
  "Dropback" = nflverse `qb_dropback == 1` (pass attempts + sacks + scrambles); EPA from
  nflverse `qb_epa`. **Implementation Task 1 must verify these exact nflverse pbp column
  names** (`passer_player_id`, `qb_dropback`, `qb_epa`) against a loaded pbp frame before
  building on them (Phase-1 confirmed pbp column stability, but these QB-level columns weren't
  used before).
- `shrink(epa_sum, dropbacks, R, K) -> float`: `(epa_sum + K*R) / (dropbacks + K)`. Pure.
- `rolling_qb_rating(qb_game_epa, R, K) -> DataFrame`: for each `(passer_player_id, season,
  week)`, the shrunk rating computed from that QB's **cumulative career-to-date** dropbacks/EPA
  over all games **strictly before** the target game (crosses seasons — career, not
  season-to-date). Returns a lookup keyed `(passer_player_id, season, week)`.
  - Constants (module-level, sane priors, NOT an overfit grid): `R = -0.10` EPA/play
    (replacement level), `K = 200` dropbacks (prior strength). A **small sane-range check**
    (e.g. R ∈ {−0.12,−0.10,−0.06}, K ∈ {150,200,250}) picked on the walk-forward result is
    allowed; no fine grid.

### Component 2 — feature integration in `training_data.py`

Per game, per team, entering week `w` of season `s`:
- `starter_rating` = `rolling_qb_rating[(nfl_games.{home,away}_qb_id, s, w)]` — the announced
  starter's own shrunk rating entering the game. (Missing/unknown QB id → replacement `R`.)
- `form_rating` = the **dropback-weighted mean QB rating embedded in the team's trailing-8-game
  window** (the QB quality the `off_epa` features already reflect): over the team's games in
  the trailing window (same window as `rolling_team_stats`, through `w-1`), weight each passer's
  `rolling_qb_rating` (entering that game) by his dropbacks in the window, and average. (No
  trailing dropbacks → replacement `R`.)
- `qb_delta = (home_starter_rating − home_form_rating) − (away_starter_rating − away_form_rating)`

Add `qb_delta` to `MOV_FEATURES` (append; keep `TOTALS_FEATURES` unchanged). The
`build_feature_frame` diff/sum path is untouched; `qb_delta` is computed alongside and joined
by `game_id`. `build_feature_frame` must remain leakage-free (all ratings point-in-time).

**Why delta, not raw starter rating:** the team-EPA features already carry the form QB's
quality; adding the *raw* starter rating would double-count for stable QBs. The delta isolates
the *change*, sitting at ~0 when `starter == form QB`.

### Component 3 — retrain + gate (`tasks/nfl_train.py`, `backtest.py`)

- `nfl_train.py` already trains from `MOV_FEATURES`, so it picks up `qb_delta` automatically →
  candidate **`nfl_mov_v2.joblib`** (do NOT overwrite v1; write v2 alongside for comparison).
- Run the existing walk-forward backtest (2019–24) **through the real gate** (mirroring the
  Phase-2/3 spread evaluation). Report, side by side vs v1: spread ATS%, units, sample size,
  reliability, and specifically the subset of games with a **non-zero `qb_delta`** (the games
  the feature is supposed to help) — the headline metric is spread ATS/units on those.

### Point-in-time / leakage

- QB ratings use only pbp **strictly before** the target game (career-to-date through `w-1`).
- Starter identity (`{home,away}_qb_id`) is a legitimate pre-game known (starters announced
  pre-kick) — not leakage.
- `form_rating` uses the through-`w-1` trailing window.
- Walk-forward: train on seasons `< S`. Same discipline as Phases 1–3.

## Testing

Pure unit tests (no DB/network):
- `shrink`: few dropbacks → near `R`; many → near true career EPA; monotonic in `K`.
- `rolling_qb_rating`: point-in-time (a future game never affects an earlier-week rating);
  career crosses seasons; unknown QB → `R`.
- `qb_delta == 0` when the starter is the sole trailing-form QB (stable case).
- `qb_delta` sign: a clear upgrade (backup form → star starter) → positive for that team;
  downgrade → negative; home-vs-away orientation correct.
- `build_feature_frame` gains a `qb_delta` column and its Phase-2 tests still pass (totals
  columns unchanged).

Empirical gate (not a unit test): the retrain + walk-forward backtest is the GO/NO-GO.

## Deliverable

- `services/nfl/qb_ratings.py` + tests; `qb_delta` in `training_data.MOV_FEATURES` + tests;
  `nfl_mov_v2.joblib` candidate committed alongside v1; a backtest report with the spread
  GO/NO-GO recommendation.
- **No** DB table, **no** live/scheduler change, **no** gate-flag flip in this build.

## Risks / notes

- **Modest event count:** real QB changes are a minority of games, so `qb_delta` is inert
  most weeks (delta ≈ 0). That bounds both upside and downside — it can't hurt the stable
  majority, but the sample of games it *can* help is small, so the backtest must look at the
  non-zero-delta subset explicitly, not just the aggregate.
- **Dropback definition:** must be consistent (nflverse `qb_dropback`); sacks/scrambles
  included so mobile-QB value isn't dropped.
- **Mid-game QB injuries** (starter knocked out): the game's "starter" is the announced
  `{home,away}_qb`; trailing form uses actual dropbacks. Acceptable for v1.
- **Constant sensitivity:** `R`/`K` are priors; keep the sane-range check coarse to avoid
  overfitting the 2019–24 window.
- **Totals untouched** → the live product carries no risk from this spike.
