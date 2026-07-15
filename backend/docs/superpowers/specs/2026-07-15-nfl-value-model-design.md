# NFL Value Betting Model

**Date:** 2026-07-15
**Status:** Approved
**Goal:** Build an NFL betting-value model as a self-contained vertical mirroring the existing NBA/MLB modules — a margin-of-victory model driving spread + moneyline, a totals model, value-based pick selection, graded snapshots, and a weekly scheduler — validated phase by phase before it produces any real betting output.

## Background

The `nba-value` backend already runs two model-driven betting verticals:

- **NBA** (original core): `services/data`, `services/ml` (`mov_model`, `spread_model_v2`, totals), `services/scoring`, `tasks/scheduler.py`, NBA-centric API routers.
- **MLB** (self-contained module added later): `services/mlb/` (ingest, features, scorer, value_calculator, model_training), `tasks/mlb_scheduler.py`, `api/mlb.py`, `mlb_*` DB tables, `models/mlb_*.joblib`, `mlb_*` config.

The MLB module is the clean template: one bounded package per sport, its own scheduler, its own API router, its own tables and model artifacts. NFL will be built the same way as a new `nfl` vertical, reusing shared infrastructure (The Odds API client, calibration, the value-math patterns, the nightly tracker) wherever it already exists.

The shared scoring philosophy carries over: a model produces a projection, a value calculator compares model probability against de-vigged market-implied probability to get an edge, and snapshots are written pre-game and graded post-game at flat $100 units.

### Scope decisions (locked during brainstorming)

- **Markets:** Spread + Moneyline + Totals. A single margin-of-victory (MOV) model drives both spread-cover and moneyline probabilities (coherent by construction); a separate totals model handles over/under. Totals are shadow-gated at launch (recorded but excluded from `best_bet`) exactly like MLB.
- **Data source:** nflverse via the `nfl_data_py` package (free, play-by-play/EPA/schedules back to 1999) for fundamentals; The Odds API (already integrated in `services/data/odds_api.py`, sport key `americanfootball_nfl`) for market odds.
- **Modeling:** Two LightGBM regressors (MOV, totals), consistent with the MLB models. Probabilities derived from residual distributions + calibration. Rejected alternatives: separate per-market binary classifiers (incoherent spread vs ML) and a pure Elo/power-rating model (thin edge — folded in as a feature instead).
- **Scope/sequencing:** One design doc; implementation decomposed into phases, data foundation first, each phase validated before the next. Social cards + frontend are explicitly a separate future project (P5).

### Known limitations (accepted, documented — not blockers)

- **Sample size.** NFL has ~272 regular-season games/season; 2010→2024 ≈ 4,000 games — far less than MLB's per-season volume. Consequence: feature engineering matters more than model complexity, the model is kept regularized to avoid overfit, and totals in particular start shadow-gated.
- **QB injuries.** A starting-QB change swings NFL lines 5-7 points, and nflverse injury data is messy. v1 does **not** include a mature QB-adjustment; it is a documented gap and a Phase-2.5 enhancement. Starting-QB identity is stored from Phase 1 so the adjustment can be built without re-ingesting.
- **Best-effort features.** `starters_out` (injury count) and `playoff_stakes` (meaningful-game heuristic) are lower-confidence: they are stored from Phase 1 but treated as *candidate* features in Phase 2 — tested for signal and benched if too noisy over 2010-2024, rather than assumed.

## Architecture

A new `nfl` vertical parallel to `mlb`:

| Layer | MLB (template) | NFL (this project) |
|---|---|---|
| DB models | `mlb_game`, `mlb_market`, `mlb_team`, `mlb_team_stats`, `mlb_prediction_snapshot`, … | `nfl_*` mirror |
| Service package | `services/mlb/` | `services/nfl/` |
| Scheduler | `tasks/mlb_scheduler.py` | `tasks/nfl_scheduler.py` |
| API router | `api/mlb.py` | `api/nfl.py` |
| ML artifacts | `models/mlb_*.joblib` | `models/nfl_*.joblib` |
| Config | `mlb_*` settings | `nfl_*` settings |

## Data model (`nfl_*` tables)

New SQLAlchemy models under `src/models/`, following the MLB table family.

**`nfl_teams`** — 32 teams. `id`, `abbr`, `name`, `conference`, `division`, `primary_color`, `secondary_color` (colors stored now for the later cards project).

**`nfl_games`** — schedule + results, one row per game.
- Identity/schedule: `id`, `season`, `week`, `season_type` (reg/post), `home_team`, `away_team`, `kickoff_utc`, `status`.
- Results: `home_score`, `away_score`.
- Venue: `roof` (dome/outdoor/retractable), `surface`, `neutral_site`.
- Situational flags (per user request): `home_qb`, `home_qb_id`, `away_qb`, `away_qb_id` (**clean** — from nflverse schedules); `is_divisional` (**clean** — derived from divisions); `is_primetime` + `primetime_slot` (TNF/SNF/MNF; **clean** — derived from kickoff day/time).

**`nfl_game_context`** — per-game situational detail.
- Rest/weather: `home_rest_days`, `away_rest_days`, `wind_mph`, `temp_f`, `is_dome`.
- Best-effort signals: `home_starters_out`, `away_starters_out` (int; from injury reports × depth-chart starters — approximate, coverage gaps logged); `home_playoff_stakes`, `away_playoff_stakes` (`alive`/`clinched`/`eliminated`; heuristic from standings, mainly meaningful weeks 15-18).

**`nfl_team_stats`** — rolling team form, one row per team per through-week (point-in-time correct: only games *before* the target week feed a week's row).
- `team`, `season`, `through_week`, `off_epa_play`, `def_epa_play`, `pass_epa`, `rush_epa`, `success_rate`, `pace`, `power_rating`.

**`nfl_markets`** — odds rows per game/book/market. `game_id`, `book`, `market_type` (spread/moneyline/total), `side`, `line`, `odds`, `captured_at`.

**`nfl_prediction_snapshots`** — near-exact copy of `mlb_prediction_snapshots` so grading, the nightly tracker script, and future cards generalize with almost no change.
- Per-market: `best_spread_*`, `best_ml_*`, `best_total_*` (each with `_pick`, `_line`, `_odds`, `_value_score`, `_profit`).
- Primary: `best_bet_*` (type, pick, value_score, profit).
- Grading: `actual_winner`, `actual_margin`, `actual_total`, `game_date`.

## Components

### `services/nfl/` package

- **`ingest.py`** — pulls `nfl_data_py.import_schedules()` (games, kickoffs, scores, roof/surface, QBs) and `import_pbp_data()` (play-by-play → EPA); upserts `nfl_games`, `nfl_game_context`. Idempotent upserts, same pattern as `services/mlb/ingest.py`.
- **`nfl_data.py`** — thin wrapper over `nfl_data_py` calls plus The Odds API (`services/data/odds_api.py`, `americanfootball_nfl`) → `nfl_markets`.
- **`features.py`** — derives `nfl_team_stats`: opponent-adjusted rolling off/def EPA, pass/rush EPA splits, success rate, pace, rest, home field, a lightweight power rating (folds in the Elo-type baseline). Strictly point-in-time — no future leakage.
- **`model_training.py`** — trains the two LightGBM regressors (mirrors `services/mlb/model_training.py`).
- **`scorer.py`** — loads both models, projects each upcoming game, scores every `nfl_markets` row, writes `best_spread`/`best_ml`/`best_total` per game (mirrors `services/mlb/scorer.py`).
- **`value_calculator.py`** — shared value math (see below), copied faithfully from the MLB/NBA implementation.

### Models (Phase 2)

- **`models/nfl_mov_v1.joblib`** — LightGBM regressor, target = point differential (home − away). Residual distribution (NFL-typical std ≈ 13.5 pts) converts one projection into P(spread cover) = P(margin > −line) and P(ML win) = P(margin > 0) — coherent by construction. Features: opponent-adjusted off/def EPA, pass/rush EPA splits, success rate, pace, rest, home field, power rating, `is_divisional`, `is_primetime`, plus candidate features (QB-change, `starters_out`).
- **`models/nfl_totals_v1.joblib`** — LightGBM regressor, target = total points. Features: combined pace/EPA, pass rates, `roof`/`is_dome`, `wind_mph`, `temp_f`, `playoff_stakes`. Residual distribution → P(over/under).
- **Calibration** — reuse the existing `calibration.py` pattern so a 60% pick wins ~60%.
- **Validation** — walk-forward by season (train on prior seasons, test on the next); 2010-2023 train, 2024 walk-forward test. Written backtest: ATS%, units, calibration curve, score-saturation check.

### Value math (`value_calculator.py`)

Carried over from the MLB/NBA implementation:

- `edge_pct` = model probability − de-vigged market-implied probability.
- **Qualification gate:** preserved **byte-for-byte** from the existing legacy formula. Hard constraint — not re-derived or re-tuned in this project.
- `sort_score` = unclamped `edge_pct × confidence × market_mult`; ranking selects the true argmax (no market-order bias).
- Display `value_score` = `100 · tanh(blended_edge_pct / 20)` (blended = `edge_pct × 0.5`) — the MLB-retune anti-saturation fix; scores spread ~30-95, essentially never peg at 100.
- `MAX_EDGE_PCT` blowup cap.
- `best_bet` = argmax across spread + ML + totals. **Totals shadow-gated:** `nfl_totals_in_best_bet=False` at launch; `best_total` still recorded; a re-entry gate (≥100 graded shadow totals, WR ≥53%, positive units — same thresholds as MLB) must pass before totals join `best_bet`. Note NFL's low game volume means ≥100 graded totals takes most of a season to reach; that is expected, and the gate simply stays closed until then rather than being loosened.

### Scheduler (`tasks/nfl_scheduler.py`)

NFL's weekly rhythm is the one real structural departure from MLB's daily cadence. An hourly tick handles games scattered across Thu / Sun early / Sun late / SNF / MNF:

- **Tue (post-MNF):** refresh `nfl_data_py` results, grade last week's snapshots, recompute `nfl_team_stats`.
- **Wed-Sun:** refresh odds daily into `nfl_markets`.
- **Per game:** snapshot ~90 min before *its own* kickoff (reuses MLB's per-game pre-kick windowing, triggered off the weekly slate).

### API (`api/nfl.py`)

Router at `/api/v1/nfl/…`, registered in `main.py` next to the MLB router.
- `/picks` — best bets, `min_value_score` threshold (default 40, like MLB).
- `/games` — slate + projections.
- `/debug/…` — backfill/odds debug endpoints under `/api/v1/nfl/debug/`.

### Config (`src/config.py`)

New `nfl_*` block: `nfl_mov_model_path`, `nfl_totals_model_path`, `nfl_suppress_totals` (default **False**), `nfl_totals_in_best_bet` (default False), `nfl_max_edge_pct`, `nfl_picks_threshold`.

> **Lesson carried from MLB:** `suppress_totals` and `totals_in_best_bet` are independent. `nfl_suppress_totals=False` keeps totals *scored and shadow-recorded* in `best_total`; `nfl_totals_in_best_bet=False` keeps them *out of `best_bet`*. Defaulting suppress to True would silently starve the shadow record (the exact MLB gotcha where the re-entry gate never accumulated data). So suppress defaults False here on purpose.

## Phasing

| Phase | Deliverable | Exit gate |
|---|---|---|
| **P1** | nflverse ingestion + `nfl_*` schema + features | 2010→present backfilled; `nfl_team_stats` computes with no future leakage; a spot-checked feature row for a known game reviewed |
| **P2** | MOV + totals models | Trained on 2010-2023, walk-forward validated on 2024; written backtest (ATS%, units, calibration, saturation) reviewed before scoring is wired |
| **P3** | scorer + value calc + snapshots | Dry-run scoring on a past week; 0 saturated scores; 0 totals in `best_bet`; snapshots write correctly |
| **P4** | weekly scheduler + `api/nfl.py` | Snapshots writing pre-kick and grading post-game on a live week; `/picks` serving |
| **P5** | social cards + frontend | *(separate future project — out of scope here)* |

Each phase gets its own implementation plan and validation. Nothing produces real betting output until P2's backtest earns it.

## Testing

- **P1:** unit tests for feature point-in-time correctness (assert a week's features never reference same-or-future-week games); ingestion idempotency (re-run upserts to zero net change); a golden spot-check row for one known historical game.
- **P2:** walk-forward backtest harness reporting ATS%, units, calibration curve, and score distribution; regression test that residual-derived P(cover)/P(win) are internally consistent (P(win) ≈ P(margin>0)).
- **P3:** dry-run scorer over a historical week asserting no saturated scores and no totals in `best_bet` while shadow-gated; value-math parity test against the legacy qualification-gate formula.
- **P4:** scheduler windowing test (a game snapshots once, ~90 min pre-kick); grading correctness against known final scores.

## Open questions / future work

- **QB-adjustment (Phase 2.5):** build a starting-QB value adjustment on top of the stored QB identities.
- **Social cards + frontend (P5):** separate project; team colors already stored.
- **Totals re-entry:** first gate check after enough shadow totals accumulate in-season (same cadence as MLB's biweekly check).
