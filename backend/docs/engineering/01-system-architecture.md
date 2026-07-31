# System Architecture Specification

**Status:** Living document. Update in the same PR as any change it describes.
**Last verified against code:** 2026-07-31

---

## 1. What TruLine is

A sports betting intelligence platform running three independent **verticals** (NBA, MLB, NFL) over one shared runtime. Each vertical follows the same seven-stage pipeline:

```
ingest schedule -> ingest stats -> ingest odds -> score -> snapshot -> grade -> report
```

Verticals share infrastructure (database, FastAPI app, deploy target, frontend shell) and share *nothing* at the model or scoring layer. NBA's scoring bug cannot affect MLB. This isolation is deliberate and should be preserved: it is why the NFL vertical could be built and deployed while dark without touching a live money path.

---

## 2. Runtime topology

```
                    ┌──────────────────────────────┐
                    │  Vercel — frontend           │
                    │  React 18 + Vite + Tailwind  │
                    │  truline.app                 │
                    └──────────────┬───────────────┘
                                   │ /api/* rewrite
                                   ▼
                    ┌──────────────────────────────┐
                    │  Railway — nba-value service │
                    │  FastAPI (uvicorn)           │
                    │  + in-process schedulers     │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────┴───────────────┐
                    │  Railway PostgreSQL          │
                    │  (TimescaleDB extensions)    │
                    └──────────────────────────────┘
```

**One Railway service runs both the API and the schedulers.** They share a process and a connection pool. A scheduler task that hangs holds a connection the API also needs. This is a known coupling, acceptable at current scale, and the first thing to split if the API ever gets latency-sensitive.

### Deploy

| Component | Trigger | Notes |
|---|---|---|
| Backend | `git push origin main` → Railway auto-deploy | **Pushing main is going live.** No staging environment exists. |
| Frontend | `vercel --prod` from repo root | Does *not* auto-deploy on push. Manual step, easily forgotten. |

`railway.json` sets `startCommand: uvicorn src.main:app`, healthcheck `/health`, restart `ON_FAILURE` max 10.

**Historical hazard:** a duplicate Railway service (`dynamic-bravery`) once ran stale code against the same database, racing the live scheduler and producing contradictory picks. Verify `railway status` resolves to the single `nba-value` service before diagnosing "impossible" data.

---

## 3. Service inventory

### `src/api/` — HTTP surface
`health, markets, bets, evaluation, admin, trends, backtest, mlb, nfl`
All except `health` mount under `settings.api_v1_prefix` (`/api/v1`).

### `src/services/` — domain logic

| Package | Owns |
|---|---|
| `data/` | External clients: `odds_api`, `balldontlie`, `nba_stats` |
| `ml/` | NBA models: MOV, spread v2, totals v1–v3, calibration, probability |
| `scoring/` | NBA scoring: `algorithm_a`, `algorithm_b`, confidence, market quality, props |
| `mlb/` | Full MLB vertical: ingest, features, scorer, value_calculator, model_training, weather, pitcher quality |
| `nfl/` | Full NFL vertical: nfl_data, ingest, features, live_features, scorer, value_calculator, calibration_fit, snapshot, season_update, backtest, qb_ratings |
| `notifications/` | `sms` (Textbelt), `pick_alerts` (message formatting) |
| `social/` | Blotato, Twitter, Typefully, image generation |
| `injuries.py` | Cross-sport injury data |

### `src/tasks/` — schedulers and batch jobs
`scheduler.py` (NBA), `mlb_scheduler.py`, `nfl_scheduler.py` (**disabled**), `social_scheduler.py`, plus retrain/backfill/backtest one-shots.

---

## 4. Data flow — MLB as the reference implementation

NBA and NFL differ in features and cadence, not in shape.

```
MLB Stats API ─┐
Odds API ──────┼─> ingest ──> mlb_games, mlb_markets, mlb_team_stats,
Weather API ───┘              mlb_pitcher_stats, mlb_game_context
                                        │
                                        ▼
                          MLBFeatureCalculator (28 features)
                                        │
                                        ▼
                          LightGBM run-diff model  ──> predicted_run_diff
                                        │
                                        ▼
                    logistic k=0.391 ──> p_home_win / p_home_cover
                                        │
                                        ▼
                    devig market odds ──> market_prob (per book, per market)
                                        │
                                        ▼
                    MLBValueCalculator ──> raw_edge, edge_pct,
                                           gate_score / sort_score / value_score
                                        │
                                        ▼
                    gate + selection ───> best_ml / best_rl / best_total / best_bet
                                        │
                                        ▼   (T-60min)
                          mlb_prediction_snapshots  ◄── IMMUTABLE RECORD
                                        │
                          ┌─────────────┼─────────────┐
                          ▼             ▼             ▼
                    SMS alert      frontend      grading
```

### Cadence (MLB)

| Task | Interval |
|---|---|
| `sync_teams` | daily 06:00 |
| `ingest_games` / `update_stats` / `ingest_weather` | 2h |
| `ingest_odds` / `run_scoring` | 30m |
| `run_snapshot` | 15m (freezes games starting within 60m) |
| `run_grading` / `health_check` | 1h |
| `sync_results` | 2h |
| `pick_alerts` (social_scheduler) | 10m |

---

## 5. State stores and their guarantees

This is the most important section in this document. Each table has a different durability contract, and confusing them has caused every serious data incident so far.

| Table | Contract | Implication |
|---|---|---|
| `mlb_markets` | **Overwritten in place.** Current odds only. | Historical candidate sets before 2026-07-31 are **unrecoverable**. Backtests over that period cannot ask "what else was available that night". |
| `mlb_odds_snapshots` | **Append-only**, written when a book's price moves. | Line movement, closing lines, CLV. Live from 2026-07-31. |
| `mlb_predictions` | Upserted per `(game_id, market_type)`. Latest scoring run wins. | Not a historical record. |
| `mlb_prediction_snapshots` | **Append-only, immutable after creation.** One row per game. | The single source of truth for what was actually bet, at what price. |
| `mlb_games` | Mutable; `status` and scores updated post-game. | |
| `mlb_game_context` | Mutable; weather, venue, park factor. | |

**Rule: grading and P&L read only from `mlb_prediction_snapshots`.** Any code that grades by re-deriving from `mlb_markets` is a bug — the odds it reads are not the odds that were bet.

NFL mirrors this with `nfl_*`. NBA uses `prediction_snapshots` / `markets` / `game_results` and carries known legacy data corruption documented in the root `CLAUDE.md` (`markets.line` and `game_results.closing_spread` are unreliable for completed games).

### Known gap
No table retains **closing lines**. Closing-line value cannot currently be computed for any vertical. See `04-research-experimentation.md` §5 — this is the highest-value missing dataset in the system.

---

## 6. External dependencies

| Provider | Used by | Failure mode |
|---|---|---|
| The Odds API | all three | Missing odds → no candidates → silent "no picks" night, indistinguishable from "no edge" |
| MLB Stats API | MLB | Stale stats → model runs on old features, no error raised |
| BallDontLie | NBA | scores/schedule |
| nflverse | NFL | Backfill 2010–2024 |
| Weather API | MLB | Collected; **`wind_factor` never computed** — see `02-feature-engineering-playbook.md` §7 |
| Textbelt | alerts | Prepaid credits; silent stop at zero |
| Blotato / Typefully / Twitter | social | Non-critical |

**Every one of these fails silently into "no picks."** There is no alert distinguishing "the model found nothing" from "the odds feed returned nothing." Adding that distinction is tracked as an open item.

---

## 7. Configuration and kill switches

All in `src/config.py` (pydantic settings, env-overridable). These are the production control surface:

```python
suppress_totals        = True    # MLB totals not scored at all
totals_in_best_bet     = False   # MLB totals cannot become a pick
runline_in_best_bet    = False   # MLB runline scored + stored, never bet
mlb_totals_model_path  = "models/mlb_totals_v2.joblib"

nfl_totals_in_best_bet = True
nfl_spread_in_best_bet = False
nfl_ml_in_best_bet     = False
nfl_scheduler_enabled  = False   # NFL generates nothing until flipped
```

**Design principle, already load-bearing: a market can be disabled without removing its code.** Runline and totals are still scored and stored for shadow evaluation while excluded from `best_bet`. This is what makes re-entry gates possible and must be preserved. See `03-model-governance.md` §4.

Environment drift is a live risk: `suppress_totals` defaults to `True` in code but the Railway env sets `SUPPRESS_TOTALS=false` so the totals shadow record keeps accumulating. **Code defaults do not describe production.** Verify against Railway env vars before reasoning about live behavior.

---

## 8. Frontend

React 18 + Vite + Tailwind, React Query for server state. Two-level nav: sport switcher (NBA / MLB / NFL), then contextual sub-pages. `frontend/vercel.json` rewrites `/api/*` to the Railway host.

The frontend renders whatever the API returns and holds no betting logic. Keep it that way — a display rule that lives only in the frontend is invisible to backtests.

---

## 9. Architectural rules

1. **Verticals do not import each other.** Shared code goes in `services/data/` or `utils/`.
2. **Snapshots are immutable.** Grading reads frozen columns, never re-derives.
3. **A market is disabled by config, not by deletion.** Preserves shadow evaluation.
4. **Model artifacts are files with an explicit feature contract.** See `03-model-governance.md`.
5. **Pushing `main` is a production deploy.** There is no staging tier.
6. **New scheduled jobs ship disabled** and are enabled as a separate, deliberate change.

---

## 10. Open architectural risks

| Risk | Impact | Status |
|---|---|---|
| API and schedulers share a process/pool | Scheduler stall degrades API | Accepted at current scale |
| No staging environment | Every change tested in prod | Accepted; mitigated by config flags |
| No closing-line capture | CLV unmeasurable | **Open — highest value** |
| Silent-failure ambiguity in ingest | "No picks" is unexplained | Open |
| `mlb_markets` overwrite | Backtests structurally limited | Accepted; would need a new append-only odds table |
| Frontend deploy is manual | Backend/frontend skew | Open |
