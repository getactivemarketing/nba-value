# TruLine NFL Vertical — Overview & Change Log

**As of 2026-07-27. Status: fully built on `main`, NOT pushed, NOT deployed. Go-live is a deliberate ~September switch.**

The NFL vertical mirrors the MLB module end-to-end: data ingestion → models → scoring/snapshots → live scheduler → API → frontend. It is **isolated** from MLB/NBA (its own `nfl_*` tables, its own scheduler engine, additive router + nav) and **inert in production** until intentionally enabled.

## Current state (one-liner)

**Totals are the live product** (+10.2u / 53.9% ATS at the go-live gate, 2019–24). **Spread + moneyline are SHADOW** — recorded but not bet, because NFL spreads are efficient and neither the base model nor a QB-adjustment feature beat the market. The whole vertical sits on local `main`; enabling it is one push + one deploy + one config flag near Week 1.

## Architecture

### Backend (`backend/src/`)
- **Data:** `services/nfl/nfl_data.py` (nflverse wrapper), `features.py` (team EPA, rolling stats, starters_out, playoff_stakes), `ingest.py` (batched upserts), `tasks/nfl_backfill.py`. Tables: `nfl_games`, `nfl_team_stats`, `nfl_game_context`, `nfl_markets`, `nfl_prediction_snapshots` (all on prod Railway, backfilled 2010–2024: 3903 games / 7806 team-stat rows).
- **Models:** `services/nfl/training_data.py` (leakage-free feature frame, point-in-time `through_week = week-1`), `model_training.py` (LightGBM MOV + totals regressors, residual-std→prob, isotonic calibrator baked into totals bundle), `backtest.py` (walk-forward). Artifacts: `models/nfl_mov_v1.joblib` (resid_std 12.54), `models/nfl_totals_v1.joblib` (13.73, with calibrator).
- **Scoring:** `services/nfl/scorer.py` (`score_game`), `value_calculator.py` (MLB machinery + NFL-calibrated gate, band 0.05/0.99), `calibration_fit.py`, `snapshot.py` (build/grade).
- **Live layer:** `services/nfl/odds_client.py` (The Odds API `americanfootball_nfl`), `season_update.py` (schedule/team-stats/odds→markets), `live_features.py` (live feature row, shares `_feature_diffs` with training), `tasks/nfl_scheduler.py` (**ships DISABLED**).
- **API:** `api/nfl.py` — `/nfl/picks`, `/nfl/games`, `/nfl/evaluation/{summary,daily}`, `/nfl/debug/odds`. Registered in `main.py`.

### Frontend (`frontend/src/`)
- `lib/nflApi.ts` (typed client + team colors), `lib/nflLogos.ts` (ESPN CDN logos).
- `components/nfl/NFLGameCard.tsx` (totals-forward card).
- `pages/NFLPicks.tsx` (Best Bets / Full Slate) → `/nfl`; `pages/NFLEvaluation.tsx` (performance) → `/nfl/performance`.
- Nav: `🏈 NFL` + `NFL Results` in `Layout.tsx`.

### Gating (config, `src/config.py`)
`nfl_totals_in_best_bet=True`, `nfl_spread_in_best_bet=False`, `nfl_ml_in_best_bet=False`, `nfl_min_edge=0.05`, `nfl_max_edge=0.99`, `nfl_scheduler_enabled=False`, `nfl_snapshot_minutes_before=90`. **best_bet is always a total; spread/ML populate snapshots but as shadow.**

## Phase history (commits on `main`)

| Phase | What | Key result |
|---|---|---|
| **P1** data foundation (`3338b96..9dfc731`) | nflverse ingest, `nfl_*` tables, 2010–24 backfill | leakage=0, divisional=96/season, QB 272/272 |
| **P2** models (`4994ffa..db69300`) | MOV + totals LightGBM, walk-forward | **totals viable 54.4%/+24u; spread NOT (50.2%)** — NFL spreads efficient |
| **P3** scoring (`02dd5dc..7fef4cc`) | scorer, value calc, snapshots, isotonic calibration | gate GO: totals 53.9%/+10.2u; spread/ML shadow |
| **P4** live layer (`4eb43ed..b45d9fb`, merge `8ccef77`) | odds client, season update, scheduler (disabled), API | 225 live 2026 markets matched; primetime tz bug fixed |
| **P2.5a** QB-adjustment spike (`1add519..db4870f`, merge `7f16e02`) | `qb_delta` shrunk-EPA feature, gate | **NO-GO** — made spread WORSE (49.1% vs 50.2%); reverted, kept `qb_ratings.py` |
| **P5** frontend + eval API (`48b776f..9d14ca4`, merge `d468be3`) | NFL pages, cards, eval endpoints | READY; build-now/deploy-at-go-live |

## Key decisions & lessons

- **Totals-forward everywhere.** Totals are the only live-gated market. Spread/ML are rendered/stored as clearly-labeled SHADOW (tracked, not bet). Never present spread/ML as a primary bettable number.
- **Spread is efficient.** The base MOV model (50.2% ATS) and the QB-EPA-delta feature (P2.5a, 49.1%) both failed to beat the market. Spread going live needs a *richer* signal (real injury/depth-based projected-starter feed, or a non-EPA metric) — **not more data** (already 15 seasons; the ceiling is market efficiency, not sample size).
- **Primetime timezone bug (P4, fixed `170e22a`):** `_kickoff_utc` stored ET-wall-clock tagged UTC without converting, so every TNF/SNF/MNF game's date was one day early → its odds silently dropped (183/225 matched). Fixed with DST-aware `ZoneInfo("America/New_York")`; re-verified 225/225. Also required for correct pre-kick snapshot windowing.
- **Totals direction lives in `best_total_direction`, NOT `best_bet_team`** (which is null for totals — the scorer never sets `team=` for totals). The frontend card must derive Over/Under from `best_total_direction`. (Caught as a Critical in P5 review, fixed `d024ac1`.)
- **Calibration must load live:** the totals bundle carries an isotonic calibrator; live scoring loads it via `nfl_totals_model_path` so live == backtest.

## Go-live checklist (~September 2026, Week 1)

This is the deliberate switch — do NOT do these until intentionally going live:
1. Set `nfl_scheduler_enabled=True` (config/env).
2. Push `main` to Railway (backend deploy — starts the scheduler).
3. `vercel --prod` from `frontend/` (frontend deploy — NFL tab lights up; manual, does not auto-deploy from GitHub — see repo gotchas).
4. Validate one real week: confirm the scheduler snapshots totals ~90 min pre-kick and grades after finals (`grade_finals` was only mock-tested — no real finals existed pre-season).
5. Wire the nightly NFL performance tracker (like the MLB one).
6. Optional: enable the live odds probe (`/nfl/debug/odds?live=true`) to sanity-check odds ingestion before Week 1.

## Operational notes / gotchas

- **Run backend NFL jobs against prod DB:** `export DATABASE_URL=$(grep -oE "postgresql://[^\"']+" src/tasks/prediction_tracker.py | head -1)`. Set `DEBUG=false` for heavy jobs (SQL echo otherwise floods logs and slows training).
- **Frontend has no jest for components** → gate is `cd frontend && npm run build` (`tsc && vite build`) / `npx tsc --noEmit`. Named exports; React Query + axios `/api/v1`; Vercel rewrites `/api/*` → Railway.
- **Team logos:** `getTeamLogo` hotlinks the ESPN CDN (`.../teamlogos/nfl/500/{abbr}.png`, our abbrs lowercased except `LA→lar`, `WAS→wsh`) with a two-tone monogram crest `onError` fallback.
- **`nflverse` era team codes** (OAK/SD/STL) are normalized to current franchises via `constants.normalize_team`.
- **The Odds API already returns 2026 lines off-season** — odds ingestion is testable now; only snapshot→grade needs completed games.

## Not done (out of scope / future)
- Spread/ML remain shadow until a richer QB/injury signal proves out (P2.5b live starter feed was gated on the NO-GO'd spike — moot for shrunk-EPA).
- No game-detail drill-down page, line-movement, or props on the NFL frontend (deferred).
- Everything is local + unpushed; `main` is ahead of origin by the full NFL vertical.
