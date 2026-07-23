# NFL Frontend + Evaluation API (P5) — Design Spec

**Date:** 2026-07-23
**Status:** Approved (brainstorming) → ready for implementation plan
**Depends on:** NFL Phases 1–4 merged (`api/nfl.py` live-code exists: `/nfl/picks`, `/nfl/games`, `/nfl/debug/odds`). See [[truline-nfl]].

## Problem / goal

The NFL vertical has a backend (odds, models, scoring, scheduler-disabled, picks/games API) but **no UI**. Build the NFL frontend — a picks/slate page and a performance page — plus the two backend evaluation endpoints the performance page needs, mirroring the existing MLB frontend + API exactly. Build now; deploy at the September go-live (no wasted work, ready to flip on).

## Scope

**In scope:**
- Backend: add `GET /nfl/evaluation/summary` + `GET /nfl/evaluation/daily` to `api/nfl.py` (mirror `api/mlb.py`).
- Frontend: `lib/nflApi.ts`, `lib/nflLogos.ts`, `components/nfl/NFLGameCard.tsx`, `pages/NFLPicks.tsx`, `pages/NFLEvaluation.tsx`, `Layout.tsx` nav + `App.tsx` routes.

**Out of scope (defer):** game-detail drill-down page; line-movement/props; any scheduler/go-live change (the frontend deploys at go-live, separately). No push/deploy in this build — commit locally on a branch.

**Data reality:** the backend isn't deployed and there are no real graded NFL games until ~Sept, so the pages render **empty / "season starts September" states** until then. `nfl_prediction_snapshots` currently holds ~12 demo rows (P3 dry-run, 2024 wk10) — enough to verify the eval page renders, not meaningful results.

## Design

### Component 1 — Evaluation API (`backend/src/api/nfl.py`)

Two endpoints mirroring `api/mlb.py`'s `/mlb/evaluation/{summary,daily}`, reading graded `nfl_prediction_snapshots` (columns `best_bet_result`/`best_bet_profit`, `best_spread_result`/`_profit`, `best_ml_result`/`_profit`, `best_total_*`, `game_date`, `kickoff_utc`):

- `GET /nfl/evaluation/summary` → `{total_predictions, graded, wins, losses, pushes, win_rate, total_profit, by_market}` where `by_market` breaks out **best_bet (totals, LIVE)** as the headline plus **spread** and **ml** as SHADOW lines (graded but tracked-not-bet). Grading semantics already exist in `snapshot.grade_snapshot`; the endpoint only aggregates stored `*_result`/`*_profit`.
- `GET /nfl/evaluation/daily?days=N` (default 30, `ge=1 le=60`) → list of `{date, predictions, wins, losses, pushes, win_rate, profit}` for the best_bet market, most-recent first.

Pydantic response models `NFLEvaluationSummary` / `NFLDailyPerformance` local to `api/nfl.py`. Register nothing new in `main.py` (router already registered). NFL is totals-forward: the summary **headlines best_bet (totals)**; spread/ML appear as clearly-labeled shadow rows, never as the primary number.

### Component 2 — API client (`frontend/src/lib/nflApi.ts`)

Copy `mlbApi.ts`'s axios boilerplate verbatim (base `/api/v1`, Bearer-token interceptor). Typed functions: `getNflPicks(minValueScore?, limit?)` → `/nfl/picks`; `getNflGames(season?, week?)` → `/nfl/games`; `getNflEvaluationSummary()` → `/nfl/evaluation/summary`; `getNflDailyEvaluation(days?)` → `/nfl/evaluation/daily`. TypeScript interfaces matching the backend response models.

### Component 3 — Logos (`frontend/src/lib/nflLogos.ts`)

Mirror `mlbLogos.ts`: a 32-team `NFL_TEAM_INFO` record keyed by abbr with `{name, city, primaryColor, secondaryColor}` (abbrs matching `constants.NFL_DIVISIONS` / `NFL_TEAM_NAME_TO_ABBR`), `getTeamLogo(abbr)` → `https://a.espncdn.com/i/teamlogos/nfl/500/{abbr}.png` (same CDN pattern the MLB/NBA cards use), `getTeamColor(abbr)`, `getTeamInfo(abbr)`. The `<img>` in the card uses an `onError` fallback to the mockup's **two-tone colored monogram crest** (primary/secondary split + abbr) so a missing logo degrades gracefully.

### Component 4 — Card (`frontend/src/components/nfl/NFLGameCard.tsx`)

The totals-forward card from the approved mockup, using the existing hardcoded design tokens (`#191c22` card, `#0b0e14` sub-card, `#a4e6ff` accent, `#66f796`/`#f59e0b`, mono numerics) and the `getScoreBadge` tier logic (≥70 strong/green, ≥60 moderate/cyan, else low; ≥65 left-edge glow):
- Status bar (week/kickoff + PRIMETIME/DIV/DOME pills), two teams (real logo + abbr + spread/ML), round **value-score badge** (the best_bet/totals value score).
- **Over/Under best-bet row highlighted** (the live-gated market); **spread + ML in a greyed "SHADOW" strip** (tracked, not bet).
- Cover-probability bar (over% fill). Consumes the `/nfl/picks` + `/nfl/games` shapes.

### Component 5 — Pages

- `pages/NFLPicks.tsx` (route `/nfl`): **Best Bets / Full Slate** toggle + `min_value_score` control (default 40), React Query (`@tanstack/react-query`, matching MLBPicks) against `getNflPicks` + `getNflGames`, grid of `NFLGameCard`. Skeleton loaders + a polished **empty state** ("NFL best bets return in September — the model is totals-forward; spread & ML are shadow-tracked until they beat the market"). Best Bets view filters to games with a qualifying `best_bet`; Full Slate shows all upcoming games (spread/ML shadow visible).
- `pages/NFLEvaluation.tsx` (route `/nfl/performance`): mirror `MLBEvaluation.tsx` — summary tiles (best_bet win%/units/record, with spread/ML shadow tiles clearly secondary) + a daily table/chart via `getNflEvaluationSummary` + `getNflDailyEvaluation`. Empty state when nothing graded.

### Component 6 — Nav + routing

- `Layout.tsx` `navItems`: add `{ path: '/nfl', label: 'NFL', icon: '🏈' }` and `{ path: '/nfl/performance', label: 'NFL Results' }` (both desktop `hidden md:flex` and the mobile scrollable row).
- `App.tsx`: `<Route path="/nfl" element={<NFLPicks/>} />` and `<Route path="/nfl/performance" element={<NFLEvaluation/>} />`, inside `<Layout>`.

## Data flow
`NFLPicks` → React Query → `nflApi.getNflPicks/getNflGames` → `/api/v1/nfl/*` → (prod: Vercel rewrite → Railway backend). `NFLGameCard` renders one game. `NFLEvaluation` → `nflApi.getNflEvaluation*`. All read-only GETs; no auth-gated mutations.

## Deploy
Build + commit locally on `nfl-frontend` (branch off `main`). **Do NOT deploy in this build.** At the September go-live: push backend to Railway + `vercel --prod` from `frontend/` (per the [[truline]] manual-deploy gotcha) — the NFL tab then serves real picks as the scheduler runs.

## Testing
- **Backend:** unit-test `/nfl/evaluation/summary` + `/daily` with a mocked session (mirror the existing `/nfl/picks` test) — seed graded snapshot rows, assert aggregation + shadow-vs-live market split; confirm both appear in the OpenAPI schema and `import src.main` is clean.
- **Frontend:** the repo's frontend test tooling is light (verify during planning); at minimum the components/pages type-check (`tsc`) and build (`vite build`) cleanly, and render against the mockup's structure. No new heavy test harness.

## Risks / notes
- **Backend-in-scope:** the eval page pulled backend work in (the `/nfl/evaluation/*` endpoints) — this build is backend + frontend, one coherent vertical, one plan.
- **Totals-forward everywhere:** picks and evaluation headline best_bet (totals); spread/ML are always the greyed shadow, never the primary metric — matches the gating (`nfl_spread_in_best_bet=False`).
- **ESPN logo dependency:** logos hotlink the ESPN CDN exactly like the live MLB/NBA cards; the two-tone crest `onError` fallback means a CDN miss never breaks a card.
- **Empty until season:** every page needs a real, designed empty state — not a spinner that never resolves — since there's no live NFL data until ~Sept.
