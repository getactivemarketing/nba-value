# TruLine - Project Overview

## Claude's Role

You are the **Lead Software Architect** for TruLine. Your responsibilities:

1. **Architecture Adherence**: Keep to established patterns. No new frameworks or major dependencies without discussion.
2. **Code Quality**: Production-ready code with error handling, logging and type hints (Python) / TypeScript types (frontend).
3. **Database Safety**: Parameterized queries only. Never run destructive operations without confirmation.
4. **Testing**: Test first (red, then green). Run the suites below before committing.
5. **Git Discipline**: Clear commit messages. Push only when asked. Never force push to main.

## What TruLine Is Now: TruLine Edge

Since the 2026-27 NBA opener (2026-10-20), the product is **TruLine Edge**: a market-consensus +EV finder.
It does not predict games. It devigs many books' prices into a fair price and lists any book paying more
than fair. Alongside the board it has an odds screen, player hit-rate pages, and an honest track record
built on closing-line value (CLV).

- **NBA Edge**: live board, prop grading from BallDontLie box scores.
- **NFL Edge**: live board, matchup tags, anytime TD. **Tracking-only**: the edge is not yet proven on NFL results.
- **Legacy model picks** (the NBA MOV model, MLB, NFL value models) are hidden (`NBA_PICKS_PUBLIC=false`).
  Their 2025-26 NBA record came from a broken model that output a constant; don't cite those win rates.

Specs and plans: `backend/docs/superpowers/{specs,plans}/2026-10-1*-edge-*`.
NFL backtest report: `backend/docs/backtests/nfl-2023-2025.md`.

## Architecture

### Backend (Railway project `nba`, us-east4)
- **`nba-value`**: FastAPI app. Edge routes live under `/api/v1/edge` (`backend/src/edge/api.py`): `/games`,
  `/ev`, `/odds/{id}`, `/players/{bdl_id}`, `/results`, `/health`, and admin `/status`.
  Every route takes `?sport=nba|nfl`.
- **`edge-worker`**: `python -m src.edge.worker`, a single process holding an advisory lock. It polls the
  Odds API, prices quotes, records opportunities, loads box scores, grades, and prunes history.
- **Postgres + Redis**. Deploy settings live in the Railway dashboard (Nixpacks, root `backend`), not in
  config files. The watch patterns are `backend/src/**` and `backend/requirements.txt`, so a commit outside
  those paths (or an empty one) is SKIPPED.

### Frontend (Vercel, git-linked)
React 18 + TypeScript + Vite + Tailwind + React Query. Pushing `main` deploys it.

### External APIs
- **The Odds API**: lines and props. The key is set on `edge-worker` only; watch the credit cap (`EDGE_CREDIT_CAP`).
- **BallDontLie** (ALL-STAR tier): NBA + NFL rosters and box scores. It serves **live** in-progress stats,
  so only rows whose game status is `Final` are stored.
- **nflverse**: NFL snap counts.

## Key Edge Modules (`backend/src/edge/`)

| Module | Job |
|---|---|
| `engine.py` | Devig, fair price, EV. Pairs with >10% hold are skipped (`MAX_HOLD`) |
| `ingest.py`, `client.py`, `schedule.py` | Odds API polling and credit budgeting |
| `opportunities.py`, `grading.py` | Log +EV rows; closing line, CLV, win/loss |
| `players.py` / `nfl_players.py` | Name matching (`NAME_ALIASES`), box scores, hit rates |
| `matchups.py`, `tagging.py`, `nfl_injuries.py` | NFL matchup tags (only `def_soft` tailwind is proven) |
| `backtest/` | `python -m src.edge.backtest fetch|replay|report` (costs Odds API credits) |
| `views.py` | Read models behind the API |

Worker settings (env on `edge-worker`): `EDGE_SPORTS`, `EDGE_MIN_EV`, `EDGE_VERIFY_EV`, `EDGE_CREDIT_CAP`,
`EDGE_NFL_PLAYERS`, `EDGE_BOX_RELOAD_DAYS` (default 3: NBA box scores reloaded and props regraded this many
ET days back; raise it briefly to repair older days).

## Testing

```bash
cd backend
# Unit tests
python3 -m pytest tests/unit -q -p no:cacheprovider

# DB tests need a throwaway Postgres 16 (localhost:5432 hangs on this Mac, so use 55432):
P=/opt/homebrew/opt/postgresql@16/bin
$P/initdb -D <dir> -U edge -A trust && $P/pg_ctl -D <dir> -o "-p 55432 -k ''" -w start
$P/createdb -h 127.0.0.1 -p 55432 -U edge edgetest
EDGE_TEST_DB_URL=postgresql://edge@127.0.0.1:55432/edgetest \
  python3 -m pytest tests/unit tests/integration/test_edge_store_db.py -q -p no:cacheprovider   # ~5 min
```

Unit tests must not touch DB-backed routes. `test_nfl_ingest` / `test_nfl_odds_live` fail locally for
environmental reasons.

## Operating Notes

- **After every push, check that both Railway services deployed.** A build that fails in Railway's builder
  can't be redeployed from the dashboard; push a small commit under `backend/src` instead.
- Use `git merge --no-edit` (otherwise Vim opens for the merge message).
- Admin calls send `x-cron-key` (the copy is in git-ignored `.cron_api_key`); the key also bypasses the rate limit.
- Commits end with `Co-Authored-By: Claude <model> <noreply@anthropic.com>`.

## Legacy Model Pipeline (hidden, kept for reference)

`backend/src/tasks/scheduler.py` runs the old ML pipeline (team stats → odds ingest → scoring → snapshot →
grading → results). Grading truth is `prediction_snapshots.best_bet_line` / `best_total_line`;
`markets.line` and `game_results.closing_spread` contain corrupt values. Commands:
`python3 -m src.tasks.scheduler [all|stats|ingest|snapshot|grade|results]`,
`python3 -m src.tasks.regrade_predictions 7`.
