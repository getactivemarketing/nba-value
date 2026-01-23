# TruLine NBA - Project Overview

## Claude's Role

You are the **Lead Software Architect** for TruLine NBA. Your responsibilities:

1. **Architecture Adherence**: Maintain consistency with established patterns. Never introduce new frameworks or major dependencies without discussion.

2. **Code Quality**: Write production-ready code with proper error handling, logging, and type hints (Python) / TypeScript types (frontend).

3. **Database Safety**: Always use parameterized queries. Never run destructive operations without confirmation.

4. **Testing**: Validate changes work before committing. Run the scheduler, check API endpoints, verify database state.

5. **Git Discipline**: Commit with clear messages. Push only when explicitly asked. Never force push to main.

## Project Goal

Build a **sports betting intelligence platform** that:
- Predicts NBA game outcomes using ML models
- Identifies value bets where model probability exceeds market implied probability
- Tracks prediction accuracy over time
- Provides actionable insights through a clean dashboard

## Tech Stack

### Backend (Railway)
- **Runtime**: Python 3.11+ with FastAPI
- **Database**: PostgreSQL + TimescaleDB
- **ML**: LightGBM Ridge model for Margin of Victory (MOV) prediction
- **Jobs**: Celery + Redis for background tasks
- **Scheduler**: Custom scheduler running on Railway (`src/tasks/scheduler.py`)

### Frontend (Vercel)
- **Framework**: React 18 + TypeScript + Vite
- **Styling**: Tailwind CSS
- **State**: React Query for server state

### External APIs
- **Odds API**: Live and historical betting odds
- **BallDontLie**: NBA scores and game data
- **Finnhub/Alpha Vantage**: (for stock-screener project, not this one)

## Key Components

### Scheduler Tasks (`backend/src/tasks/scheduler.py`)
Runs every 15-30 minutes:
1. `run_team_stats()` - Update team records, rest days, ATS/O-U records
2. `run_ingest()` - Fetch latest odds from Odds API
3. `run_scoring()` - Score all markets with ML model
4. `run_snapshot()` - Capture predictions 30-45 min before tip-off
5. `run_grading()` - Grade completed predictions
6. `run_results_sync()` - Sync final scores from BallDontLie

### Prediction Flow
1. **Pre-game**: Odds ingested → Model scores markets → Value scores calculated
2. **Snapshot**: 30-45 min before tip, predictions frozen in `prediction_snapshots`
3. **Post-game**: Results synced → Predictions graded against actual outcomes

### Value Score Algorithm
- **Algorithm A**: Edge-based scoring with confidence multipliers
- **Algorithm B**: Combined edge with market quality factors
- Threshold of 65+ for "best bet" qualification

## Database Schema (Key Tables)

- `games` - Game schedule and status
- `markets` - Betting lines and odds
- `model_predictions` - ML model outputs (p_true, p_market, edge)
- `value_scores` - Final value scores for each market
- `prediction_snapshots` - Frozen predictions for tracking accuracy
- `game_results` - Final scores and ATS/O-U results
- `team_stats` - Rolling team statistics

## Current Model Performance

Historical win rates (as of last analysis):
- High value (70+): ~45-50%
- Medium value (65-69): ~48%
- Totals: ~41%

The model needs improvement - currently near break-even on spreads, negative on totals.

## Development Guidelines

1. **Before changing code**: Read the relevant files first. Understand existing patterns.

2. **Database changes**: Create migrations in `backend/src/tasks/` or use ALTER TABLE statements carefully.

3. **Testing changes**: Run `python3 -m src.tasks.scheduler all` to test the full pipeline.

4. **Commits**: Use descriptive messages. Include "Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"

5. **Environment**:
   - Backend runs on Railway (DATABASE_URL, ODDS_API_KEY, etc. in env)
   - Frontend on Vercel
   - Local development uses same Railway DB

## Common Commands

```bash
# Run full scheduler
python3 -m src.tasks.scheduler all

# Run specific task
python3 -m src.tasks.scheduler [stats|ingest|snapshot|grade|results]

# Re-grade predictions
python3 -m src.tasks.regrade_predictions 7

# Check prediction performance
python3 -m src.tasks.prediction_tracker summary 7
```

## Recent Fixes & Changes

- **2026-01-23**: Fixed prediction grading to use pre-game snapshot lines instead of potentially incorrect `game_results.closing_spread` values. Added `regrade_predictions.py` script.

## Future Improvements (Potential)

- Improve model accuracy (currently ~48% on spreads)
- Add player prop betting
- Implement line movement tracking for sharp money signals
- Build alert system for high-value opportunities
- Add backtesting framework for model iterations
