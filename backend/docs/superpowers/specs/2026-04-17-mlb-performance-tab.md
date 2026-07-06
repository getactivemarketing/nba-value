# MLB Performance Tab — Design Spec

## Goal

Add an MLB performance/results page to truline.app so users (and the founder) can track season record, P/L, NRFI accuracy, and underdog ML track record. Mirrors the existing NBA evaluation page with MLB-specific additions.

## Backend — 4 new endpoints in `src/api/mlb.py`

All use SQLAlchemy async sessions (existing pattern in `mlb.py`). Query `mlb_prediction_snapshots` and `mlb_games`.

### `GET /api/v1/mlb/evaluation/summary`

Params: `days` (int, default 90), `min_value` (float, default 0)

Returns:
```json
{
  "period_days": 90,
  "total_picks": 120,
  "wins": 65,
  "losses": 52,
  "pushes": 3,
  "win_rate": 55.6,
  "profit": 234.50,
  "roi": 2.0,
  "by_type": {
    "moneyline": { "wins": 30, "losses": 20, "profit": 180.0 },
    "runline": { "wins": 25, "losses": 22, "profit": 34.5 },
    "total": { "wins": 10, "losses": 10, "profit": 20.0 }
  }
}
```

Source: `mlb_prediction_snapshots WHERE actual_winner IS NOT NULL AND best_bet_type IS NOT NULL`

### `GET /api/v1/mlb/evaluation/nrfi`

Params: `days` (int, default 90)

Returns:
```json
{
  "total_picks": 45,
  "nrfi_hits": 30,
  "accuracy": 66.7,
  "recent": [
    {
      "date": "2026-04-16",
      "away_team": "NYY",
      "home_team": "BOS",
      "nrfi_pct_predicted": 71,
      "result": "hit",
      "first_inning_runs": 0
    }
  ]
}
```

Source: `mlb_games WHERE pregame_tweet_posted = TRUE AND status = 'final'`. NRFI hit = `(home_first_inning_runs + away_first_inning_runs) == 0`.

### `GET /api/v1/mlb/evaluation/underdogs`

Params: `days` (int, default 90)

Returns:
```json
{
  "total_picks": 25,
  "wins": 12,
  "losses": 13,
  "profit": 287.0,
  "avg_odds_american": 155,
  "biggest_wins": [
    {
      "date": "2026-04-15",
      "team": "GSW",
      "opponent": "LAC",
      "odds_american": 180,
      "profit": 180.0,
      "score": "118-105"
    }
  ]
}
```

Source: `mlb_prediction_snapshots WHERE best_bet_type = 'moneyline' AND best_bet_odds >= 2.0 AND best_bet_result IS NOT NULL`

### `GET /api/v1/mlb/evaluation/daily`

Params: `days` (int, default 14)

Returns: Same structure as NBA `/evaluation/daily` — per-day breakdown with individual picks, grouped by date, including by_type subtotals.

Source: `mlb_prediction_snapshots WHERE actual_winner IS NOT NULL AND best_bet_type IS NOT NULL`

## Frontend — New page + route + nav

### `src/pages/MLBEvaluation.tsx`

Layout (top to bottom):
1. **Season summary cards** (4 in a row): W-L Record, P/L Units, ROI%, Total Picks
2. **By bet type** (3 mini cards): Moneyline record + P/L, Runline record + P/L, Total record + P/L
3. **NRFI accuracy section**: accuracy %, hit/miss bar, last 10 picks with results
4. **Underdog ML section**: record, P/L, biggest wins table (date, team, odds, profit)
5. **Daily results**: expandable per-day cards with individual pick details
6. **Cumulative P/L chart**: reuse Recharts line chart pattern

### `src/App.tsx`

Add route: `<Route path="/mlb/performance" element={<MLBEvaluation />} />`

### `src/components/Layout.tsx`

Add nav item: `{ path: '/mlb/performance', label: 'MLB Results', icon: '📊' }`

## Styling

Follows existing dark theme: `bg-tru-bg`, `text-txt-primary`, `accent-cyan` accents, `tru-surface` cards, `tru-border` borders. Uses Tailwind throughout. No new dependencies.

## Files Changed

- `backend/src/api/mlb.py` — add 4 evaluation endpoints (~200 lines)
- `frontend/src/pages/MLBEvaluation.tsx` — new page (~300 lines)
- `frontend/src/App.tsx` — add import + route (2 lines)
- `frontend/src/components/Layout.tsx` — add nav item (1 line)
