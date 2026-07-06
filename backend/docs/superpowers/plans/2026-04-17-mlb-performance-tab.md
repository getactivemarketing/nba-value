# MLB Performance Tab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an MLB performance page to truline.app showing season record, P/L, NRFI accuracy, underdog ML track record, and daily results.

**Architecture:** Backend already has `/mlb/evaluation/summary` and `/mlb/evaluation/daily` endpoints. We add 2 new endpoints (NRFI accuracy, underdog ML) and enhance the summary endpoint with by-type breakdown. Frontend gets a new `MLBEvaluation.tsx` page consuming all 4 endpoints, plus a route and nav link.

**Tech Stack:** FastAPI + SQLAlchemy async (backend), React 18 + TypeScript + Tailwind + React Query + Recharts (frontend)

**Spec:** `docs/superpowers/specs/2026-04-17-mlb-performance-tab.md`

---

### Task 1: Add by-type breakdown to existing summary endpoint + add NRFI and underdog endpoints

**Files:**
- Modify: `backend/src/api/mlb.py`

- [ ] **Step 1: Add by-type breakdown to EvaluationSummary model and endpoint**

In `mlb.py`, find the `EvaluationSummary` Pydantic model (around line 178) and add a `by_type` field:

```python
class EvaluationSummary(BaseModel):
    """Overall evaluation summary."""
    total_predictions: int
    graded_predictions: int
    wins: int
    losses: int
    pushes: int
    overall_win_rate: float | None
    total_profit: float
    by_value_tier: dict
    by_type: dict = {}  # ADD THIS LINE
```

Then in `get_evaluation_summary()` (around line 471), add by-type aggregation after the value tier loop. Before the `return EvaluationSummary(...)` block, add:

```python
        # Group by bet type
        by_type = {}
        for snap in snapshots:
            bt = (snap.best_bet_type or "unknown").lower()
            if bt not in by_type:
                by_type[bt] = {"wins": 0, "losses": 0, "pushes": 0, "profit": 0.0}
            if snap.best_bet_result == "win":
                by_type[bt]["wins"] += 1
            elif snap.best_bet_result == "loss":
                by_type[bt]["losses"] += 1
            else:
                by_type[bt]["pushes"] += 1
            by_type[bt]["profit"] += float(snap.best_bet_profit or 0)

        for bt_stats in by_type.values():
            decided = bt_stats["wins"] + bt_stats["losses"]
            bt_stats["win_rate"] = round(bt_stats["wins"] / decided, 3) if decided > 0 else None
            bt_stats["profit"] = round(bt_stats["profit"], 2)
```

And add `by_type=by_type` to the return statement.

- [ ] **Step 2: Add NRFI accuracy endpoint**

Add after `get_evaluation_summary()`:

```python
@router.get("/evaluation/nrfi")
async def get_nrfi_evaluation(
    days: int = Query(90, ge=1, le=365, description="Lookback days"),
) -> dict:
    """Get NRFI pick accuracy — how often our pregame NRFI predictions were correct."""
    from sqlalchemy import text

    async with async_session() as session:
        cutoff = date.today() - timedelta(days=days)

        result = await session.execute(text("""
            SELECT
                game_date, away_team, home_team,
                away_first_inning_runs, home_first_inning_runs
            FROM mlb_games
            WHERE pregame_tweet_posted = TRUE
              AND status = 'final'
              AND home_first_inning_runs IS NOT NULL
              AND game_date >= :cutoff
            ORDER BY game_date DESC
        """), {"cutoff": cutoff})
        rows = result.fetchall()

        total = len(rows)
        hits = sum(1 for r in rows if (r.away_first_inning_runs or 0) + (r.home_first_inning_runs or 0) == 0)

        recent = []
        for r in rows[:20]:
            first_inning_runs = (r.away_first_inning_runs or 0) + (r.home_first_inning_runs or 0)
            recent.append({
                "date": r.game_date.isoformat(),
                "away_team": r.away_team,
                "home_team": r.home_team,
                "result": "hit" if first_inning_runs == 0 else "miss",
                "first_inning_runs": first_inning_runs,
            })

        return {
            "total_picks": total,
            "nrfi_hits": hits,
            "accuracy": round(hits / total * 100, 1) if total > 0 else None,
            "recent": recent,
        }
```

- [ ] **Step 3: Add underdog ML evaluation endpoint**

Add after the NRFI endpoint:

```python
@router.get("/evaluation/underdogs")
async def get_underdog_evaluation(
    days: int = Query(90, ge=1, le=365, description="Lookback days"),
) -> dict:
    """Get underdog moneyline pick performance — picks at +100 or higher."""
    async with async_session() as session:
        cutoff = date.today() - timedelta(days=days)

        stmt = select(MLBPredictionSnapshot).where(
            and_(
                MLBPredictionSnapshot.game_date >= cutoff,
                MLBPredictionSnapshot.best_bet_type == "moneyline",
                MLBPredictionSnapshot.best_bet_odds >= 2.0,  # +100 or higher = underdog
                MLBPredictionSnapshot.best_bet_result.isnot(None),
            )
        ).order_by(desc(MLBPredictionSnapshot.game_date))

        result = await session.execute(stmt)
        snapshots = list(result.scalars().all())

        total = len(snapshots)
        wins = sum(1 for s in snapshots if s.best_bet_result == "win")
        losses = sum(1 for s in snapshots if s.best_bet_result == "loss")
        profit = sum(float(s.best_bet_profit or 0) for s in snapshots)

        # Average odds (American)
        odds_list = [float(s.best_bet_odds) for s in snapshots if s.best_bet_odds]
        avg_decimal = sum(odds_list) / len(odds_list) if odds_list else 0
        avg_american = int(round((avg_decimal - 1) * 100)) if avg_decimal >= 2.0 else 0

        # Biggest wins (by profit)
        won = sorted(
            [s for s in snapshots if s.best_bet_result == "win"],
            key=lambda s: float(s.best_bet_profit or 0),
            reverse=True,
        )[:5]

        biggest_wins = []
        for s in won:
            odds_dec = float(s.best_bet_odds or 0)
            odds_am = int(round((odds_dec - 1) * 100)) if odds_dec >= 2.0 else 0
            score = None
            if s.home_score is not None and s.away_score is not None:
                score = f"{s.away_score}-{s.home_score}"
            biggest_wins.append({
                "date": s.game_date.isoformat() if s.game_date else None,
                "team": s.best_bet_team,
                "odds_american": odds_am,
                "profit": round(float(s.best_bet_profit or 0), 2),
                "score": score,
            })

        return {
            "total_picks": total,
            "wins": wins,
            "losses": losses,
            "profit": round(profit, 2),
            "avg_odds_american": avg_american,
            "biggest_wins": biggest_wins,
        }
```

- [ ] **Step 4: Verify backend starts**

```bash
cd /Applications/XAMPP/xamppfiles/htdocs/Sites/Truline/backend
python3 -c "from src.api.mlb import router; print('Routes:', [r.path for r in router.routes])"
```

Expected: should include `/evaluation/summary`, `/evaluation/daily`, `/evaluation/nrfi`, `/evaluation/underdogs`

- [ ] **Step 5: Commit**

```bash
git add src/api/mlb.py
git commit -m "feat: add NRFI accuracy and underdog ML evaluation endpoints + by-type breakdown"
```

---

### Task 2: Add API client functions and hooks in frontend

**Files:**
- Modify: `frontend/src/lib/mlbApi.ts`
- Create: `frontend/src/hooks/useMLBEvaluation.ts`

- [ ] **Step 1: Add new types and API functions to `mlbApi.ts`**

Add these types after the existing `MLBEvaluationSummary` interface (around line 148):

```typescript
export interface NRFIEvaluation {
  total_picks: number;
  nrfi_hits: number;
  accuracy: number | null;
  recent: {
    date: string;
    away_team: string;
    home_team: string;
    result: 'hit' | 'miss';
    first_inning_runs: number;
  }[];
}

export interface UnderdogEvaluation {
  total_picks: number;
  wins: number;
  losses: number;
  profit: number;
  avg_odds_american: number;
  biggest_wins: {
    date: string | null;
    team: string;
    odds_american: number;
    profit: number;
    score: string | null;
  }[];
}
```

Add these API functions inside the `mlbApi` object (after `getFirstInningStats`):

```typescript
  async getNRFIEvaluation(days: number = 90): Promise<NRFIEvaluation> {
    const response = await client.get<NRFIEvaluation>(`/mlb/evaluation/nrfi?days=${days}`);
    return response.data;
  },

  async getUnderdogEvaluation(days: number = 90): Promise<UnderdogEvaluation> {
    const response = await client.get<UnderdogEvaluation>(`/mlb/evaluation/underdogs?days=${days}`);
    return response.data;
  },
```

- [ ] **Step 2: Create `useMLBEvaluation.ts` hooks file**

Create `frontend/src/hooks/useMLBEvaluation.ts`:

```typescript
import { useQuery } from '@tanstack/react-query';
import { mlbApi } from '@/lib/mlbApi';

export function useMLBEvaluationSummary() {
  return useQuery({
    queryKey: ['mlb', 'evaluation', 'summary'],
    queryFn: () => mlbApi.getEvaluationSummary(),
    staleTime: 300000,
  });
}

export function useMLBDailyResults(days: number = 14) {
  return useQuery({
    queryKey: ['mlb', 'evaluation', 'daily', days],
    queryFn: () => mlbApi.getDailyEvaluation(days),
    staleTime: 300000,
  });
}

export function useNRFIEvaluation(days: number = 90) {
  return useQuery({
    queryKey: ['mlb', 'evaluation', 'nrfi', days],
    queryFn: () => mlbApi.getNRFIEvaluation(days),
    staleTime: 300000,
  });
}

export function useUnderdogEvaluation(days: number = 90) {
  return useQuery({
    queryKey: ['mlb', 'evaluation', 'underdogs', days],
    queryFn: () => mlbApi.getUnderdogEvaluation(days),
    staleTime: 300000,
  });
}
```

- [ ] **Step 3: Commit**

```bash
cd /Applications/XAMPP/xamppfiles/htdocs/Sites/Truline/frontend
git add src/lib/mlbApi.ts src/hooks/useMLBEvaluation.ts
git commit -m "feat: add MLB evaluation API client and React Query hooks"
```

---

### Task 3: Create MLBEvaluation page

**Files:**
- Create: `frontend/src/pages/MLBEvaluation.tsx`

- [ ] **Step 1: Create the page**

Create `frontend/src/pages/MLBEvaluation.tsx`:

```tsx
import { useState } from 'react';
import { useMLBEvaluationSummary, useMLBDailyResults, useNRFIEvaluation, useUnderdogEvaluation } from '@/hooks/useMLBEvaluation';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { getTeamInfo, formatOdds } from '@/lib/mlbApi';

export function MLBEvaluation() {
  const [dailyDays, setDailyDays] = useState(14);

  const { data: summary, isLoading: summaryLoading } = useMLBEvaluationSummary();
  const { data: daily, isLoading: dailyLoading } = useMLBDailyResults(dailyDays);
  const { data: nrfi, isLoading: nrfiLoading } = useNRFIEvaluation(90);
  const { data: underdogs, isLoading: underDogsLoading } = useUnderdogEvaluation(90);

  const selectClass =
    'text-sm bg-[#0b0e14] border border-[#1e293b] text-[#f1f5f9] rounded px-2 py-1 font-mono focus:outline-none focus:border-[#a4e6ff]';

  const decided = (summary?.wins ?? 0) + (summary?.losses ?? 0);
  const winRate = decided > 0 ? ((summary?.wins ?? 0) / decided * 100).toFixed(1) : null;
  const roi = decided > 0 ? ((summary?.total_profit ?? 0) / (decided * 100) * 100).toFixed(1) : null;

  return (
    <div className="max-w-6xl mx-auto px-4 py-6 space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-[#f1f5f9] font-display tracking-tight">
          MLB <span className="text-[#a4e6ff]">PERFORMANCE</span>
        </h1>
        <p className="text-sm text-[#64748b] mt-1 font-mono">
          Season record, NRFI accuracy, and underdog ML track record
        </p>
      </div>

      {/* Season Summary Cards */}
      {summaryLoading && (
        <div className="flex justify-center py-8"><LoadingSpinner /></div>
      )}
      {!summaryLoading && summary && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <StatCard label="Record" value={`${summary.wins}-${summary.losses}${summary.pushes > 0 ? `-${summary.pushes}` : ''}`} />
          <StatCard label="Total Picks" value={String(summary.graded_predictions)} color="cyan" />
          <StatCard
            label="P/L (units)"
            value={`${summary.total_profit >= 0 ? '+' : ''}${summary.total_profit.toFixed(0)}u`}
            color={summary.total_profit >= 0 ? 'green' : 'red'}
          />
          <StatCard
            label="ROI"
            value={roi ? `${Number(roi) >= 0 ? '+' : ''}${roi}%` : '-'}
            color={Number(roi) >= 0 ? 'green' : 'red'}
          />
        </div>
      )}

      {/* By Bet Type */}
      {!summaryLoading && summary?.by_type && Object.keys(summary.by_type).length > 0 && (
        <div className="bg-[#191c22] rounded-xl border border-[#1e293b] p-5">
          <h2 className="text-[10px] text-[#64748b] uppercase font-bold tracking-widest mb-4">By Bet Type</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {['moneyline', 'runline', 'total'].map((bt) => {
              const stats = (summary.by_type as Record<string, any>)[bt];
              if (!stats) return null;
              const d = stats.wins + stats.losses;
              return (
                <div key={bt} className="bg-[#0b0e14] rounded-lg p-4 border border-[#1e293b]">
                  <div className="text-[10px] text-[#64748b] uppercase font-bold tracking-widest mb-2">
                    {bt === 'moneyline' ? 'Moneyline' : bt === 'runline' ? 'Runline' : 'Total'}
                  </div>
                  <div className="text-xl font-black font-mono text-[#f1f5f9]">
                    {stats.wins}-{stats.losses}
                  </div>
                  <div className={`text-sm font-mono mt-1 ${stats.profit >= 0 ? 'text-[#66f796]' : 'text-[#ef4444]'}`}>
                    {stats.profit >= 0 ? '+' : ''}{stats.profit.toFixed(0)}u
                    {d > 0 && ` (${(stats.wins / d * 100).toFixed(0)}%)`}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* NRFI Accuracy */}
      {nrfiLoading && (
        <div className="flex justify-center py-8"><LoadingSpinner /></div>
      )}
      {!nrfiLoading && nrfi && nrfi.total_picks > 0 && (
        <div className="bg-[#191c22] rounded-xl border border-[#1e293b] p-5">
          <h2 className="text-[10px] text-[#64748b] uppercase font-bold tracking-widest mb-4">NRFI Pick Accuracy</h2>
          <div className="grid grid-cols-3 gap-4 mb-4">
            <StatCard label="Accuracy" value={nrfi.accuracy ? `${nrfi.accuracy}%` : '-'} color="cyan" small />
            <StatCard label="Hits" value={String(nrfi.nrfi_hits)} color="green" small />
            <StatCard label="Misses" value={String(nrfi.total_picks - nrfi.nrfi_hits)} color="red" small />
          </div>
          {nrfi.recent.length > 0 && (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-[10px] text-[#64748b] uppercase font-bold tracking-widest border-b border-[#1e293b]">
                    <th className="pb-2">Date</th>
                    <th className="pb-2">Matchup</th>
                    <th className="pb-2 text-center">1st Inn Runs</th>
                    <th className="pb-2 text-right">Result</th>
                  </tr>
                </thead>
                <tbody>
                  {nrfi.recent.slice(0, 10).map((pick, i) => (
                    <tr key={i} className="border-b border-[#1e293b]">
                      <td className="py-2 text-[#94a3b8] font-mono">{pick.date}</td>
                      <td className="py-2 text-[#f1f5f9]">
                        {getTeamInfo(pick.away_team).name} @ {getTeamInfo(pick.home_team).name}
                      </td>
                      <td className="py-2 text-center font-mono text-[#94a3b8]">{pick.first_inning_runs}</td>
                      <td className={`py-2 text-right font-mono font-bold ${pick.result === 'hit' ? 'text-[#66f796]' : 'text-[#ef4444]'}`}>
                        {pick.result === 'hit' ? 'NRFI' : 'YRFI'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {/* Underdog ML Track Record */}
      {underDogsLoading && (
        <div className="flex justify-center py-8"><LoadingSpinner /></div>
      )}
      {!underDogsLoading && underdogs && underdogs.total_picks > 0 && (
        <div className="bg-[#191c22] rounded-xl border border-[#1e293b] p-5">
          <h2 className="text-[10px] text-[#64748b] uppercase font-bold tracking-widest mb-4">Underdog ML Picks</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
            <StatCard label="Record" value={`${underdogs.wins}-${underdogs.losses}`} small />
            <StatCard
              label="P/L"
              value={`${underdogs.profit >= 0 ? '+' : ''}${underdogs.profit.toFixed(0)}u`}
              color={underdogs.profit >= 0 ? 'green' : 'red'}
              small
            />
            <StatCard label="Avg Odds" value={`+${underdogs.avg_odds_american}`} color="cyan" small />
            <StatCard label="Total Picks" value={String(underdogs.total_picks)} small />
          </div>
          {underdogs.biggest_wins.length > 0 && (
            <>
              <h3 className="text-[10px] text-[#64748b] uppercase font-bold tracking-widest mb-2 mt-4">Biggest Wins</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left text-[10px] text-[#64748b] uppercase font-bold tracking-widest border-b border-[#1e293b]">
                      <th className="pb-2">Date</th>
                      <th className="pb-2">Team</th>
                      <th className="pb-2 text-center">Odds</th>
                      <th className="pb-2 text-center">Score</th>
                      <th className="pb-2 text-right">Profit</th>
                    </tr>
                  </thead>
                  <tbody>
                    {underdogs.biggest_wins.map((w, i) => (
                      <tr key={i} className="border-b border-[#1e293b]">
                        <td className="py-2 text-[#94a3b8] font-mono">{w.date}</td>
                        <td className="py-2 text-[#f1f5f9] font-medium">{getTeamInfo(w.team).name}</td>
                        <td className="py-2 text-center font-mono text-[#a4e6ff]">+{w.odds_american}</td>
                        <td className="py-2 text-center font-mono text-[#94a3b8]">{w.score || '-'}</td>
                        <td className="py-2 text-right font-mono font-bold text-[#66f796]">+{w.profit.toFixed(0)}u</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </div>
      )}

      {/* Daily Results */}
      <div className="bg-[#191c22] rounded-xl border border-[#1e293b] p-5">
        <div className="flex justify-between items-center mb-4 flex-wrap gap-3">
          <h2 className="text-[10px] text-[#64748b] uppercase font-bold tracking-widest">Daily Results</h2>
          <select value={dailyDays} onChange={(e) => setDailyDays(Number(e.target.value))} className={selectClass}>
            <option value={7}>7 days</option>
            <option value={14}>14 days</option>
            <option value={30}>30 days</option>
          </select>
        </div>

        {dailyLoading && (
          <div className="flex justify-center py-8"><LoadingSpinner /></div>
        )}

        {!dailyLoading && daily && daily.length > 0 && (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-[10px] text-[#64748b] uppercase font-bold tracking-widest border-b border-[#1e293b]">
                  <th className="pb-2">Date</th>
                  <th className="pb-2 text-center">Picks</th>
                  <th className="pb-2 text-center">Record</th>
                  <th className="pb-2 text-center">Win %</th>
                  <th className="pb-2 text-right">P/L</th>
                </tr>
              </thead>
              <tbody>
                {daily.map((day) => {
                  const record = `${day.wins}-${day.losses}${day.pushes > 0 ? `-${day.pushes}` : ''}`;
                  const wr = day.win_rate ? `${(day.win_rate * 100).toFixed(0)}%` : '-';
                  return (
                    <tr key={day.date} className="border-b border-[#1e293b]">
                      <td className="py-2 text-[#94a3b8] font-mono">{day.date}</td>
                      <td className="py-2 text-center font-mono text-[#f1f5f9]">{day.predictions}</td>
                      <td className="py-2 text-center font-mono text-[#f1f5f9]">{record}</td>
                      <td className="py-2 text-center font-mono text-[#94a3b8]">{wr}</td>
                      <td className={`py-2 text-right font-mono font-bold ${day.profit >= 0 ? 'text-[#66f796]' : 'text-[#ef4444]'}`}>
                        {day.profit >= 0 ? '+' : ''}{day.profit.toFixed(0)}u
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}

        {!dailyLoading && (!daily || daily.length === 0) && (
          <div className="text-center py-8 text-[#64748b] font-mono text-sm">
            No daily results available yet
          </div>
        )}
      </div>
    </div>
  );
}

// Reusable stat card
function StatCard({ label, value, color, small }: {
  label: string;
  value: string;
  color?: 'green' | 'red' | 'cyan';
  small?: boolean;
}) {
  const colorClass = color === 'green' ? 'text-[#66f796]'
    : color === 'red' ? 'text-[#ef4444]'
    : color === 'cyan' ? 'text-[#a4e6ff]'
    : 'text-[#f1f5f9]';

  return (
    <div className="bg-[#0b0e14] rounded-lg p-4 border border-[#1e293b]">
      <div className="text-[10px] text-[#64748b] uppercase font-bold tracking-widest mb-2">{label}</div>
      <div className={`${small ? 'text-xl' : 'text-2xl'} font-black font-mono ${colorClass}`}>{value}</div>
    </div>
  );
}
```

- [ ] **Step 2: Verify it compiles**

```bash
cd /Applications/XAMPP/xamppfiles/htdocs/Sites/Truline/frontend
npx tsc --noEmit 2>&1 | head -20
```

- [ ] **Step 3: Commit**

```bash
git add src/pages/MLBEvaluation.tsx
git commit -m "feat: create MLB evaluation page with season stats, NRFI, underdogs, daily results"
```

---

### Task 4: Add route and nav link

**Files:**
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/components/Layout.tsx`

- [ ] **Step 1: Add route to App.tsx**

Add import at top:
```tsx
import { MLBEvaluation } from '@/pages/MLBEvaluation';
```

Add route inside `<Routes>`:
```tsx
<Route path="/mlb/performance" element={<MLBEvaluation />} />
```

- [ ] **Step 2: Add nav item to Layout.tsx**

In the `navItems` array, add after the `MLB` entry:
```tsx
{ path: '/mlb/performance', label: 'MLB Results' },
```

- [ ] **Step 3: Verify frontend builds**

```bash
cd /Applications/XAMPP/xamppfiles/htdocs/Sites/Truline/frontend
npm run build 2>&1 | tail -5
```

- [ ] **Step 4: Commit**

```bash
git add src/App.tsx src/components/Layout.tsx
git commit -m "feat: add MLB performance route and nav link"
```

---

### Task 5: Deploy and verify

- [ ] **Step 1: Push backend**

```bash
cd /Applications/XAMPP/xamppfiles/htdocs/Sites/Truline/backend
git push origin main
```

Then: `cd /Applications/XAMPP/xamppfiles/htdocs/Sites/Truline && railway up`

- [ ] **Step 2: Deploy frontend**

```bash
cd /Applications/XAMPP/xamppfiles/htdocs/Sites/Truline/frontend
vercel --prod
```

- [ ] **Step 3: Verify endpoints**

Visit `https://nba-value-production.up.railway.app/api/v1/mlb/evaluation/summary` in a browser.
Visit `https://nba-value-production.up.railway.app/api/v1/mlb/evaluation/nrfi?days=90`
Visit `https://nba-value-production.up.railway.app/api/v1/mlb/evaluation/underdogs?days=90`

All should return JSON with real data.

- [ ] **Step 4: Verify frontend**

Visit `https://truline.app/mlb/performance` — should show the full MLB performance page with season stats, NRFI accuracy, underdog record, and daily results.
