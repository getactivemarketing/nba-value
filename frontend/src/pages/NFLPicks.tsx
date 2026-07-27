import { useState, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { nflApi, type NFLGameSummary } from '@/lib/nflApi';
import { NFLGameCard } from '@/components/nfl/NFLGameCard';
import { ErrorMessage } from '@/components/ui/ErrorMessage';

type ViewMode = 'best-bets' | 'full-slate';

const DEFAULT_MIN_VALUE_SCORE = 40;

function GameCardSkeleton() {
  return (
    <div className="rounded-xl bg-[#191c22] border border-[#1e293b] overflow-hidden animate-pulse">
      <div className="p-5 pb-3">
        <div className="flex items-center justify-between mb-4">
          <div className="h-3 bg-[#272a31] rounded w-32" />
          <div className="h-3 bg-[#272a31] rounded w-16" />
        </div>
        <div className="flex justify-between items-start mb-5">
          <div className="flex flex-col gap-2">
            <div className="flex items-center gap-2.5">
              <div className="w-8 h-8 bg-[#272a31] rounded-full" />
              <div className="h-5 bg-[#272a31] rounded w-24" />
            </div>
            <div className="flex items-center gap-2.5">
              <div className="w-8 h-8 bg-[#272a31] rounded-full" />
              <div className="h-5 bg-[#272a31] rounded w-20" />
            </div>
          </div>
          <div className="flex flex-col items-end gap-1">
            <div className="h-3 bg-[#272a31] rounded w-16" />
            <div className="h-8 bg-[#272a31] rounded-full w-24" />
          </div>
        </div>
        <div className="pt-4 border-t border-slate-700/30">
          <div className="h-3 bg-[#272a31] rounded w-20 mb-2" />
          <div className="h-7 bg-[#272a31] rounded-lg w-40" />
        </div>
      </div>
      <div className="bg-[#0b0e14] border-t border-[#1e293b] px-5 py-2.5">
        <div className="h-3 bg-[#272a31] rounded w-48" />
      </div>
    </div>
  );
}

function EmptyStatePanel() {
  return (
    <div className="col-span-full rounded-xl bg-[#191c22] border border-[#1e293b] p-10 text-center">
      <p className="text-3xl mb-3">🏈</p>
      <h3 className="text-lg font-black font-mono text-txt-primary tracking-tight mb-2">
        NFL best bets return in September
      </h3>
      <p className="text-sm text-slate-500 font-mono max-w-md mx-auto leading-relaxed">
        The model is totals-forward for launch — spread &amp; moneyline picks stay shadow-tracked
        until they show an edge over the market.
      </p>
    </div>
  );
}

function ViewToggle({ mode, onChange }: { mode: ViewMode; onChange: (m: ViewMode) => void }) {
  return (
    <div className="inline-flex bg-[#191c22] border border-[#1e293b] rounded-lg p-1">
      {(
        [
          { key: 'best-bets' as const, label: 'Best Bets' },
          { key: 'full-slate' as const, label: 'Full Slate' },
        ]
      ).map((opt) => (
        <button
          key={opt.key}
          onClick={() => onChange(opt.key)}
          className={`px-4 py-1.5 rounded-md text-sm font-bold font-mono transition-all ${
            mode === opt.key
              ? 'bg-[#a4e6ff] text-[#0b0e14]'
              : 'text-slate-400 hover:text-txt-primary'
          }`}
        >
          {opt.label}
        </button>
      ))}
    </div>
  );
}

export function NFLPicks() {
  const [viewMode, setViewMode] = useState<ViewMode>('best-bets');
  const [minValueScore, setMinValueScore] = useState<number>(DEFAULT_MIN_VALUE_SCORE);

  // Picks feed — drives the min_value_score-qualified pick count shown alongside the toggle.
  const {
    data: picksData,
    isLoading: picksLoading,
    error: picksError,
  } = useQuery({
    queryKey: ['nfl-picks', minValueScore],
    queryFn: () => nflApi.getPicks(minValueScore),
  });

  // Games feed — card source for BOTH views so NFLGameCard always gets NFLGameSummary.
  const {
    data: gamesData,
    isLoading: gamesLoading,
    error: gamesError,
  } = useQuery({
    queryKey: ['nfl-games'],
    queryFn: () => nflApi.getGames(),
  });

  const games = gamesData?.games ?? [];

  const bestBetGames = useMemo(
    () =>
      games.filter(
        (g): g is NFLGameSummary =>
          g.best_bet_type === 'total' &&
          g.best_bet_value_score != null &&
          g.best_bet_value_score >= minValueScore
      ),
    [games, minValueScore]
  );

  const isLoading = viewMode === 'best-bets' ? gamesLoading || picksLoading : gamesLoading;
  const error = gamesError || (viewMode === 'best-bets' ? picksError : null);
  const displayedGames = viewMode === 'best-bets' ? bestBetGames : games;
  const qualifyingCount = picksData?.total ?? bestBetGames.length;

  return (
    <div className="max-w-7xl mx-auto px-4 py-6">
      {/* Header */}
      <div className="mb-6">
        <div className="flex items-center gap-3 mb-1">
          <h1 className="text-2xl font-black text-txt-primary font-mono tracking-tight">
            NFL <span className="text-[#a4e6ff]">Best Bets</span>
          </h1>
          <div className="flex items-center gap-1.5">
            <div className="w-1.5 h-1.5 rounded-full bg-[#66f796] animate-pulse" />
            <span className="text-[10px] text-slate-500 font-bold uppercase tracking-widest font-mono">
              Live Feed
            </span>
          </div>
        </div>
        <p className="text-slate-500 text-sm font-mono">
          Totals-forward model — spread &amp; moneyline are shadow-tracked until they beat the market.
        </p>
      </div>

      {/* Controls: view toggle + min value score */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-6">
        <ViewToggle mode={viewMode} onChange={setViewMode} />

        <div className="flex items-center gap-3 bg-[#191c22] border border-[#1e293b] rounded-lg px-4 py-2">
          <label htmlFor="min-value-score" className="text-[10px] text-slate-500 font-bold uppercase tracking-widest font-mono whitespace-nowrap">
            Min Value
          </label>
          <input
            id="min-value-score"
            type="range"
            min={0}
            max={100}
            step={5}
            value={minValueScore}
            onChange={(e) => setMinValueScore(Number(e.target.value))}
            className="w-32 accent-[#a4e6ff]"
          />
          <span className="text-sm font-black font-mono text-[#a4e6ff] w-8 text-right">
            {minValueScore}
          </span>
        </div>
      </div>

      {error && (
        <div className="mb-6">
          <ErrorMessage error={error as Error} message="Failed to load NFL data. Please try again." />
        </div>
      )}

      {viewMode === 'best-bets' && !isLoading && !error && (
        <p className="text-xs text-slate-500 font-mono mb-4">
          {qualifyingCount} qualifying pick{qualifyingCount === 1 ? '' : 's'} at {minValueScore}+ value score
        </p>
      )}

      <div className="grid gap-6 lg:grid-cols-2">
        {isLoading ? (
          <>
            <GameCardSkeleton />
            <GameCardSkeleton />
            <GameCardSkeleton />
            <GameCardSkeleton />
          </>
        ) : displayedGames.length === 0 ? (
          <EmptyStatePanel />
        ) : (
          displayedGames.map((game) => <NFLGameCard key={game.game_id} game={game} />)
        )}
      </div>
    </div>
  );
}
