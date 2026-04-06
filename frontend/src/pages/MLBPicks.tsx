import { useState, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { mlbApi, type MLBTopPick } from '@/lib/mlbApi';
import { MLBGameCard } from '@/components/mlb/MLBGameCard';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { ErrorMessage } from '@/components/ui/ErrorMessage';

// Generate dates for the date picker (similar to NBA)
function getDateRange(): Date[] {
  const dates: Date[] = [];
  const today = new Date();

  for (let i = -3; i <= 4; i++) {
    const date = new Date(today);
    date.setDate(today.getDate() + i);
    dates.push(date);
  }

  return dates;
}

function formatDateLabel(date: Date): { day: string; date: string } {
  const today = new Date();
  const isToday = date.toDateString() === today.toDateString();

  if (isToday) {
    return {
      day: 'TODAY',
      date: date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }).toUpperCase(),
    };
  }

  return {
    day: date.toLocaleDateString('en-US', { weekday: 'short' }).toUpperCase(),
    date: date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }).toUpperCase(),
  };
}

function formatDateParam(date: Date): string {
  return date.toISOString().split('T')[0];
}

function isPastDate(date: Date): boolean {
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const check = new Date(date);
  check.setHours(0, 0, 0, 0);
  return check < today;
}

function GameCardSkeleton() {
  return (
    <div className="bg-tru-card rounded-xl border border-tru-border overflow-hidden animate-pulse">
      <div className="bg-tru-surface h-10" />
      <div className="p-4">
        <div className="flex justify-between items-center">
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 bg-tru-border rounded-full" />
            <div>
              <div className="h-4 bg-tru-border rounded w-12 mb-2" />
              <div className="h-6 bg-tru-border rounded w-20" />
            </div>
          </div>
          <div className="text-center">
            <div className="h-4 bg-tru-border rounded w-16 mb-2 mx-auto" />
            <div className="h-6 bg-tru-border rounded w-12 mx-auto" />
          </div>
          <div className="flex items-center gap-3">
            <div className="text-right">
              <div className="h-4 bg-tru-border rounded w-12 mb-2 ml-auto" />
              <div className="h-6 bg-tru-border rounded w-20" />
            </div>
            <div className="w-12 h-12 bg-tru-border rounded-full" />
          </div>
        </div>
      </div>
    </div>
  );
}

function TopPicksPanel({ picks }: { picks: MLBTopPick[] }) {
  if (picks.length === 0) {
    return (
      <div className="bg-tru-card rounded-xl border border-tru-border p-4">
        <p className="text-txt-muted text-center">No high-value picks for today</p>
      </div>
    );
  }

  return (
    <div className="bg-tru-card rounded-xl border border-tru-border overflow-hidden">
      <div className="bg-value-hot/10 border-b border-value-hot/20 px-4 py-2">
        <h3 className="font-semibold text-value-hot">Top Value Picks</h3>
      </div>
      <div className="divide-y divide-tru-border">
        {picks.slice(0, 5).map((pick, idx) => (
          <div key={idx} className="p-3 hover:bg-tru-surface transition-colors">
            <div className="flex justify-between items-start">
              <div>
                <p className="text-sm font-medium text-txt-primary">
                  {pick.away_team} @ {pick.home_team}
                </p>
                <p className="text-sm text-txt-secondary">
                  {pick.bet_type === 'moneyline'
                    ? `${pick.team} ML`
                    : pick.bet_type === 'runline'
                    ? `${pick.team} ${pick.line && pick.line > 0 ? '+' : ''}${pick.line}`
                    : `${pick.bet_type === 'over' ? 'Over' : 'Under'} ${pick.line}`}
                  {' '}
                  <span className="text-txt-muted font-mono">({pick.odds_american > 0 ? '+' : ''}{pick.odds_american})</span>
                </p>
                {pick.home_starter && pick.away_starter && (
                  <p className="text-xs text-txt-muted mt-1">
                    {pick.away_starter} vs {pick.home_starter}
                  </p>
                )}
              </div>
              <div className="text-right">
                <span className={`inline-block px-2 py-1 rounded text-sm font-bold font-mono ${
                  pick.value_score >= 70
                    ? 'bg-value-hot/10 text-value-hot'
                    : 'bg-value-warm/10 text-value-warm'
                }`}>
                  {pick.value_score.toFixed(0)}
                </span>
                <p className="text-xs text-txt-muted mt-1 font-mono">
                  +{(pick.edge * 100).toFixed(1)}% edge
                </p>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export function MLBPicks() {
  const [selectedDate, setSelectedDate] = useState<Date>(new Date());
  const dates = useMemo(() => getDateRange(), []);

  const dateParam = formatDateParam(selectedDate);
  const isPast = isPastDate(selectedDate);

  // Fetch games
  const {
    data: gamesData,
    isLoading: gamesLoading,
    error: gamesError,
  } = useQuery({
    queryKey: ['mlb-games', dateParam],
    queryFn: () => mlbApi.getGames(dateParam),
    refetchInterval: isPast ? false : 60000, // Refresh every minute for today
  });

  // Fetch top picks
  const {
    data: picksData,
    isLoading: picksLoading,
  } = useQuery({
    queryKey: ['mlb-top-picks', dateParam],
    queryFn: () => mlbApi.getTopPicks(65, dateParam),
    enabled: !isPast,
  });

  const games = gamesData?.games || [];
  const topPicks = picksData?.picks || [];

  return (
    <div className="max-w-6xl mx-auto px-4 py-6">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-txt-primary font-display">MLB Picks</h1>
        <p className="text-txt-secondary">Value betting picks powered by our run differential model</p>
      </div>

      {/* Date selector */}
      <div className="flex gap-2 mb-6 overflow-x-auto pb-2">
        {dates.map((date, idx) => {
          const label = formatDateLabel(date);
          const isSelected = date.toDateString() === selectedDate.toDateString();
          const isDisabled = false; // Could disable far future dates

          return (
            <button
              key={idx}
              onClick={() => setSelectedDate(date)}
              disabled={isDisabled}
              className={`flex-shrink-0 px-4 py-2 rounded-lg text-center transition-all ${
                isSelected
                  ? 'bg-accent-cyan text-tru-bg font-semibold'
                  : isDisabled
                  ? 'bg-tru-surface text-txt-muted cursor-not-allowed'
                  : 'bg-tru-surface text-txt-secondary hover:bg-tru-border hover:text-txt-primary'
              }`}
            >
              <div className="text-xs font-medium font-mono">{label.day}</div>
              <div className="text-sm">{label.date}</div>
            </button>
          );
        })}
      </div>

      {gamesError && (
        <ErrorMessage error={gamesError as Error} message="Failed to load games. Please try again." />
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Main games list */}
        <div className="lg:col-span-2 space-y-4">
          {gamesLoading ? (
            <>
              <GameCardSkeleton />
              <GameCardSkeleton />
              <GameCardSkeleton />
            </>
          ) : games.length === 0 ? (
            <div className="bg-tru-card rounded-xl border border-tru-border p-8 text-center">
              <p className="text-txt-muted">No games scheduled for this date</p>
            </div>
          ) : (
            games.map((game) => (
              <MLBGameCard key={game.game_id} game={game} />
            ))
          )}
        </div>

        {/* Sidebar with top picks */}
        <div className="space-y-4">
          {!isPast && (
            <>
              {picksLoading ? (
                <div className="bg-tru-card rounded-xl border border-tru-border p-8">
                  <LoadingSpinner />
                </div>
              ) : (
                <TopPicksPanel picks={topPicks} />
              )}
            </>
          )}

          {/* Legend */}
          <div className="bg-tru-card rounded-xl border border-tru-border p-4">
            <h3 className="font-semibold text-txt-primary mb-3">Value Score Guide</h3>
            <div className="space-y-2 text-sm">
              <div className="flex items-center gap-2">
                <span className="w-8 h-5 bg-value-hot/10 rounded flex items-center justify-center text-xs font-bold font-mono text-value-hot">
                  70+
                </span>
                <span className="text-txt-secondary">Strong Value</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-8 h-5 bg-value-warm/10 rounded flex items-center justify-center text-xs font-bold font-mono text-value-warm">
                  65+
                </span>
                <span className="text-txt-secondary">Good Value</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-8 h-5 bg-tru-surface rounded flex items-center justify-center text-xs font-bold font-mono text-txt-muted">
                  &lt;65
                </span>
                <span className="text-txt-secondary">Low Value</span>
              </div>
            </div>
            <div className="mt-4 pt-3 border-t border-tru-border">
              <p className="text-xs text-txt-muted font-mono">
                QS = Quality Score (pitcher rating 0-100)
                <br />
                PF = Park Factor (run environment)
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
