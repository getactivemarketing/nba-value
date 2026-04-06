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
    <div className="bg-[#191c22] rounded-xl overflow-hidden animate-pulse">
      <div className="p-5">
        <div className="flex justify-between items-start mb-5">
          <div className="flex items-center gap-5">
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
          </div>
          <div className="flex flex-col items-end gap-1">
            <div className="h-3 bg-[#272a31] rounded w-16" />
            <div className="h-8 bg-[#272a31] rounded-full w-24" />
          </div>
        </div>
        <div className="h-1.5 bg-[#272a31] rounded-full w-full mb-5" />
        <div className="grid grid-cols-2 gap-3">
          <div className="bg-[#0b0e14] p-3.5 rounded-lg h-20" />
          <div className="bg-[#0b0e14] p-3.5 rounded-lg h-20" />
        </div>
      </div>
    </div>
  );
}

function TopPicksPanel({ picks }: { picks: MLBTopPick[] }) {
  if (picks.length === 0) {
    return (
      <div className="bg-[#191c22] rounded-xl p-4">
        <p className="text-slate-500 text-center text-sm font-mono">No high-value picks for today</p>
      </div>
    );
  }

  return (
    <div className="bg-[#191c22] rounded-xl overflow-hidden">
      <div className="bg-[#66f796]/5 border-b border-[#66f796]/10 px-4 py-3">
        <h3 className="text-[10px] text-[#66f796] font-bold uppercase tracking-widest">Top Value Picks</h3>
      </div>
      <div className="divide-y divide-slate-700/30">
        {picks.slice(0, 5).map((pick, idx) => (
          <div key={idx} className="p-3 hover:bg-[#0b0e14] transition-colors">
            <div className="flex justify-between items-start">
              <div>
                <p className="text-sm font-bold text-txt-primary">
                  {pick.away_team} @ {pick.home_team}
                </p>
                <p className="text-xs text-slate-400 font-mono mt-0.5">
                  {pick.bet_type === 'moneyline'
                    ? `${pick.team} ML`
                    : pick.bet_type === 'runline'
                    ? `${pick.team} ${pick.line && pick.line > 0 ? '+' : ''}${pick.line}`
                    : `${pick.bet_type === 'over' ? 'Over' : 'Under'} ${pick.line}`}
                  {' '}
                  <span className="text-slate-500">({pick.odds_american > 0 ? '+' : ''}{pick.odds_american})</span>
                </p>
                {pick.home_starter && pick.away_starter && (
                  <p className="text-[10px] text-slate-500 mt-1 font-mono">
                    {pick.away_starter} vs {pick.home_starter}
                  </p>
                )}
              </div>
              <div className="text-right flex-shrink-0 ml-3">
                <span className={`inline-block px-2 py-1 rounded text-sm font-black font-mono ${
                  pick.value_score >= 70
                    ? 'bg-[#66f796]/10 text-[#66f796]'
                    : 'bg-[#a4e6ff]/10 text-[#a4e6ff]'
                }`}>
                  {pick.value_score.toFixed(0)}
                </span>
                <p className="text-[10px] text-slate-500 mt-1 font-mono font-bold">
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

  // Count NRFI results for completed games
  const completedGames = games.filter(g => g.status === 'final');
  const nrfiGames = completedGames.filter(
    g => g.home_first_inning_runs !== null &&
         g.away_first_inning_runs !== null &&
         g.home_first_inning_runs + g.away_first_inning_runs === 0
  );
  const yrfiGames = completedGames.filter(
    g => g.home_first_inning_runs !== null &&
         g.away_first_inning_runs !== null &&
         g.home_first_inning_runs + g.away_first_inning_runs > 0
  );

  return (
    <div className="max-w-6xl mx-auto px-4 py-6">
      {/* Header */}
      <div className="mb-6">
        <div className="flex items-center gap-3 mb-1">
          <h1 className="text-2xl font-black text-txt-primary font-mono tracking-tight">
            MLB TERMINAL <span className="text-[#a4e6ff]">NRFI</span>
          </h1>
          <div className="flex items-center gap-1.5">
            <div className="w-1.5 h-1.5 rounded-full bg-[#66f796] animate-pulse" />
            <span className="text-[10px] text-slate-500 font-bold uppercase tracking-widest font-mono">Live Feed</span>
          </div>
        </div>
        <p className="text-slate-500 text-sm font-mono">
          High-density first inning probability intelligence
        </p>
      </div>

      {/* Date selector */}
      <div className="flex gap-2 mb-6 overflow-x-auto pb-2">
        {dates.map((date, idx) => {
          const label = formatDateLabel(date);
          const isSelected = date.toDateString() === selectedDate.toDateString();
          const isDisabled = false;

          return (
            <button
              key={idx}
              onClick={() => setSelectedDate(date)}
              disabled={isDisabled}
              className={`flex-shrink-0 px-4 py-2 rounded-lg text-center transition-all ${
                isSelected
                  ? 'bg-[#a4e6ff] text-[#0b0e14] font-bold'
                  : isDisabled
                  ? 'bg-[#191c22] text-slate-600 cursor-not-allowed'
                  : 'bg-[#191c22] text-slate-400 hover:bg-[#272a31] hover:text-txt-primary'
              }`}
            >
              <div className="text-[10px] font-bold font-mono uppercase tracking-widest">{label.day}</div>
              <div className="text-sm font-mono">{label.date}</div>
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
            <div className="bg-[#191c22] rounded-xl p-8 text-center">
              <p className="text-slate-500 font-mono text-sm">No games scheduled for this date</p>
            </div>
          ) : (
            games.map((game) => (
              <MLBGameCard key={game.game_id} game={game} />
            ))
          )}
        </div>

        {/* Sidebar */}
        <div className="space-y-4">
          {/* NRFI Stats Summary (for days with completed games) */}
          {completedGames.length > 0 && (nrfiGames.length > 0 || yrfiGames.length > 0) && (
            <div className="bg-[#191c22] rounded-xl overflow-hidden">
              <div className="bg-[#a4e6ff]/5 border-b border-[#a4e6ff]/10 px-4 py-3">
                <h3 className="text-[10px] text-[#a4e6ff] font-bold uppercase tracking-widest">
                  First Inning Stats
                </h3>
              </div>
              <div className="p-4">
                <div className="grid grid-cols-3 gap-3 text-center">
                  <div>
                    <div className="text-[10px] text-slate-500 font-bold uppercase tracking-widest mb-1">NRFI</div>
                    <div className="text-xl font-black font-mono text-[#66f796]">{nrfiGames.length}</div>
                  </div>
                  <div>
                    <div className="text-[10px] text-slate-500 font-bold uppercase tracking-widest mb-1">YRFI</div>
                    <div className="text-xl font-black font-mono text-[#f59e0b]">{yrfiGames.length}</div>
                  </div>
                  <div>
                    <div className="text-[10px] text-slate-500 font-bold uppercase tracking-widest mb-1">Rate</div>
                    <div className="text-xl font-black font-mono text-[#a4e6ff]">
                      {((nrfiGames.length / (nrfiGames.length + yrfiGames.length)) * 100).toFixed(0)}%
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Top picks */}
          {!isPast && (
            <>
              {picksLoading ? (
                <div className="bg-[#191c22] rounded-xl p-8">
                  <LoadingSpinner />
                </div>
              ) : (
                <TopPicksPanel picks={topPicks} />
              )}
            </>
          )}

          {/* Legend */}
          <div className="bg-[#191c22] rounded-xl p-4">
            <h3 className="text-[10px] text-slate-500 font-bold uppercase tracking-widest mb-3">
              Value Score Guide
            </h3>
            <div className="space-y-2 text-sm">
              <div className="flex items-center gap-2">
                <span className="w-10 h-5 bg-[#66f796]/10 border border-[#66f796]/30 rounded flex items-center justify-center text-[10px] font-black font-mono text-[#66f796]">
                  70+
                </span>
                <span className="text-slate-400 text-xs">Strong Value</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-10 h-5 bg-[#a4e6ff]/10 border border-[#a4e6ff]/30 rounded flex items-center justify-center text-[10px] font-black font-mono text-[#a4e6ff]">
                  60+
                </span>
                <span className="text-slate-400 text-xs">Moderate Value</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-10 h-5 bg-[#32353c]/50 border border-[#32353c] rounded flex items-center justify-center text-[10px] font-black font-mono text-slate-500">
                  &lt;60
                </span>
                <span className="text-slate-400 text-xs">Low Value</span>
              </div>
            </div>
            <div className="mt-4 pt-3 border-t border-slate-700/30">
              <p className="text-[10px] text-slate-500 font-mono leading-relaxed">
                QS = Quality Score (pitcher rating 0-100)
                <br />
                PF = Park Factor (run environment)
                <br />
                NRFI = No Runs First Inning
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
