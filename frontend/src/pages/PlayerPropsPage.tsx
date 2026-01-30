import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useUpcomingGames } from '@/hooks/useMarkets';
import { usePlayerProps } from '@/hooks/usePlayerProps';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { ErrorMessage } from '@/components/ui/ErrorMessage';
import { clsx } from 'clsx';
import { api } from '@/lib/api';
import type { PlayerProp, ScoredProp, TopPropsResponse } from '@/lib/api';

// Format prop type for display
function formatPropType(propType: string): string {
  const labels: Record<string, string> = {
    points: 'Points',
    rebounds: 'Rebounds',
    assists: 'Assists',
    threes: '3-Pointers',
    pra: 'Pts+Reb+Ast',
    points_rebounds: 'Pts+Reb',
    points_assists: 'Pts+Ast',
    rebounds_assists: 'Reb+Ast',
    steals: 'Steals',
    blocks: 'Blocks',
    turnovers: 'Turnovers',
  };
  return labels[propType] || propType.charAt(0).toUpperCase() + propType.slice(1);
}

// Format decimal odds to American
function formatOdds(odds: number | null): string {
  if (odds === null) return '-';
  if (odds >= 2.0) {
    return `+${Math.round((odds - 1) * 100)}`;
  } else {
    return `${Math.round(-100 / (odds - 1))}`;
  }
}

// Format time until game
function formatTimeToTip(tipTime: string): string {
  const tip = new Date(tipTime);
  const now = new Date();
  const diffMs = tip.getTime() - now.getTime();
  const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
  const diffMins = Math.floor((diffMs % (1000 * 60 * 60)) / (1000 * 60));

  if (diffHours < 0) return 'Live';
  if (diffHours === 0) return `${diffMins}m`;
  if (diffHours < 24) return `${diffHours}h ${diffMins}m`;
  return tip.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' });
}

// Hook for top props
function useTopProps(limit: number = 10, minScore: number = 50) {
  return useQuery<TopPropsResponse>({
    queryKey: ['topProps', limit, minScore],
    queryFn: () => api.getTopProps(limit, minScore),
    staleTime: 300000, // 5 minutes
    refetchInterval: 600000, // 10 minutes
  });
}

// Top Picks Card Component
function TopPickCard({ prop }: { prop: ScoredProp }) {
  const isOver = prop.recommendation === 'OVER';
  const relevantOdds = isOver ? prop.over_odds : prop.under_odds;

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-4 hover:shadow-md transition-shadow">
      <div className="flex items-start justify-between mb-2">
        <div>
          <div className="font-semibold text-gray-900">{prop.player_name}</div>
          <div className="text-sm text-gray-500">{formatPropType(prop.prop_type)}</div>
        </div>
        <div className={clsx(
          'px-2 py-1 rounded-full text-sm font-bold',
          prop.value_score >= 70 ? 'bg-amber-100 text-amber-700' :
          prop.value_score >= 60 ? 'bg-green-100 text-green-700' :
          'bg-blue-100 text-blue-700'
        )}>
          {prop.value_score.toFixed(0)}
        </div>
      </div>

      <div className="flex items-center gap-3 mb-3">
        <div className={clsx(
          'px-3 py-1.5 rounded-lg font-bold text-lg',
          isOver ? 'bg-green-50 text-green-700' : 'bg-red-50 text-red-700'
        )}>
          {prop.recommendation} {prop.line}
        </div>
        {relevantOdds && (
          <div className={clsx(
            'text-sm font-medium',
            relevantOdds >= 2 ? 'text-green-600' : 'text-gray-600'
          )}>
            {formatOdds(relevantOdds)}
          </div>
        )}
      </div>

      <div className="text-sm text-gray-600 mb-2">
        {prop.reasoning}
      </div>

      <div className="flex items-center gap-4 text-xs text-gray-500">
        <span>Season Avg: {prop.season_avg?.toFixed(1) ?? '-'}</span>
        <span>Edge: {prop.edge_pct ? `${prop.edge_pct > 0 ? '+' : ''}${prop.edge_pct.toFixed(1)}%` : '-'}</span>
        <span className="capitalize">{prop.book}</span>
      </div>
    </div>
  );
}

// Game card with expandable props
function GamePropsCard({
  gameId,
  homeTeam,
  awayTeam,
  tipTime,
  propTypeFilter,
}: {
  gameId: string;
  homeTeam: string;
  awayTeam: string;
  tipTime: string;
  propTypeFilter: string;
}) {
  const [expanded, setExpanded] = useState(false);
  const { data, isLoading, error } = usePlayerProps(expanded ? gameId : null);

  // Filter and group props
  const filteredProps = data?.props?.filter(p =>
    propTypeFilter === 'all' || p.prop_type === propTypeFilter
  ) || [];

  // Group by player
  const groupedByPlayer = new Map<string, PlayerProp[]>();
  for (const prop of filteredProps) {
    const existing = groupedByPlayer.get(prop.player_name) || [];
    existing.push(prop);
    groupedByPlayer.set(prop.player_name, existing);
  }

  // Sort players by total props available
  const sortedPlayers = Array.from(groupedByPlayer.entries())
    .sort((a, b) => b[1].length - a[1].length);

  return (
    <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
      {/* Game Header */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full px-4 py-3 flex items-center justify-between hover:bg-gray-50 transition-colors"
      >
        <div className="flex items-center gap-4">
          <div className="text-left">
            <div className="font-semibold text-gray-900">
              {awayTeam} @ {homeTeam}
            </div>
            <div className="text-sm text-gray-500">
              {formatTimeToTip(tipTime)}
            </div>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {data?.props && data.props.length > 0 && (
            <span className="text-xs bg-blue-100 text-blue-700 px-2 py-1 rounded-full">
              {data.props.length} props
            </span>
          )}
          <svg
            className={clsx(
              'w-5 h-5 text-gray-400 transition-transform',
              expanded && 'rotate-180'
            )}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </div>
      </button>

      {/* Expanded Props */}
      {expanded && (
        <div className="border-t border-gray-200 px-4 py-3">
          {isLoading && (
            <div className="flex justify-center py-4">
              <LoadingSpinner size="sm" />
            </div>
          )}

          {error && (
            <div className="text-sm text-red-600 py-2">
              Failed to load props
            </div>
          )}

          {!isLoading && !error && sortedPlayers.length === 0 && (
            <div className="text-sm text-gray-500 text-center py-4">
              No player props available for this game yet.
              <br />
              <span className="text-xs">Props are typically available 24-48 hours before tip-off.</span>
            </div>
          )}

          {!isLoading && sortedPlayers.length > 0 && (
            <div className="space-y-3">
              {sortedPlayers.map(([playerName, props]) => (
                <div key={playerName} className="border-b border-gray-100 pb-3 last:border-0 last:pb-0">
                  <div className="font-medium text-gray-800 mb-2">{playerName}</div>
                  <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-2">
                    {props.map((prop, idx) => (
                      <div
                        key={`${prop.prop_type}-${idx}`}
                        className="flex items-center justify-between bg-gray-50 rounded px-3 py-2"
                      >
                        <div>
                          <div className="text-xs text-gray-500">{formatPropType(prop.prop_type)}</div>
                          <div className="font-semibold text-gray-900">{prop.line}</div>
                        </div>
                        <div className="text-right text-sm">
                          <div className={clsx(
                            prop.over_odds && prop.over_odds >= 2 ? 'text-green-600' : 'text-gray-600'
                          )}>
                            O {formatOdds(prop.over_odds)}
                          </div>
                          <div className={clsx(
                            prop.under_odds && prop.under_odds >= 2 ? 'text-green-600' : 'text-gray-600'
                          )}>
                            U {formatOdds(prop.under_odds)}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export function PlayerPropsPage() {
  const [propTypeFilter, setPropTypeFilter] = useState('all');
  const { data: games, isLoading: gamesLoading, error: gamesError } = useUpcomingGames(48);
  const { data: topProps, isLoading: topLoading } = useTopProps(10, 50);

  const propTypes = [
    { value: 'all', label: 'All Props' },
    { value: 'points', label: 'Points' },
    { value: 'rebounds', label: 'Rebounds' },
    { value: 'assists', label: 'Assists' },
    { value: 'threes', label: '3-Pointers' },
  ];

  if (gamesLoading) {
    return (
      <div className="flex justify-center py-12">
        <LoadingSpinner size="lg" />
      </div>
    );
  }

  if (gamesError) {
    return <ErrorMessage error={gamesError as Error} />;
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Player Props</h1>
        <p className="text-gray-500 mt-1">
          Top value props based on season averages
        </p>
      </div>

      {/* Top Picks Section */}
      <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-xl p-6">
        <div className="flex items-center gap-2 mb-4">
          <svg className="w-6 h-6 text-amber-400" fill="currentColor" viewBox="0 0 20 20">
            <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
          </svg>
          <h2 className="text-xl font-bold text-white">Top Picks</h2>
        </div>

        {topLoading ? (
          <div className="flex justify-center py-8">
            <LoadingSpinner size="md" />
          </div>
        ) : topProps?.props && topProps.props.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {topProps.props.slice(0, 6).map((prop, idx) => (
              <TopPickCard key={`${prop.player_name}-${prop.prop_type}-${idx}`} prop={prop} />
            ))}
          </div>
        ) : (
          <div className="bg-slate-700/50 rounded-lg p-6 text-center">
            <p className="text-slate-300">No value props found yet.</p>
            <p className="text-slate-400 text-sm mt-1">
              Props are analyzed once player data is loaded.
            </p>
          </div>
        )}

        {topProps?.props && topProps.props.length > 6 && (
          <div className="mt-4 text-center">
            <span className="text-slate-400 text-sm">
              +{topProps.props.length - 6} more value props below
            </span>
          </div>
        )}
      </div>

      {/* All Top Props Table (if more than 6) */}
      {topProps?.props && topProps.props.length > 6 && (
        <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
          <div className="px-4 py-3 border-b border-gray-200">
            <h3 className="font-semibold text-gray-900">All Value Props</h3>
          </div>
          <div className="divide-y divide-gray-100">
            {topProps.props.slice(6).map((prop, idx) => (
              <div key={`${prop.player_name}-${prop.prop_type}-${idx}`} className="px-4 py-3 flex items-center justify-between">
                <div className="flex items-center gap-4">
                  <div className={clsx(
                    'w-10 h-10 rounded-full flex items-center justify-center text-sm font-bold',
                    prop.value_score >= 70 ? 'bg-amber-100 text-amber-700' :
                    prop.value_score >= 60 ? 'bg-green-100 text-green-700' :
                    'bg-blue-100 text-blue-700'
                  )}>
                    {prop.value_score.toFixed(0)}
                  </div>
                  <div>
                    <div className="font-medium text-gray-900">{prop.player_name}</div>
                    <div className="text-sm text-gray-500">{formatPropType(prop.prop_type)}</div>
                  </div>
                </div>
                <div className="flex items-center gap-4">
                  <div className={clsx(
                    'px-2 py-1 rounded text-sm font-semibold',
                    prop.recommendation === 'OVER' ? 'bg-green-50 text-green-700' : 'bg-red-50 text-red-700'
                  )}>
                    {prop.recommendation} {prop.line}
                  </div>
                  <div className="text-sm text-gray-500 w-24 text-right">
                    Avg: {prop.season_avg?.toFixed(1) ?? '-'}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Filter & Browse Section */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <h3 className="text-lg font-semibold text-gray-900">Browse All Props by Game</h3>
        <div className="flex gap-2">
          {propTypes.map((type) => (
            <button
              key={type.value}
              onClick={() => setPropTypeFilter(type.value)}
              className={clsx(
                'px-3 py-1.5 text-sm font-medium rounded-lg transition-colors',
                propTypeFilter === type.value
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              )}
            >
              {type.label}
            </button>
          ))}
        </div>
      </div>

      {/* Games List */}
      {games && games.length > 0 ? (
        <div className="space-y-3">
          {games.map((game) => (
            <GamePropsCard
              key={game.game_id}
              gameId={game.game_id}
              homeTeam={game.home_team}
              awayTeam={game.away_team}
              tipTime={game.tip_time}
              propTypeFilter={propTypeFilter}
            />
          ))}
        </div>
      ) : (
        <div className="bg-white rounded-lg border border-gray-200 p-8 text-center">
          <p className="text-gray-500">No upcoming games found</p>
        </div>
      )}
    </div>
  );
}
