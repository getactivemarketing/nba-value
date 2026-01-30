import { useState } from 'react';
import { useUpcomingGames } from '@/hooks/useMarkets';
import { usePlayerProps } from '@/hooks/usePlayerProps';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { ErrorMessage } from '@/components/ui/ErrorMessage';
import { clsx } from 'clsx';
import type { PlayerProp } from '@/lib/api';

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
  const { data: games, isLoading, error } = useUpcomingGames(48); // Next 48 hours

  const propTypes = [
    { value: 'all', label: 'All Props' },
    { value: 'points', label: 'Points' },
    { value: 'rebounds', label: 'Rebounds' },
    { value: 'assists', label: 'Assists' },
    { value: 'threes', label: '3-Pointers' },
  ];

  if (isLoading) {
    return (
      <div className="flex justify-center py-12">
        <LoadingSpinner size="lg" />
      </div>
    );
  }

  if (error) {
    return <ErrorMessage error={error as Error} />;
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Player Props</h1>
          <p className="text-gray-500 mt-1">
            View player prop lines for upcoming games
          </p>
        </div>

        {/* Filter */}
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

      {/* Info Banner */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <div className="flex items-start gap-3">
          <svg className="w-5 h-5 text-blue-600 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <div className="text-sm text-blue-800">
            <p className="font-medium">Player Props Data</p>
            <p className="mt-1">
              Click on a game to view available player props. Props are typically available 24-48 hours before tip-off and update periodically.
            </p>
          </div>
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
