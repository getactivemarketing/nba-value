import { clsx } from 'clsx';
import type { PlayerProp } from '@/lib/api';

interface PlayerPropsProps {
  props: PlayerProp[];
  homeTeam: string;
  awayTeam: string;
}

// Format prop type for display
function formatPropType(propType: string): string {
  const labels: Record<string, string> = {
    points: 'PTS',
    rebounds: 'REB',
    assists: 'AST',
    threes: '3PM',
    pra: 'PRA',
    points_rebounds: 'P+R',
    points_assists: 'P+A',
    rebounds_assists: 'R+A',
    steals: 'STL',
    blocks: 'BLK',
    turnovers: 'TO',
  };
  return labels[propType] || propType.toUpperCase();
}

// Format odds for display
function formatOdds(odds: number | null): string {
  if (odds === null) return '-';
  // Convert decimal to American
  if (odds >= 2.0) {
    return `+${Math.round((odds - 1) * 100)}`;
  } else {
    return `${Math.round(-100 / (odds - 1))}`;
  }
}

// Group props by player
function groupPropsByPlayer(props: PlayerProp[]): Map<string, PlayerProp[]> {
  const grouped = new Map<string, PlayerProp[]>();

  for (const prop of props) {
    const existing = grouped.get(prop.player_name) || [];
    existing.push(prop);
    grouped.set(prop.player_name, existing);
  }

  return grouped;
}

export function PlayerProps({ props, homeTeam: _homeTeam, awayTeam: _awayTeam }: PlayerPropsProps) {
  if (!props || props.length === 0) {
    return (
      <div className="bg-gray-50 rounded-lg p-4 text-center text-gray-500 text-sm">
        No player props available for this game
      </div>
    );
  }

  const groupedProps = groupPropsByPlayer(props);

  // Sort players by number of props (most props first) and then alphabetically
  const sortedPlayers = Array.from(groupedProps.entries()).sort((a, b) => {
    if (b[1].length !== a[1].length) return b[1].length - a[1].length;
    return a[0].localeCompare(b[0]);
  });

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-4">
      <div className="flex items-center justify-between mb-3">
        <h4 className="text-sm font-semibold text-gray-900">Player Props</h4>
        <span className="text-xs text-gray-500">
          {props.length} props from {props[0]?.book || 'sportsbook'}
        </span>
      </div>

      <div className="space-y-3 max-h-64 overflow-y-auto">
        {sortedPlayers.slice(0, 10).map(([playerName, playerProps]) => (
          <div key={playerName} className="border-b border-gray-100 pb-2 last:border-0">
            <div className="text-sm font-medium text-gray-800 mb-1">
              {playerName}
            </div>
            <div className="flex flex-wrap gap-2">
              {playerProps.map((prop, idx) => (
                <div
                  key={`${prop.prop_type}-${idx}`}
                  className="inline-flex items-center gap-1 text-xs bg-gray-50 rounded px-2 py-1"
                >
                  <span className="font-medium text-gray-600">
                    {formatPropType(prop.prop_type)}
                  </span>
                  <span className="text-gray-900 font-semibold">
                    {prop.line}
                  </span>
                  <span className="text-gray-400">|</span>
                  <span className={clsx(
                    'text-[10px]',
                    prop.over_odds && prop.over_odds >= 2 ? 'text-green-600' : 'text-gray-500'
                  )}>
                    o{formatOdds(prop.over_odds)}
                  </span>
                  <span className={clsx(
                    'text-[10px]',
                    prop.under_odds && prop.under_odds >= 2 ? 'text-green-600' : 'text-gray-500'
                  )}>
                    u{formatOdds(prop.under_odds)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

      {sortedPlayers.length > 10 && (
        <div className="mt-2 text-center text-xs text-gray-400">
          +{sortedPlayers.length - 10} more players
        </div>
      )}
    </div>
  );
}

// Compact summary version for GameCard
export function PlayerPropsPreview({ propsCount }: { propsCount: number }) {
  if (propsCount === 0) return null;

  return (
    <span className="inline-flex items-center gap-1 text-xs text-gray-500">
      <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
      </svg>
      {propsCount} props
    </span>
  );
}
