import { useState } from 'react';
import { getTeamLogo, getTeamColor } from '@/lib/teamLogos';
import type { HistoricalGame } from '@/lib/api';

interface HistoricalGameCardProps {
  game: HistoricalGame;
}

// Team logo component with fallback
function TeamLogo({ team, size = 48 }: { team: string; size?: number }) {
  const [imgError, setImgError] = useState(false);
  const logoUrl = getTeamLogo(team);
  const teamColor = getTeamColor(team);

  if (imgError || !logoUrl) {
    return (
      <div
        className="flex items-center justify-center rounded-full font-bold text-white"
        style={{
          width: size,
          height: size,
          backgroundColor: teamColor,
          fontSize: size * 0.35
        }}
      >
        {team.slice(0, 3)}
      </div>
    );
  }

  return (
    <img
      src={logoUrl}
      alt={team}
      width={size}
      height={size}
      className="object-contain"
      onError={() => setImgError(true)}
    />
  );
}

function formatDate(dateStr: string): string {
  const d = new Date(dateStr + 'T12:00:00');
  return d.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' });
}

function formatSpread(spread: number | null): string {
  if (spread === null) return '-';
  return spread > 0 ? `+${spread}` : `${spread}`;
}

function getResultBadge(result: string | null): { text: string; color: string } | null {
  if (!result) return null;

  switch (result) {
    case 'home_cover':
      return { text: 'HOME COVER', color: 'bg-emerald-100 text-emerald-700' };
    case 'away_cover':
      return { text: 'AWAY COVER', color: 'bg-blue-100 text-blue-700' };
    case 'over':
      return { text: 'OVER', color: 'bg-orange-100 text-orange-700' };
    case 'under':
      return { text: 'UNDER', color: 'bg-purple-100 text-purple-700' };
    case 'push':
      return { text: 'PUSH', color: 'bg-gray-100 text-gray-600' };
    default:
      return null;
  }
}

export function HistoricalGameCard({ game }: HistoricalGameCardProps) {
  const homeWon = game.home_score > game.away_score;
  const spreadResult = getResultBadge(game.spread_result);
  const totalResult = getResultBadge(game.total_result);

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
      {/* Dark Header */}
      <div className="bg-slate-700 text-white px-4 py-2 text-sm font-medium tracking-wide flex justify-between items-center">
        <span>{game.away_team} @ {game.home_team}</span>
        <span className="text-gray-300 text-xs">FINAL</span>
      </div>

      {/* Main Content */}
      <div className="p-4">
        {/* Teams and Score Row */}
        <div className="flex items-center justify-between">
          {/* Away Team */}
          <div className="flex items-center gap-3">
            <TeamLogo team={game.away_team} size={56} />
            <div>
              <div className="text-xl font-bold text-gray-900">{game.away_team}</div>
              <div className={`text-3xl font-bold ${!homeWon ? 'text-gray-900' : 'text-gray-400'}`}>
                {game.away_score}
              </div>
            </div>
          </div>

          {/* Center - Date and Results */}
          <div className="text-center px-4">
            <div className="text-sm font-medium text-gray-900">{formatDate(game.game_date)}</div>
            <div className="flex flex-col gap-1 mt-2">
              {spreadResult && (
                <span className={`text-xs px-2 py-0.5 rounded ${spreadResult.color}`}>
                  {spreadResult.text}
                </span>
              )}
              {totalResult && (
                <span className={`text-xs px-2 py-0.5 rounded ${totalResult.color}`}>
                  {totalResult.text}
                </span>
              )}
            </div>
          </div>

          {/* Home Team */}
          <div className="flex items-center gap-3">
            <div className="text-right">
              <div className="text-xl font-bold text-gray-900">{game.home_team}</div>
              <div className={`text-3xl font-bold ${homeWon ? 'text-gray-900' : 'text-gray-400'}`}>
                {game.home_score}
              </div>
            </div>
            <TeamLogo team={game.home_team} size={56} />
          </div>
        </div>

        {/* Betting Results */}
        <div className="mt-4 pt-4 border-t border-gray-100">
          <div className="grid grid-cols-2 gap-4 text-sm">
            {/* Spread Result */}
            <div className="bg-gray-50 rounded-lg p-3">
              <div className="text-xs text-gray-500 mb-1">SPREAD</div>
              <div className="font-semibold text-gray-900">
                {game.closing_spread !== null ? (
                  <>
                    {game.home_team} {formatSpread(game.closing_spread)}
                  </>
                ) : (
                  'N/A'
                )}
              </div>
              {game.spread_margin !== null && (
                <div className="text-xs text-gray-500 mt-1">
                  {game.spread_result === 'home_cover' ? game.home_team : game.away_team} covered by {Math.abs(game.spread_margin).toFixed(1)}
                </div>
              )}
            </div>

            {/* Total Result */}
            <div className="bg-gray-50 rounded-lg p-3">
              <div className="text-xs text-gray-500 mb-1">TOTAL</div>
              <div className="font-semibold text-gray-900">
                {game.closing_total !== null ? (
                  <>
                    o/u {game.closing_total}
                  </>
                ) : (
                  'N/A'
                )}
              </div>
              {game.total_margin !== null && (
                <div className="text-xs text-gray-500 mt-1">
                  Actual: {game.total_score} ({game.total_result === 'over' ? 'Over' : game.total_result === 'under' ? 'Under' : 'Push'} by {Math.abs(game.total_margin).toFixed(1)})
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
