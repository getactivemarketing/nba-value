import type { MLBGame, ValueBetInfo } from '@/lib/mlbApi';
import { getTeamInfo, formatOdds } from '@/lib/mlbApi';
import { PitcherMatchup } from './PitcherMatchup';

interface MLBGameCardProps {
  game: MLBGame;
}

function ValueBadge({ value }: { value: ValueBetInfo }) {
  const scoreColor = value.value_score >= 70
    ? 'bg-green-100 text-green-800 border-green-200'
    : value.value_score >= 60
    ? 'bg-yellow-100 text-yellow-800 border-yellow-200'
    : 'bg-gray-100 text-gray-800 border-gray-200';

  const label = value.market_type === 'moneyline'
    ? `${value.team} ML`
    : value.market_type === 'runline'
    ? `${value.team} ${value.line && value.line > 0 ? '+' : ''}${value.line}`
    : value.bet_type === 'over'
    ? `O ${value.line}`
    : `U ${value.line}`;

  return (
    <div className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-lg border ${scoreColor}`}>
      <span className="font-semibold">{label}</span>
      <span className="text-sm">({formatOdds(value.odds_decimal)})</span>
      <span className="font-bold">{value.value_score.toFixed(0)}</span>
    </div>
  );
}

function TeamDisplay({ abbr, isHome, score }: { abbr: string; isHome: boolean; score?: number | null }) {
  const team = getTeamInfo(abbr);

  return (
    <div className={`flex items-center gap-3 ${isHome ? 'flex-row-reverse' : ''}`}>
      <div
        className="w-12 h-12 rounded-full flex items-center justify-center text-white font-bold text-sm"
        style={{ backgroundColor: team.color }}
      >
        {abbr}
      </div>
      <div className={isHome ? 'text-right' : 'text-left'}>
        <p className="text-xs text-gray-500 uppercase">{isHome ? 'Home' : 'Away'}</p>
        <p className="font-bold text-lg text-gray-900">{team.name}</p>
        {score !== undefined && score !== null && (
          <p className="text-2xl font-bold text-gray-900">{score}</p>
        )}
      </div>
    </div>
  );
}

export function MLBGameCard({ game }: MLBGameCardProps) {
  const gameTime = game.game_time ? new Date(game.game_time) : null;
  const isCompleted = game.status === 'final';
  const isLive = game.status === 'in_progress';

  // Format time
  const timeDisplay = gameTime
    ? gameTime.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' })
    : 'TBD';

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
      {/* Header with venue/weather */}
      <div className="bg-slate-800 text-white px-4 py-2 flex justify-between items-center">
        <div className="flex items-center gap-2">
          {game.context?.venue_name && (
            <span className="text-sm">{game.context.venue_name}</span>
          )}
          {game.context?.is_dome && (
            <span className="text-xs bg-slate-700 px-2 py-0.5 rounded">Dome</span>
          )}
        </div>
        <div className="flex items-center gap-3 text-sm">
          {game.context?.temperature && !game.context.is_dome && (
            <span>{game.context.temperature}°F</span>
          )}
          {game.context?.park_factor && game.context.park_factor !== 1.0 && (
            <span className={game.context.park_factor > 1 ? 'text-red-300' : 'text-blue-300'}>
              PF: {game.context.park_factor.toFixed(2)}
            </span>
          )}
          {isLive && (
            <span className="bg-red-500 px-2 py-0.5 rounded text-xs font-bold animate-pulse">
              LIVE
            </span>
          )}
          {isCompleted && (
            <span className="bg-gray-600 px-2 py-0.5 rounded text-xs">FINAL</span>
          )}
          {!isLive && !isCompleted && (
            <span>{timeDisplay}</span>
          )}
        </div>
      </div>

      {/* Team matchup */}
      <div className="p-4">
        <div className="flex justify-between items-center">
          <TeamDisplay
            abbr={game.away_team}
            isHome={false}
            score={isCompleted || isLive ? game.away_score : undefined}
          />

          <div className="text-center px-4">
            {game.predicted_run_diff !== null && !isCompleted && (
              <div className="mb-2">
                <span className="text-xs text-gray-500 block">Predicted</span>
                <span className={`text-lg font-bold ${
                  game.predicted_run_diff > 0 ? 'text-blue-600' : 'text-red-600'
                }`}>
                  {game.predicted_run_diff > 0 ? '+' : ''}{game.predicted_run_diff.toFixed(1)}
                </span>
              </div>
            )}
            {game.predicted_total !== null && !isCompleted && (
              <div>
                <span className="text-xs text-gray-500 block">Total</span>
                <span className="text-lg font-bold text-gray-700">
                  {game.predicted_total.toFixed(1)}
                </span>
              </div>
            )}
            {isCompleted && game.home_score !== null && game.away_score !== null && (
              <div className="text-sm text-gray-500">
                Total: {game.home_score + game.away_score}
              </div>
            )}
          </div>

          <TeamDisplay
            abbr={game.home_team}
            isHome={true}
            score={isCompleted || isLive ? game.home_score : undefined}
          />
        </div>

        {/* Pitcher matchup */}
        {!isCompleted && (
          <PitcherMatchup
            homePitcher={game.home_starter}
            awayPitcher={game.away_starter}
          />
        )}

        {/* Value bets */}
        {!isCompleted && game.best_bet && (
          <div className="mt-4 pt-3 border-t border-gray-100">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-gray-500 uppercase tracking-wide">
                Best Value
              </span>
              <ValueBadge value={game.best_bet} />
            </div>

            {/* Additional value bets */}
            <div className="mt-2 flex flex-wrap gap-2">
              {game.best_ml && game.best_ml !== game.best_bet && (
                <ValueBadge value={game.best_ml} />
              )}
              {game.best_rl && game.best_rl !== game.best_bet && (
                <ValueBadge value={game.best_rl} />
              )}
              {game.best_total && game.best_total !== game.best_bet && (
                <ValueBadge value={game.best_total} />
              )}
            </div>
          </div>
        )}

        {/* Markets summary */}
        {!isCompleted && game.markets.length > 0 && (
          <div className="mt-3 pt-3 border-t border-gray-100">
            <div className="grid grid-cols-3 gap-2 text-center text-sm">
              {game.markets.find(m => m.market_type === 'moneyline') && (
                <div>
                  <p className="text-xs text-gray-500">Moneyline</p>
                  <p className="font-semibold">
                    {game.markets.find(m => m.market_type === 'moneyline')?.home_odds
                      ? formatOdds(game.markets.find(m => m.market_type === 'moneyline')!.home_odds!)
                      : '-'}
                  </p>
                </div>
              )}
              {game.markets.find(m => m.market_type === 'runline') && (
                <div>
                  <p className="text-xs text-gray-500">Run Line</p>
                  <p className="font-semibold">
                    {game.markets.find(m => m.market_type === 'runline')?.line || '-1.5'}
                  </p>
                </div>
              )}
              {game.markets.find(m => m.market_type === 'total') && (
                <div>
                  <p className="text-xs text-gray-500">Total</p>
                  <p className="font-semibold">
                    O/U {game.markets.find(m => m.market_type === 'total')?.line || '-'}
                  </p>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
