import type { MLBGame, ValueBetInfo } from '@/lib/mlbApi';
import { getTeamInfo, formatOdds } from '@/lib/mlbApi';
import { PitcherMatchup } from './PitcherMatchup';

interface MLBGameCardProps {
  game: MLBGame;
}

function ValueBadge({ value }: { value: ValueBetInfo }) {
  const scoreColor = value.value_score >= 70
    ? 'bg-value-hot/10 text-value-hot border-value-hot/20'
    : value.value_score >= 60
    ? 'bg-value-warm/10 text-value-warm border-value-warm/20'
    : 'bg-tru-surface text-txt-secondary border-tru-border';

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
      <span className="text-sm font-mono">({formatOdds(value.odds_decimal)})</span>
      <span className="font-bold font-mono">{value.value_score.toFixed(0)}</span>
    </div>
  );
}

function TeamDisplay({ abbr, isHome, score }: { abbr: string; isHome: boolean; score?: number | null }) {
  const team = getTeamInfo(abbr);

  return (
    <div className={`flex items-center gap-3 ${isHome ? 'flex-row-reverse' : ''}`}>
      <div
        className="w-12 h-12 rounded-full flex items-center justify-center text-white font-bold text-sm opacity-90"
        style={{ backgroundColor: team.color }}
      >
        {abbr}
      </div>
      <div className={isHome ? 'text-right' : 'text-left'}>
        <p className="text-xs text-txt-muted uppercase">{isHome ? 'Home' : 'Away'}</p>
        <p className="font-bold text-lg text-txt-primary">{team.name}</p>
        {score !== undefined && score !== null && (
          <p className="text-2xl font-bold text-txt-primary font-mono">{score}</p>
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

  // Win probability
  const homeWinPct = game.p_home_win ? Math.round(game.p_home_win * 100) : null;
  const awayWinPct = game.p_away_win ? Math.round(game.p_away_win * 100) : null;

  return (
    <div className="bg-tru-card rounded-xl border border-tru-border overflow-hidden">
      {/* Header with venue/weather */}
      <div className="bg-tru-surface text-txt-secondary px-4 py-2 flex justify-between items-center">
        <div className="flex items-center gap-2">
          {game.context?.venue_name && (
            <span className="text-sm">{game.context.venue_name}</span>
          )}
          {game.context?.is_dome && (
            <span className="text-xs bg-tru-border px-2 py-0.5 rounded text-txt-muted">Dome</span>
          )}
        </div>
        <div className="flex items-center gap-3 text-sm font-mono">
          {game.context?.temperature && !game.context.is_dome && (
            <span>{game.context.temperature}°F</span>
          )}
          {game.context?.park_factor && game.context.park_factor !== 1.0 && (
            <span className={game.context.park_factor > 1 ? 'text-loss' : 'text-accent-blue'}>
              PF: {game.context.park_factor.toFixed(2)}
            </span>
          )}
          {isLive && (
            <span className="bg-loss px-2 py-0.5 rounded text-xs font-bold text-white animate-pulse">
              LIVE
            </span>
          )}
          {isCompleted && (
            <span className="bg-tru-border px-2 py-0.5 rounded text-xs text-txt-muted">FINAL</span>
          )}
          {!isLive && !isCompleted && (
            <span className="text-txt-primary">{timeDisplay}</span>
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
            {/* Win probability bar */}
            {homeWinPct !== null && awayWinPct !== null && !isCompleted && (
              <div className="mb-2">
                <div className="flex justify-between text-xs text-txt-muted mb-1 font-mono">
                  <span>{awayWinPct}%</span>
                  <span>{homeWinPct}%</span>
                </div>
                <div className="w-28 h-2 bg-tru-border rounded-full overflow-hidden flex">
                  <div
                    className="bg-loss/70 h-full"
                    style={{ width: `${awayWinPct}%` }}
                  />
                  <div
                    className="bg-accent-blue/70 h-full"
                    style={{ width: `${homeWinPct}%` }}
                  />
                </div>
              </div>
            )}

            {game.predicted_run_diff !== null && !isCompleted && (
              <div className="mb-1">
                <span className="text-xs text-txt-muted block">Run Diff</span>
                <span className={`text-lg font-bold font-mono ${
                  game.predicted_run_diff > 0 ? 'text-accent-blue' : 'text-loss'
                }`}>
                  {game.predicted_run_diff > 0 ? '+' : ''}{game.predicted_run_diff.toFixed(1)}
                </span>
              </div>
            )}
            {game.predicted_total !== null && !isCompleted && (
              <div>
                <span className="text-xs text-txt-muted block">Total</span>
                <span className="text-sm font-bold text-txt-secondary font-mono">
                  {game.predicted_total.toFixed(1)}
                </span>
              </div>
            )}

            {/* Completed game summary */}
            {isCompleted && game.home_score !== null && game.away_score !== null && (
              <div>
                <div className="text-sm text-txt-muted font-mono">
                  Total: {game.home_score + game.away_score}
                </div>
                {/* First inning result */}
                {game.home_first_inning_runs !== null && game.away_first_inning_runs !== null && (
                  <div className="mt-1">
                    <span className="text-xs text-txt-muted">1st Inn</span>
                    <p className={`text-xs font-semibold font-mono ${
                      game.home_first_inning_runs + game.away_first_inning_runs === 0
                        ? 'text-accent-cyan' : 'text-push'
                    }`}>
                      {game.away_first_inning_runs}-{game.home_first_inning_runs}
                      {game.home_first_inning_runs + game.away_first_inning_runs === 0
                        ? ' NRFI' : ''}
                    </p>
                  </div>
                )}
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
          <div className="mt-4 pt-3 border-t border-tru-border">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-txt-muted uppercase tracking-wide">
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
          <div className="mt-3 pt-3 border-t border-tru-border">
            <div className="grid grid-cols-3 gap-2 text-center text-sm">
              {game.markets.find(m => m.market_type === 'moneyline') && (
                <div className="bg-tru-surface rounded-lg py-2">
                  <p className="text-xs text-txt-muted">Moneyline</p>
                  <p className="font-semibold text-txt-primary font-mono">
                    {game.markets.find(m => m.market_type === 'moneyline')?.home_odds
                      ? formatOdds(game.markets.find(m => m.market_type === 'moneyline')!.home_odds!)
                      : '-'}
                  </p>
                </div>
              )}
              {game.markets.find(m => m.market_type === 'runline') && (
                <div className="bg-tru-surface rounded-lg py-2">
                  <p className="text-xs text-txt-muted">Run Line</p>
                  <p className="font-semibold text-txt-primary font-mono">
                    {game.markets.find(m => m.market_type === 'runline')?.line || '-1.5'}
                  </p>
                </div>
              )}
              {game.markets.find(m => m.market_type === 'total') && (
                <div className="bg-tru-surface rounded-lg py-2">
                  <p className="text-xs text-txt-muted">Total</p>
                  <p className="font-semibold text-txt-primary font-mono">
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
