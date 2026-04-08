import { useState } from 'react';
import type { MLBGame, ValueBetInfo, PitcherInfo, FirstInningStats } from '@/lib/mlbApi';
import { getTeamInfo, formatOdds } from '@/lib/mlbApi';
import { getMLBLogo } from '@/lib/mlbLogos';

interface MLBGameCardProps {
  game: MLBGame;
  firstInningStats?: Map<string, FirstInningStats>;
}

function MLBLogoCircle({ team, size = 32 }: { team: string; size?: number }) {
  const [imgError, setImgError] = useState(false);
  const teamInfo = getTeamInfo(team);
  const logoUrl = getMLBLogo(team);

  if (imgError || !logoUrl) {
    return (
      <div
        className="rounded-full flex items-center justify-center text-[10px] font-bold text-white flex-shrink-0"
        style={{ width: size, height: size, backgroundColor: teamInfo.color }}
      >
        {team}
      </div>
    );
  }

  return (
    <div
      className="rounded-full bg-white/5 flex items-center justify-center flex-shrink-0 overflow-hidden p-0.5"
      style={{ width: size, height: size }}
    >
      <img
        src={logoUrl}
        alt={team}
        className="w-full h-full object-contain"
        onError={() => setImgError(true)}
      />
    </div>
  );
}

function ValueBadge({ value }: { value: ValueBetInfo }) {
  const scoreColor = value.value_score >= 70
    ? 'bg-[#66f796]/10 text-[#66f796] border-[#66f796]/20'
    : value.value_score >= 60
    ? 'bg-[#f59e0b]/10 text-[#f59e0b] border-[#f59e0b]/20'
    : 'bg-[#191c22] text-txt-secondary border-[#32353c]';

  const label = value.market_type === 'moneyline'
    ? `${value.team} ML`
    : value.market_type === 'runline'
    ? `${value.team} ${value.line && value.line > 0 ? '+' : ''}${value.line}`
    : value.bet_type === 'over'
    ? `O ${value.line}`
    : `U ${value.line}`;

  return (
    <div className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-lg border ${scoreColor}`}>
      <span className="font-semibold text-sm">{label}</span>
      <span className="text-xs font-mono">({formatOdds(value.odds_decimal)})</span>
      <span className="font-bold font-mono text-sm">{value.value_score.toFixed(0)}</span>
    </div>
  );
}

function PitcherCard({ pitcher, teamAbbr }: { pitcher: PitcherInfo | null; teamAbbr: string }) {
  const team = getTeamInfo(teamAbbr);

  if (!pitcher) {
    return (
      <div className="bg-[#0b0e14] p-3.5 rounded-lg">
        <div className="flex items-center justify-between mb-2.5">
          <span className="text-xs font-bold text-slate-400">TBD</span>
          <span className="text-[10px] font-mono text-[#a4e6ff] px-2 py-0.5 bg-[#a4e6ff]/10 rounded">{teamAbbr}</span>
        </div>
        <div className="grid grid-cols-3 gap-2">
          {['ERA', 'WHIP', 'K/9'].map(stat => (
            <div key={stat} className="text-center">
              <div className="text-[10px] text-slate-500 font-bold uppercase tracking-widest">{stat}</div>
              <div className="text-sm font-mono font-bold text-slate-600">--</div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  const hand = pitcher.throws === 'L' ? 'LHP' : 'RHP';

  return (
    <div className="bg-[#0b0e14] p-3.5 rounded-lg">
      <div className="flex items-center justify-between mb-2.5">
        <span className="text-xs font-bold text-slate-400">
          <span className="text-[10px] font-mono text-slate-500 mr-1.5">{hand}</span>
          {pitcher.name}
        </span>
        <span
          className="text-[10px] font-mono font-bold px-2 py-0.5 rounded"
          style={{ backgroundColor: team.color + '22', color: team.color }}
        >
          {teamAbbr}
        </span>
      </div>
      <div className="grid grid-cols-3 gap-2">
        <div className="text-center">
          <div className="text-[10px] text-slate-500 font-bold uppercase tracking-widest">ERA</div>
          <div className="text-sm font-mono font-bold text-txt-primary">
            {pitcher.era !== null ? pitcher.era.toFixed(2) : '--'}
          </div>
        </div>
        <div className="text-center">
          <div className="text-[10px] text-slate-500 font-bold uppercase tracking-widest">WHIP</div>
          <div className="text-sm font-mono font-bold text-txt-primary">
            {pitcher.whip !== null ? pitcher.whip.toFixed(2) : '--'}
          </div>
        </div>
        <div className="text-center">
          <div className="text-[10px] text-slate-500 font-bold uppercase tracking-widest">K/9</div>
          <div className="text-sm font-mono font-bold text-txt-primary">
            {pitcher.k_per_9 !== null ? pitcher.k_per_9.toFixed(1) : '--'}
          </div>
        </div>
      </div>
    </div>
  );
}

export function MLBGameCard({ game, firstInningStats }: MLBGameCardProps) {
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

  // Team info
  const awayTeam = getTeamInfo(game.away_team);
  const homeTeam = getTeamInfo(game.home_team);

  // Moneyline odds
  const mlMarket = game.markets.find(m => m.market_type === 'moneyline');
  const awayOdds = mlMarket?.away_odds ? formatOdds(mlMarket.away_odds) : null;
  const homeOdds = mlMarket?.home_odds ? formatOdds(mlMarket.home_odds) : null;

  // NRFI probability: combine both teams' first inning scoreless rates
  // (probability neither team scores in the 1st)
  const homeFI = firstInningStats?.get(game.home_team);
  const awayFI = firstInningStats?.get(game.away_team);
  let nrfiScore: number | null = null;
  let nrfiMinGames = 0;
  if (homeFI && awayFI && homeFI.games > 0 && awayFI.games > 0) {
    const homeScoreless = 1 - homeFI.score_pct;
    const awayScoreless = 1 - awayFI.score_pct;
    nrfiScore = homeScoreless * awayScoreless * 100;
    nrfiMinGames = Math.min(homeFI.games, awayFI.games);
  }
  // Fall back to best_bet value_score if no first inning data
  if (nrfiScore === null && game.best_bet?.value_score) {
    nrfiScore = game.best_bet.value_score;
  }
  const hasHighValue = nrfiScore !== null && nrfiScore >= 65;

  // Score badge styling (for NRFI probability)
  const getScoreBadge = (score: number) => {
    if (score >= 70) {
      return {
        bg: 'bg-[#66f796]/10 border-[#66f796]/30',
        text: 'text-[#66f796]',
        label: 'STRONG NRFI',
      };
    }
    if (score >= 55) {
      return {
        bg: 'bg-[#a4e6ff]/10 border-[#a4e6ff]/30',
        text: 'text-[#a4e6ff]',
        label: 'LEAN NRFI',
      };
    }
    if (score >= 40) {
      return {
        bg: 'bg-[#32353c]/50 border-[#32353c]',
        text: 'text-slate-400',
        label: 'TOSSUP',
      };
    }
    return {
      bg: 'bg-[#f59e0b]/10 border-[#f59e0b]/30',
      text: 'text-[#f59e0b]',
      label: 'LEAN YRFI',
    };
  };

  // First inning result
  const hasFirstInning = game.home_first_inning_runs !== null && game.away_first_inning_runs !== null;
  const isNRFI = hasFirstInning &&
    (game.home_first_inning_runs! + game.away_first_inning_runs!) === 0;

  return (
    <div className="bg-[#191c22] rounded-xl relative overflow-hidden">
      {/* Left edge glow for high-value games */}
      {hasHighValue && (
        <div className="absolute left-0 top-0 bottom-0 w-1 bg-[#a4e6ff] shadow-[0_0_8px_rgba(164,230,255,0.4)]" />
      )}

      <div className="p-5">
        {/* Status bar */}
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            {game.context?.venue_name && (
              <span className="text-[10px] text-slate-500 uppercase font-bold tracking-widest">
                {game.context.venue_name}
              </span>
            )}
            {game.context?.is_dome && (
              <span className="text-[10px] bg-[#32353c] px-2 py-0.5 rounded text-slate-400 font-mono">DOME</span>
            )}
          </div>
          <div className="flex items-center gap-2">
            {isLive && (
              <span className="bg-red-500/20 text-red-400 px-2 py-0.5 rounded text-[10px] font-bold font-mono uppercase tracking-widest animate-pulse">
                LIVE
              </span>
            )}
            {isCompleted && (
              <span className="bg-[#32353c] px-2 py-0.5 rounded text-[10px] text-slate-400 font-bold font-mono uppercase tracking-widest">
                FINAL
              </span>
            )}
            {!isLive && !isCompleted && (
              <span className="text-[10px] text-[#a4e6ff] font-bold font-mono uppercase tracking-widest">
                {timeDisplay}
              </span>
            )}
          </div>
        </div>

        {/* Top row: Teams + Context + Edge Score */}
        <div className="flex justify-between items-start mb-5">
          {/* Teams + Context */}
          <div className="flex items-center gap-5">
            {/* Teams stacked */}
            <div className="flex flex-col gap-1.5">
              {/* Away team */}
              <div className="flex items-center gap-2.5">
                <MLBLogoCircle team={game.away_team} />
                <span className="text-base font-bold text-txt-primary">{awayTeam.name}</span>
                {awayOdds && (
                  <span className="text-slate-500 font-mono text-xs">{awayOdds}</span>
                )}
                {(isCompleted || isLive) && game.away_score !== null && (
                  <span className="text-lg font-bold text-txt-primary font-mono ml-1">{game.away_score}</span>
                )}
              </div>
              {/* Home team */}
              <div className="flex items-center gap-2.5">
                <MLBLogoCircle team={game.home_team} />
                <span className="text-base font-bold text-txt-primary">{homeTeam.name}</span>
                {homeOdds && (
                  <span className="text-slate-500 font-mono text-xs">{homeOdds}</span>
                )}
                {(isCompleted || isLive) && game.home_score !== null && (
                  <span className="text-lg font-bold text-txt-primary font-mono ml-1">{game.home_score}</span>
                )}
              </div>
            </div>

            {/* Divider */}
            <div className="h-12 w-px bg-slate-700/30 mx-1 hidden sm:block" />

            {/* Context grid */}
            <div className="hidden sm:grid grid-cols-2 gap-x-6 gap-y-1.5">
              {game.context?.park_factor != null && (
                <div className="flex flex-col">
                  <span className="text-[10px] text-slate-500 uppercase font-bold tracking-widest">Park Factor</span>
                  <span className={`text-sm font-mono font-semibold ${
                    game.context.park_factor > 1 ? 'text-red-400' : 'text-[#a4e6ff]'
                  }`}>
                    {game.context.park_factor.toFixed(2)}
                  </span>
                </div>
              )}
              {game.context?.temperature != null && !game.context?.is_dome && (
                <div className="flex flex-col">
                  <span className="text-[10px] text-slate-500 uppercase font-bold tracking-widest">Weather</span>
                  <span className="text-sm font-mono font-semibold text-txt-primary">
                    {game.context.temperature}°F
                  </span>
                </div>
              )}
              {game.context?.wind_speed != null && !game.context?.is_dome && (
                <div className="flex flex-col">
                  <span className="text-[10px] text-slate-500 uppercase font-bold tracking-widest">Wind</span>
                  <span className="text-sm font-mono font-semibold text-txt-primary">
                    {game.context.wind_speed} mph
                  </span>
                </div>
              )}
              {game.predicted_total !== null && !isCompleted && (
                <div className="flex flex-col">
                  <span className="text-[10px] text-slate-500 uppercase font-bold tracking-widest">Proj Total</span>
                  <span className="text-sm font-mono font-semibold text-txt-primary">
                    {game.predicted_total.toFixed(1)}
                  </span>
                </div>
              )}
            </div>
          </div>

          {/* NRFI Chance badge - RIGHT SIDE */}
          {nrfiScore !== null && (() => {
            const badge = getScoreBadge(nrfiScore);
            const isNrfiFromStats = homeFI && awayFI;
            return (
              <div className="flex flex-col items-end flex-shrink-0">
                <span className="text-[10px] text-slate-500 uppercase font-bold tracking-widest mb-1">
                  {isNrfiFromStats ? 'NRFI Chance' : 'Edge Score'}
                </span>
                <div className={`${badge.bg} border px-3 py-1 rounded-full flex items-center gap-2`}>
                  <span className={`${badge.text} font-black font-mono text-lg`}>
                    {nrfiScore.toFixed(0)}{isNrfiFromStats ? '%' : ''}
                  </span>
                  <span className={`text-[10px] ${badge.text} font-bold tracking-widest`}>
                    {badge.label}
                  </span>
                </div>
                {isNrfiFromStats && nrfiMinGames > 0 && (
                  <span className="text-[9px] text-slate-600 font-mono mt-1">
                    n={nrfiMinGames}+
                  </span>
                )}
              </div>
            );
          })()}
        </div>

        {/* Win Probability Bar */}
        {homeWinPct !== null && awayWinPct !== null && !isCompleted && (
          <div className="mb-5">
            <div className="flex justify-between text-[10px] text-slate-500 font-bold uppercase tracking-widest mb-1.5">
              <span>{game.away_team} Win: {awayWinPct}%</span>
              <span>{game.home_team} Win: {homeWinPct}%</span>
            </div>
            <div className="h-1.5 w-full bg-[#32353c] rounded-full overflow-hidden flex">
              <div
                className="h-full bg-gradient-to-r from-[#a4e6ff] to-[#00d1ff] transition-all"
                style={{ width: `${awayWinPct}%` }}
              />
            </div>
          </div>
        )}

        {/* Pitcher comparison */}
        {!isCompleted && (game.away_starter || game.home_starter) && (
          <div className="mb-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-[10px] text-slate-500 uppercase font-bold tracking-widest">
                Pitcher Matchup
              </span>
              {game.predicted_run_diff !== null && (
                <span className={`text-[10px] font-bold font-mono px-2 py-0.5 rounded ${
                  game.predicted_run_diff > 0
                    ? 'bg-[#a4e6ff]/10 text-[#a4e6ff]'
                    : 'bg-red-400/10 text-red-400'
                }`}>
                  {game.predicted_run_diff > 0 ? 'HOME' : 'AWAY'} +{Math.abs(game.predicted_run_diff).toFixed(1)}
                </span>
              )}
            </div>
            <div className="grid grid-cols-2 gap-3">
              <PitcherCard pitcher={game.away_starter} teamAbbr={game.away_team} />
              <PitcherCard pitcher={game.home_starter} teamAbbr={game.home_team} />
            </div>
          </div>
        )}

        {/* Completed game: score + NRFI result */}
        {isCompleted && game.home_score !== null && game.away_score !== null && (
          <div className="mt-4 pt-4 border-t border-slate-700/30">
            <div className="flex justify-between items-center">
              <div className="flex items-center gap-4">
                <span className="font-mono text-lg font-bold text-txt-primary">
                  {game.away_score} - {game.home_score}
                </span>
                <span className="text-slate-500 font-mono text-sm">
                  Total: {game.away_score + game.home_score}
                </span>
              </div>
              {hasFirstInning && (
                <div className={`px-3 py-1 rounded-full text-xs font-bold font-mono flex items-center gap-1.5 ${
                  isNRFI
                    ? 'bg-[#66f796]/10 text-[#66f796] border border-[#66f796]/30'
                    : 'bg-[#f59e0b]/10 text-[#f59e0b] border border-[#f59e0b]/30'
                }`}>
                  <span>1st: {game.away_first_inning_runs}-{game.home_first_inning_runs}</span>
                  <span>{isNRFI ? 'NRFI \u2713' : 'YRFI'}</span>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Value bets */}
        {!isCompleted && game.best_bet && (
          <div className="mt-4 pt-4 border-t border-slate-700/30">
            <div className="flex items-center justify-between mb-2">
              <span className="text-[10px] text-slate-500 uppercase font-bold tracking-widest">
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
          <div className="mt-3 pt-3 border-t border-slate-700/30">
            <div className="grid grid-cols-3 gap-2 text-center text-sm">
              {game.markets.find(m => m.market_type === 'moneyline') && (
                <div className="bg-[#0b0e14] rounded-lg py-2">
                  <p className="text-[10px] text-slate-500 font-bold uppercase tracking-widest">Moneyline</p>
                  <p className="font-semibold text-txt-primary font-mono text-sm">
                    {game.markets.find(m => m.market_type === 'moneyline')?.home_odds
                      ? formatOdds(game.markets.find(m => m.market_type === 'moneyline')!.home_odds!)
                      : '-'}
                  </p>
                </div>
              )}
              {game.markets.find(m => m.market_type === 'runline') && (
                <div className="bg-[#0b0e14] rounded-lg py-2">
                  <p className="text-[10px] text-slate-500 font-bold uppercase tracking-widest">Run Line</p>
                  <p className="font-semibold text-txt-primary font-mono text-sm">
                    {game.markets.find(m => m.market_type === 'runline')?.line || '-1.5'}
                  </p>
                </div>
              )}
              {game.markets.find(m => m.market_type === 'total') && (
                <div className="bg-[#0b0e14] rounded-lg py-2">
                  <p className="text-[10px] text-slate-500 font-bold uppercase tracking-widest">Total</p>
                  <p className="font-semibold text-txt-primary font-mono text-sm">
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
