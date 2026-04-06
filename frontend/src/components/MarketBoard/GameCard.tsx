import { Link } from 'react-router-dom';
import { getTeamColor } from '@/lib/teamLogos';
import type { Market } from '@/types/market';
import type { TeamTrends, GamePrediction, TornadoFactor, TeamInjuries, HeadToHead, SharpMoney } from '@/lib/api';
import { ValueBadge } from '@/components/ui/ValueBadge';
import { SharpMoneyBadge } from '@/components/ui/SharpMoneyBadge';

interface GameCardProps {
  gameId: string;
  homeTeam: string;
  awayTeam: string;
  tipTime: string;
  markets: Market[];
  homeTrends?: TeamTrends;
  awayTrends?: TeamTrends;
  homeInjuries?: TeamInjuries;
  awayInjuries?: TeamInjuries;
  prediction?: GamePrediction | null;
  tornadoChart?: TornadoFactor[];
  headToHead?: HeadToHead | null;
  sharpMoney?: SharpMoney | null;
}

function formatGameTime(tipTime: string): { date: string; time: string } {
  const d = new Date(tipTime);
  const date = d.toLocaleDateString('en-US', { weekday: 'long', month: 'short', day: 'numeric' });
  const time = d.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', hour12: true }) + ' ET';
  return { date, time };
}

function formatLine(line: number | null, showPlus = true): string {
  if (line === null) return '';
  if (showPlus && line > 0) return `+${line}`;
  return `${line}`;
}

// Get best value score for a team's spread market
function getTeamSpreadValue(markets: Market[], _team: string, isHome: boolean): { score: number; line: number | null; marketId: string; winProb: number } | null {
  const spreadMarkets = markets.filter(m =>
    m.market_type === 'spread' &&
    ((isHome && m.outcome_label.includes('home')) || (!isHome && m.outcome_label.includes('away')))
  );

  if (spreadMarkets.length === 0) return null;

  const best = spreadMarkets.reduce((a, b) => {
    return b.algo_a_value_score > a.algo_a_value_score ? b : a;
  });

  // p_true is the model's probability this bet wins (covers the spread)
  const winProb = Math.round(best.p_true * 100);
  return { score: best.algo_a_value_score, line: best.line, marketId: best.market_id, winProb };
}

// Get consensus spread line (returns home team's perspective)
function getConsensusLine(markets: Market[], type: string): number | null {
  // For spreads, only look at home_spread to get consistent perspective
  const lines = markets
    .filter(m => m.market_type === type && m.line !== null && (type !== 'spread' || m.outcome_label.includes('home')))
    .map(m => m.line as number);

  if (lines.length === 0) return null;

  const counts = lines.reduce((acc, line) => {
    acc[line] = (acc[line] || 0) + 1;
    return acc;
  }, {} as Record<number, number>);

  const [mostCommon] = Object.entries(counts).sort(([, a], [, b]) => b - a);
  return parseFloat(mostCommon[0]);
}

// Get total line
function getTotalLine(markets: Market[]): number | null {
  return getConsensusLine(markets, 'total');
}

// Get moneyline odds for display
function getMoneylineOdds(markets: Market[], isHome: boolean): string | null {
  const mlMarkets = markets.filter(m =>
    m.market_type === 'moneyline' &&
    ((isHome && m.outcome_label.includes('home')) || (!isHome && m.outcome_label.includes('away')))
  );
  if (mlMarkets.length === 0) return null;
  const best = mlMarkets[0];
  if (best.odds_american !== undefined) {
    return best.odds_american > 0 ? `+${best.odds_american}` : `${best.odds_american}`;
  }
  // Convert decimal to american
  if (best.odds_decimal >= 2.0) {
    return `+${Math.round((best.odds_decimal - 1) * 100)}`;
  }
  return `${Math.round(-100 / (best.odds_decimal - 1))}`;
}

// Injury severity badge
function InjuryBadge({ severity, impact }: { severity: string; impact: number }) {
  const getColor = () => {
    if (severity === 'severe') return 'bg-red-900/30 text-red-400 border-red-800';
    if (severity === 'moderate') return 'bg-amber-900/30 text-amber-400 border-amber-800';
    if (severity === 'minor') return 'bg-yellow-900/30 text-yellow-400 border-yellow-800';
    return 'bg-[#0b0e14] text-[#64748b] border-[#1e293b]';
  };

  if (severity === 'none' || impact === 0) return null;

  return (
    <span className={`${getColor()} text-xs px-1.5 py-0.5 rounded border font-medium`}>
      {impact}% hurt
    </span>
  );
}


// Edge score badge styling (matching MLB)
function getScoreBadge(score: number) {
  if (score >= 70) {
    return {
      classes: 'bg-[#66f796]/10 border-[#66f796]/30 text-[#66f796]',
      label: 'STRONG',
    };
  }
  if (score >= 60) {
    return {
      classes: 'bg-[#a4e6ff]/10 border-[#a4e6ff]/30 text-[#a4e6ff]',
      label: 'MODERATE',
    };
  }
  return {
    classes: 'bg-[#32353c]/50 border-[#32353c] text-slate-400',
    label: 'LOW',
  };
}

export function GameCard({ gameId: _gameId, homeTeam, awayTeam, tipTime, markets, homeTrends, awayTrends, homeInjuries, awayInjuries, prediction, tornadoChart: _tornadoChart, headToHead, sharpMoney }: GameCardProps) {
  const { date, time } = formatGameTime(tipTime);

  // Get spread values for each team
  const awaySpread = getTeamSpreadValue(markets, awayTeam, false);
  const homeSpread = getTeamSpreadValue(markets, homeTeam, true);

  // Get total line and consensus spread
  const totalLine = getTotalLine(markets);
  const consensusSpread = getConsensusLine(markets, 'spread');

  // Get moneyline odds
  const awayML = getMoneylineOdds(markets, false);
  const homeML = getMoneylineOdds(markets, true);

  // Best value score across all markets for this game
  const allScores = markets.map(m => m.algo_a_value_score).filter(s => s > 0);
  const bestScore = allScores.length > 0 ? Math.max(...allScores) : 0;
  const hasHighValue = bestScore >= 65;

  // Get best market for linking
  const bestMarketId = awaySpread?.marketId || homeSpread?.marketId || '';

  // Win probability from prediction or spread values
  const awayWinProb = awaySpread?.winProb ?? null;
  const homeWinProb = homeSpread?.winProb ?? null;

  // Edge score badge
  const badge = bestScore > 0 ? getScoreBadge(bestScore) : null;

  // Team colors
  const awayColor = getTeamColor(awayTeam);
  const homeColor = getTeamColor(homeTeam);

  return (
    <Link
      to={bestMarketId ? `/bet/${bestMarketId}` : '#'}
      className="block bg-[#191c22] rounded-xl border border-[#1e293b] overflow-hidden relative hover:border-[#a4e6ff]/30 transition-colors"
    >
      {/* Left edge glow for high-value games */}
      {hasHighValue && (
        <div className="absolute left-0 top-0 bottom-0 w-1 bg-[#a4e6ff] shadow-[0_0_8px_rgba(164,230,255,0.4)]" />
      )}

      <div className="p-5">
        {/* Game time bar */}
        <div className="flex items-center justify-between mb-4">
          <span className="text-[10px] text-slate-500 uppercase font-bold tracking-widest">{date}</span>
          <div className="flex items-center gap-2">
            {/* B2B indicators */}
            {awayTrends?.is_b2b && (
              <span className="text-[10px] bg-amber-900/30 text-amber-400 px-2 py-0.5 rounded font-mono font-bold">
                {awayTeam} B2B
              </span>
            )}
            {homeTrends?.is_b2b && (
              <span className="text-[10px] bg-amber-900/30 text-amber-400 px-2 py-0.5 rounded font-mono font-bold">
                {homeTeam} B2B
              </span>
            )}
            <span className="text-[10px] text-[#a4e6ff] font-bold font-mono uppercase tracking-widest">{time}</span>
          </div>
        </div>

        {/* Top row: Teams + Lines + Edge Score */}
        <div className="flex justify-between items-start mb-5">
          {/* Teams + Context */}
          <div className="flex items-center gap-5">
            {/* Teams stacked */}
            <div className="flex flex-col gap-1.5">
              {/* Away team */}
              <div className="flex items-center gap-2.5">
                <div
                  className="w-8 h-8 rounded-full flex items-center justify-center text-[10px] font-bold text-white flex-shrink-0"
                  style={{ backgroundColor: awayColor }}
                >
                  {awayTeam.slice(0, 3)}
                </div>
                <span className="text-base font-bold text-[#f1f5f9]">{awayTeam}</span>
                {awaySpread && (
                  <span className="text-slate-500 font-mono text-xs">{formatLine(awaySpread.line)}</span>
                )}
              </div>
              {/* Home team */}
              <div className="flex items-center gap-2.5">
                <div
                  className="w-8 h-8 rounded-full flex items-center justify-center text-[10px] font-bold text-white flex-shrink-0"
                  style={{ backgroundColor: homeColor }}
                >
                  {homeTeam.slice(0, 3)}
                </div>
                <span className="text-base font-bold text-[#f1f5f9]">{homeTeam}</span>
                {homeSpread && (
                  <span className="text-slate-500 font-mono text-xs">{formatLine(homeSpread.line)}</span>
                )}
              </div>
            </div>

            {/* Divider */}
            <div className="h-12 w-px bg-slate-700/30 mx-1 hidden sm:block" />

            {/* Lines grid */}
            <div className="hidden sm:grid grid-cols-3 gap-x-5 gap-y-1">
              <div>
                <span className="text-[10px] text-slate-500 uppercase font-bold tracking-widest block">Spread</span>
                <span className="text-sm font-mono text-[#f1f5f9]">
                  {consensusSpread !== null ? formatLine(consensusSpread) : '-'}
                </span>
              </div>
              <div>
                <span className="text-[10px] text-slate-500 uppercase font-bold tracking-widest block">Moneyline</span>
                <span className="text-sm font-mono text-[#f1f5f9]">
                  {awayML || '-'} / {homeML || '-'}
                </span>
              </div>
              <div>
                <span className="text-[10px] text-slate-500 uppercase font-bold tracking-widest block">Total</span>
                <span className="text-sm font-mono text-[#f1f5f9]">
                  {totalLine !== null ? `O/U ${totalLine}` : '-'}
                </span>
              </div>
            </div>
          </div>

          {/* Edge Score badge - RIGHT SIDE */}
          {badge && bestScore > 0 && (
            <div className="flex flex-col items-end flex-shrink-0">
              <span className="text-[10px] text-slate-500 uppercase font-bold tracking-widest mb-1">Edge Probability</span>
              <div className={`${badge.classes} border px-3 py-1 rounded-full flex items-center gap-2`}>
                <span className="font-black font-mono text-lg">
                  {bestScore.toFixed(0)}
                </span>
                <span className="text-[10px] font-bold">
                  {badge.label}
                </span>
              </div>
            </div>
          )}
        </div>

        {/* Win Probability Bar */}
        {awayWinProb !== null && homeWinProb !== null && (
          <div className="mb-5">
            <div className="flex justify-between text-[10px] text-slate-500 font-bold uppercase tracking-widest mb-1.5">
              <span>{awayTeam} Cover: {awayWinProb}%</span>
              <span>{homeTeam} Cover: {homeWinProb}%</span>
            </div>
            <div className="h-1.5 w-full bg-[#32353c] rounded-full overflow-hidden flex">
              <div
                className="h-full bg-gradient-to-r from-[#a4e6ff] to-[#00d1ff] transition-all"
                style={{ width: `${awayWinProb}%` }}
              />
            </div>
          </div>
        )}

        {/* Predicted Score */}
        {prediction?.predicted_score && (
          <div className="mb-4">
            <div className="flex items-center justify-between">
              <span className="text-[10px] text-slate-500 uppercase font-bold tracking-widest">Projected Score</span>
              {prediction.confidence && (
                <span className={`text-[10px] font-bold font-mono px-2 py-0.5 rounded ${
                  prediction.confidence === 'high'
                    ? 'bg-[#66f796]/10 text-[#66f796]'
                    : prediction.confidence === 'medium'
                    ? 'bg-amber-400/10 text-amber-400'
                    : 'bg-[#32353c] text-slate-400'
                }`}>
                  {prediction.confidence.toUpperCase()}
                </span>
              )}
            </div>
            <div className="flex items-center gap-3 mt-1">
              <span className="font-mono font-bold text-lg text-[#f1f5f9]">{awayTeam} {prediction.predicted_score.away}</span>
              <span className="text-slate-500 font-mono">-</span>
              <span className="font-mono font-bold text-lg text-[#f1f5f9]">{homeTeam} {prediction.predicted_score.home}</span>
            </div>
          </div>
        )}

        {/* Sharp Money Indicator */}
        {sharpMoney && sharpMoney.signal !== 'neutral' && (
          <div className="mb-4">
            <SharpMoneyBadge
              signal={sharpMoney.signal}
              homeTeam={homeTeam}
              awayTeam={awayTeam}
              spreadMovement={sharpMoney.spread_movement}
              size="sm"
            />
          </div>
        )}

        {/* Team stats - compact grid (matching MLB pitcher cards) */}
        {(homeTrends || awayTrends) && (
          <div className="grid grid-cols-2 gap-3 mb-4">
            {/* Away stats card */}
            <div className="bg-[#0b0e14] p-3.5 rounded-lg">
              <div className="flex items-center justify-between mb-2.5">
                <span className="text-xs font-bold text-slate-400">{awayTeam}</span>
                <span
                  className="text-[10px] font-mono font-bold px-2 py-0.5 rounded"
                  style={{ backgroundColor: awayColor + '22', color: awayColor }}
                >
                  {awayTrends?.record || '0-0'}
                </span>
              </div>
              <div className="grid grid-cols-3 gap-2 text-center">
                <div>
                  <div className="text-[10px] text-slate-500 font-bold uppercase tracking-widest">L10</div>
                  <div className="text-sm font-mono font-bold text-[#f1f5f9]">{awayTrends?.l10_record || '-'}</div>
                </div>
                <div>
                  <div className="text-[10px] text-slate-500 font-bold uppercase tracking-widest">ATS</div>
                  <div className="text-sm font-mono font-bold text-[#f1f5f9]">{awayTrends?.ats_record || '-'}</div>
                </div>
                <div>
                  <div className="text-[10px] text-slate-500 font-bold uppercase tracking-widest">O/U</div>
                  <div className="text-sm font-mono font-bold text-[#f1f5f9]">{awayTrends?.ou_record || '-'}</div>
                </div>
              </div>
              {awayTrends?.net_rtg_l10 != null && (
                <div className="mt-2 pt-2 border-t border-slate-700/20 flex justify-between items-center">
                  <span className="text-[10px] text-slate-500 font-bold uppercase tracking-widest">Net Rtg</span>
                  <span className={`text-sm font-mono font-bold ${
                    awayTrends.net_rtg_l10 > 0 ? 'text-[#66f796]' : awayTrends.net_rtg_l10 < 0 ? 'text-red-400' : 'text-slate-400'
                  }`}>
                    {awayTrends.net_rtg_l10 > 0 ? '+' : ''}{awayTrends.net_rtg_l10.toFixed(1)}
                  </span>
                </div>
              )}
            </div>

            {/* Home stats card */}
            <div className="bg-[#0b0e14] p-3.5 rounded-lg">
              <div className="flex items-center justify-between mb-2.5">
                <span className="text-xs font-bold text-slate-400">{homeTeam}</span>
                <span
                  className="text-[10px] font-mono font-bold px-2 py-0.5 rounded"
                  style={{ backgroundColor: homeColor + '22', color: homeColor }}
                >
                  {homeTrends?.record || '0-0'}
                </span>
              </div>
              <div className="grid grid-cols-3 gap-2 text-center">
                <div>
                  <div className="text-[10px] text-slate-500 font-bold uppercase tracking-widest">L10</div>
                  <div className="text-sm font-mono font-bold text-[#f1f5f9]">{homeTrends?.l10_record || '-'}</div>
                </div>
                <div>
                  <div className="text-[10px] text-slate-500 font-bold uppercase tracking-widest">ATS</div>
                  <div className="text-sm font-mono font-bold text-[#f1f5f9]">{homeTrends?.ats_record || '-'}</div>
                </div>
                <div>
                  <div className="text-[10px] text-slate-500 font-bold uppercase tracking-widest">O/U</div>
                  <div className="text-sm font-mono font-bold text-[#f1f5f9]">{homeTrends?.ou_record || '-'}</div>
                </div>
              </div>
              {homeTrends?.net_rtg_l10 != null && (
                <div className="mt-2 pt-2 border-t border-slate-700/20 flex justify-between items-center">
                  <span className="text-[10px] text-slate-500 font-bold uppercase tracking-widest">Net Rtg</span>
                  <span className={`text-sm font-mono font-bold ${
                    homeTrends.net_rtg_l10 > 0 ? 'text-[#66f796]' : homeTrends.net_rtg_l10 < 0 ? 'text-red-400' : 'text-slate-400'
                  }`}>
                    {homeTrends.net_rtg_l10 > 0 ? '+' : ''}{homeTrends.net_rtg_l10.toFixed(1)}
                  </span>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Head-to-Head Record */}
        {headToHead && headToHead.total_games > 0 && (
          <div className="mb-4 flex items-center justify-between bg-[#0b0e14] rounded-lg px-3.5 py-2.5">
            <span className="text-[10px] text-slate-500 uppercase font-bold tracking-widest">H2H (Last {headToHead.total_games})</span>
            <span className="font-mono font-bold text-sm text-[#f1f5f9]">
              {awayTeam} {headToHead.away_wins}-{headToHead.home_wins} {homeTeam}
            </span>
          </div>
        )}

        {/* Injury Report */}
        {((awayInjuries?.players_out?.length ?? 0) > 0 || (homeInjuries?.players_out?.length ?? 0) > 0) && (
          <div className="pt-4 border-t border-slate-700/30">
            <div className="text-[10px] text-slate-500 uppercase font-bold tracking-widest mb-2.5 flex items-center gap-1.5">
              <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
              Injury Report
            </div>
            <div className="grid grid-cols-2 gap-3 text-xs">
              {/* Away Team Injuries */}
              <div className="bg-[#0b0e14] p-2.5 rounded-lg">
                <div className="flex items-center gap-1 mb-1">
                  <span className="font-bold text-slate-400">{awayTeam}</span>
                  {awayInjuries && <InjuryBadge severity={awayInjuries.severity} impact={awayInjuries.impact_score} />}
                </div>
                {awayInjuries?.players_out && awayInjuries.players_out.length > 0 ? (
                  <div className="text-[#94a3b8] font-mono text-[11px]">
                    <span className="text-red-400 font-bold">OUT: </span>
                    {awayInjuries.players_out.slice(0, 3).join(', ')}
                    {awayInjuries.players_out.length > 3 && <span className="text-slate-600"> +{awayInjuries.players_out.length - 3}</span>}
                  </div>
                ) : (
                  <div className="text-[#66f796] font-mono text-[11px]">Healthy</div>
                )}
              </div>
              {/* Home Team Injuries */}
              <div className="bg-[#0b0e14] p-2.5 rounded-lg">
                <div className="flex items-center gap-1 mb-1">
                  <span className="font-bold text-slate-400">{homeTeam}</span>
                  {homeInjuries && <InjuryBadge severity={homeInjuries.severity} impact={homeInjuries.impact_score} />}
                </div>
                {homeInjuries?.players_out && homeInjuries.players_out.length > 0 ? (
                  <div className="text-[#94a3b8] font-mono text-[11px]">
                    <span className="text-red-400 font-bold">OUT: </span>
                    {homeInjuries.players_out.slice(0, 3).join(', ')}
                    {homeInjuries.players_out.length > 3 && <span className="text-slate-600"> +{homeInjuries.players_out.length - 3}</span>}
                  </div>
                ) : (
                  <div className="text-[#66f796] font-mono text-[11px]">Healthy</div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Prediction / Best Bets */}
        {prediction && (
          <div className="mt-4 pt-4 border-t border-slate-700/30">
            <div className="flex items-center justify-between mb-3">
              <span className="text-[10px] text-slate-500 uppercase font-bold tracking-widest">Our Picks</span>
              <span className={`text-[10px] font-bold font-mono px-2 py-0.5 rounded ${
                prediction.confidence === 'high'
                  ? 'bg-[#66f796]/10 text-[#66f796]'
                  : prediction.confidence === 'medium'
                  ? 'bg-amber-400/10 text-amber-400'
                  : 'bg-[#32353c] text-slate-400'
              }`}>
                {prediction.confidence?.toUpperCase() || 'LOW'}
              </span>
            </div>

            <div className="grid grid-cols-2 gap-3 mb-3">
              {/* Winner Prediction */}
              <div className="bg-[#0b0e14] rounded-lg p-2.5">
                <div className="text-[10px] text-slate-500 font-bold uppercase tracking-widest mb-1">Moneyline</div>
                <div className="flex items-center gap-2">
                  <div
                    className="w-5 h-5 rounded-full flex items-center justify-center text-[7px] font-bold text-white flex-shrink-0"
                    style={{ backgroundColor: getTeamColor(prediction.winner) }}
                  >
                    {prediction.winner.slice(0, 3)}
                  </div>
                  <span className="font-bold text-[#f1f5f9] text-sm font-mono">{prediction.winner}</span>
                  <span className="text-[10px] text-slate-500 font-mono">({prediction.winner_prob}%)</span>
                </div>
              </div>

              {/* Spread Pick */}
              {prediction.spread_pick && (
                <div className="bg-[#0b0e14] rounded-lg p-2.5">
                  <div className="text-[10px] text-slate-500 font-bold uppercase tracking-widest mb-1">Spread</div>
                  <div className="flex items-center gap-2">
                    <div
                      className="w-5 h-5 rounded-full flex items-center justify-center text-[7px] font-bold text-white flex-shrink-0"
                      style={{ backgroundColor: getTeamColor(prediction.spread_pick.team) }}
                    >
                      {prediction.spread_pick.team.slice(0, 3)}
                    </div>
                    <span className="font-bold text-[#f1f5f9] text-sm font-mono">
                      {prediction.spread_pick.team} {prediction.spread_pick.line != null ? (prediction.spread_pick.line > 0 ? `+${prediction.spread_pick.line}` : prediction.spread_pick.line) : ''}
                    </span>
                    {prediction.spread_pick.value_score >= 50 && (
                      <ValueBadge score={prediction.spread_pick.value_score} size="sm" />
                    )}
                  </div>
                </div>
              )}
            </div>

            {/* Best Value Bet (if different from spread) */}
            {prediction.best_bet && prediction.best_bet.value_score >= 50 && prediction.best_bet.type !== 'spread' && (
              <div className="bg-[#66f796]/5 border border-[#66f796]/20 rounded-lg px-3 py-2 mb-3 flex items-center gap-2">
                <span className="text-[10px] text-[#66f796] font-bold uppercase tracking-widest">Best Value:</span>
                <span className="font-bold font-mono text-sm text-[#66f796]">
                  {prediction.best_bet.type === 'total'
                    ? `${prediction.best_bet.team} ${prediction.best_bet.line}`
                    : `${prediction.best_bet.team} ML`
                  }
                </span>
                <span className="text-[#66f796] font-mono text-xs">({prediction.best_bet.value_score}%)</span>
              </div>
            )}

            {/* Key Factors */}
            {prediction.factors.length > 0 && (
              <ul className="text-xs text-[#94a3b8] space-y-0.5">
                {prediction.factors.map((factor, i) => (
                  <li key={i} className="flex items-start gap-1.5 font-mono text-[11px]">
                    <span className="text-slate-600">-</span>
                    <span>{factor}</span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}
      </div>
    </Link>
  );
}
