import { useState } from 'react';
import { Link } from 'react-router-dom';
import { getTeamLogo, getTeamColor } from '@/lib/teamLogos';
import type { Market } from '@/types/market';
import type { TeamTrends, GamePrediction, TornadoFactor, TeamInjuries, HeadToHead, SharpMoney } from '@/lib/api';
import { TornadoChart } from './TornadoChart';
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

export function GameCard({ gameId: _gameId, homeTeam, awayTeam, tipTime, markets, homeTrends, awayTrends, homeInjuries, awayInjuries, prediction, tornadoChart, headToHead, sharpMoney }: GameCardProps) {
  const { date, time } = formatGameTime(tipTime);

  // Get spread values for each team
  const awaySpread = getTeamSpreadValue(markets, awayTeam, false);
  const homeSpread = getTeamSpreadValue(markets, homeTeam, true);

  // Get total line
  const totalLine = getTotalLine(markets);

  // Get best market for linking
  const bestMarketId = awaySpread?.marketId || homeSpread?.marketId || '';

  return (
    <Link
      to={bestMarketId ? `/bet/${bestMarketId}` : '#'}
      className="block bg-[#191c22] rounded-xl border border-[#1e293b] overflow-hidden hover:border-[#a4e6ff]/30 transition-colors"
    >
      {/* Dark Header */}
      <div className="bg-[#0b0e14] text-[#f1f5f9] px-4 py-2 text-sm font-medium tracking-wide">
        {awayTeam} @ {homeTeam}
      </div>

      {/* Main Content */}
      <div className="p-4">
        {/* Teams Row */}
        <div className="flex items-center justify-between">
          {/* Away Team */}
          <div className="flex items-center gap-3">
            <TeamLogo team={awayTeam} size={56} />
            <div>
              <div className="text-xl font-bold text-[#f1f5f9]">{awayTeam}</div>
              <div className="flex items-center gap-2 mt-1">
                {awaySpread && (
                  <>
                    {awaySpread.score > 0 && <ValueBadge score={awaySpread.score} size="lg" />}
                    <span className="text-lg font-semibold text-[#e1e2eb]">
                      {formatLine(awaySpread.line)}
                    </span>
                  </>
                )}
              </div>
              {awaySpread && awaySpread.score > 0 ? (
                <div className="text-xs text-[#64748b] mt-0.5">
                  {awaySpread.winProb}% to cover
                </div>
              ) : awaySpread && (
                <div className="text-xs text-[#475569] mt-0.5">
                  No value
                </div>
              )}
            </div>
          </div>

          {/* Center - Date/Time/Predicted Score */}
          <div className="text-center px-4">
            <div className="text-sm font-medium text-[#f1f5f9]">{date}</div>
            <div className="text-sm text-[#64748b]">{time}</div>
            {/* Predicted Score - Covers style */}
            {prediction?.predicted_score && (
              <div className="mt-2 bg-[#0b0e14] rounded-lg px-3 py-1.5">
                <div className="text-xs text-[#64748b] mb-0.5">Projected</div>
                <div className="flex items-center justify-center gap-2">
                  <span className="text-lg font-bold text-[#f1f5f9]">{prediction.predicted_score.away}</span>
                  <span className="text-[#475569]">-</span>
                  <span className="text-lg font-bold text-[#f1f5f9]">{prediction.predicted_score.home}</span>
                </div>
              </div>
            )}
            {totalLine && !prediction?.predicted_score && (
              <div className="mt-2 text-sm">
                <span className="text-[#64748b]">o/u</span>{' '}
                <span className="font-semibold text-[#f1f5f9]">{totalLine}</span>
              </div>
            )}
            {/* Sharp Money Indicator */}
            {sharpMoney && sharpMoney.signal !== 'neutral' && (
              <div className="mt-2">
                <SharpMoneyBadge
                  signal={sharpMoney.signal}
                  homeTeam={homeTeam}
                  awayTeam={awayTeam}
                  spreadMovement={sharpMoney.spread_movement}
                  size="sm"
                />
              </div>
            )}
          </div>

          {/* Home Team */}
          <div className="flex items-center gap-3">
            <div className="text-right">
              <div className="text-xl font-bold text-[#f1f5f9]">{homeTeam}</div>
              <div className="flex items-center justify-end gap-2 mt-1">
                {homeSpread && (
                  <>
                    <span className="text-lg font-semibold text-[#e1e2eb]">
                      {formatLine(homeSpread.line)}
                    </span>
                    {homeSpread.score > 0 && <ValueBadge score={homeSpread.score} size="lg" />}
                  </>
                )}
              </div>
              {homeSpread && homeSpread.score > 0 ? (
                <div className="text-xs text-[#64748b] mt-0.5">
                  {homeSpread.winProb}% to cover
                </div>
              ) : homeSpread && (
                <div className="text-xs text-[#475569] mt-0.5">
                  No value
                </div>
              )}
            </div>
            <TeamLogo team={homeTeam} size={56} />
          </div>
        </div>

        {/* Team Stats */}
        <div className="mt-4 pt-4 border-t border-[#1e293b]">
          {/* Stats Table */}
          <div className="grid grid-cols-3 gap-2 text-xs">
            {/* Header Row */}
            <div className="font-medium text-[#64748b]">{awayTeam}</div>
            <div className="text-center font-medium text-[#64748b]"></div>
            <div className="text-right font-medium text-[#64748b]">{homeTeam}</div>

            {/* Overall Record */}
            <div className="font-semibold text-[#f1f5f9]">{awayTrends?.record || '0-0'}</div>
            <div className="text-center text-[#475569]">Record</div>
            <div className="text-right font-semibold text-[#f1f5f9]">{homeTrends?.record || '0-0'}</div>

            {/* Home/Road Record */}
            <div className="text-[#94a3b8]">{awayTrends?.away_record || '0-0'} <span className="text-[#475569]">Road</span></div>
            <div className="text-center text-[#475569]">Split</div>
            <div className="text-right text-[#94a3b8]"><span className="text-[#475569]">Home</span> {homeTrends?.home_record || '0-0'}</div>

            {/* L10 Record */}
            <div className="text-[#94a3b8]">{awayTrends?.l10_record || '0-0'}</div>
            <div className="text-center text-[#475569]">L10</div>
            <div className="text-right text-[#94a3b8]">{homeTrends?.l10_record || '0-0'}</div>

            {/* Net Rating L10 */}
            <div className={`font-medium ${(awayTrends?.net_rtg_l10 ?? 0) > 0 ? 'text-[#10b981]' : (awayTrends?.net_rtg_l10 ?? 0) < 0 ? 'text-[#ef4444]' : 'text-[#64748b]'}`}>
              {awayTrends?.net_rtg_l10 != null ? `${(awayTrends?.net_rtg_l10 ?? 0) > 0 ? '+' : ''}${(awayTrends?.net_rtg_l10 ?? 0).toFixed(1)}` : '-'}
            </div>
            <div className="text-center text-[#475569]">Net Rtg</div>
            <div className={`text-right font-medium ${(homeTrends?.net_rtg_l10 ?? 0) > 0 ? 'text-[#10b981]' : (homeTrends?.net_rtg_l10 ?? 0) < 0 ? 'text-[#ef4444]' : 'text-[#64748b]'}`}>
              {homeTrends?.net_rtg_l10 != null ? `${(homeTrends?.net_rtg_l10 ?? 0) > 0 ? '+' : ''}${(homeTrends?.net_rtg_l10 ?? 0).toFixed(1)}` : '-'}
            </div>

            {/* ATS Record L10 */}
            <div className="text-[#94a3b8]">{awayTrends?.ats_record || '0-0'}</div>
            <div className="text-center text-[#475569]">ATS</div>
            <div className="text-right text-[#94a3b8]">{homeTrends?.ats_record || '0-0'}</div>

            {/* O/U Record L10 */}
            <div className="text-[#94a3b8]">{awayTrends?.ou_record || '0o-0u'}</div>
            <div className="text-center text-[#475569]">O/U</div>
            <div className="text-right text-[#94a3b8]">{homeTrends?.ou_record || '0o-0u'}</div>
          </div>

          {/* B2B Indicators */}
          {(awayTrends?.is_b2b || homeTrends?.is_b2b) && (
            <div className="mt-2 flex justify-between text-xs">
              <div>{awayTrends?.is_b2b && <span className="text-amber-600 font-medium">B2B</span>}</div>
              <div>{homeTrends?.is_b2b && <span className="text-amber-600 font-medium">B2B</span>}</div>
            </div>
          )}

          {/* Head-to-Head Record */}
          {headToHead && headToHead.total_games > 0 && (
            <div className="mt-3 pt-3 border-t border-[#1e293b]">
              <div className="flex items-center justify-between text-xs">
                <span className="text-[#64748b] font-medium">H2H (Last {headToHead.total_games})</span>
                <span className="font-semibold text-[#e1e2eb]">
                  {awayTeam} {headToHead.away_wins}-{headToHead.home_wins} {homeTeam}
                </span>
              </div>
            </div>
          )}
        </div>

        {/* Injury Report Section */}
        {((awayInjuries?.players_out?.length ?? 0) > 0 || (homeInjuries?.players_out?.length ?? 0) > 0) && (
          <div className="mt-3 pt-3 border-t border-[#1e293b]">
            <div className="text-xs font-semibold text-[#64748b] mb-2 flex items-center gap-1">
              <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
              INJURY REPORT
            </div>
            <div className="grid grid-cols-2 gap-3 text-xs">
              {/* Away Team Injuries */}
              <div>
                <div className="flex items-center gap-1 mb-1">
                  <span className="font-medium text-[#e1e2eb]">{awayTeam}</span>
                  {awayInjuries && <InjuryBadge severity={awayInjuries.severity} impact={awayInjuries.impact_score} />}
                </div>
                {awayInjuries?.players_out && awayInjuries.players_out.length > 0 ? (
                  <div className="text-[#94a3b8]">
                    <span className="text-[#ef4444] font-medium">OUT: </span>
                    {awayInjuries.players_out.slice(0, 3).join(', ')}
                    {awayInjuries.players_out.length > 3 && <span className="text-[#475569]"> +{awayInjuries.players_out.length - 3}</span>}
                  </div>
                ) : (
                  <div className="text-[#10b981]">Healthy</div>
                )}
              </div>
              {/* Home Team Injuries */}
              <div className="text-right">
                <div className="flex items-center justify-end gap-1 mb-1">
                  {homeInjuries && <InjuryBadge severity={homeInjuries.severity} impact={homeInjuries.impact_score} />}
                  <span className="font-medium text-[#e1e2eb]">{homeTeam}</span>
                </div>
                {homeInjuries?.players_out && homeInjuries.players_out.length > 0 ? (
                  <div className="text-[#94a3b8]">
                    <span className="text-[#ef4444] font-medium">OUT: </span>
                    {homeInjuries.players_out.slice(0, 3).join(', ')}
                    {homeInjuries.players_out.length > 3 && <span className="text-[#475569]"> +{homeInjuries.players_out.length - 3}</span>}
                  </div>
                ) : (
                  <div className="text-[#10b981]">Healthy</div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Tornado Chart - Matchup Comparison */}
        {tornadoChart && tornadoChart.length > 0 && (
          <div className="mt-4 pt-4 border-t border-[#1e293b]">
            <div className="text-xs font-semibold text-[#94a3b8] mb-2 text-center">MATCHUP BREAKDOWN</div>
            <TornadoChart
              factors={tornadoChart}
              homeTeam={homeTeam}
              awayTeam={awayTeam}
            />
          </div>
        )}

        {/* Prediction Section */}
        {prediction && (
          <div className="mt-4 pt-4 border-t border-[#1e293b] bg-[#0b0e14] -mx-4 -mb-4 px-4 py-3">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <span className="text-sm font-bold text-[#a4e6ff]">OUR PICKS</span>
                <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${
                  prediction.confidence === 'high'
                    ? 'bg-emerald-900/30 text-[#10b981]'
                    : prediction.confidence === 'medium'
                    ? 'bg-amber-900/30 text-amber-400'
                    : 'bg-[#191c22] text-[#64748b]'
                }`}>
                  {prediction.confidence === 'high' ? 'High Confidence' : prediction.confidence === 'medium' ? 'Medium' : 'Low'}
                </span>
              </div>
            </div>

            {/* Two-column picks: Winner and Spread */}
            <div className="grid grid-cols-2 gap-3 mb-3">
              {/* Winner Prediction */}
              <div className="bg-[#191c22] rounded-lg p-2 border border-[#1e293b]">
                <div className="text-xs text-[#64748b] mb-1">MONEYLINE</div>
                <div className="flex items-center gap-2">
                  <TeamLogo team={prediction.winner} size={20} />
                  <span className="font-bold text-[#f1f5f9] text-sm">{prediction.winner}</span>
                  <span className="text-xs text-[#64748b]">({prediction.winner_prob}%)</span>
                </div>
              </div>

              {/* Spread Pick */}
              {prediction.spread_pick && (
                <div className="bg-[#191c22] rounded-lg p-2 border border-[#1e293b]">
                  <div className="text-xs text-[#64748b] mb-1">SPREAD</div>
                  <div className="flex items-center gap-2">
                    <TeamLogo team={prediction.spread_pick.team} size={20} />
                    <span className="font-bold text-[#f1f5f9] text-sm">
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
              <div className="text-sm mb-2 bg-emerald-900/20 rounded px-2 py-1 border border-emerald-800/30">
                <span className="text-[#10b981] font-medium">Best Value: </span>
                <span className="font-semibold text-[#10b981]">
                  {prediction.best_bet.type === 'total'
                    ? `${prediction.best_bet.team} ${prediction.best_bet.line}`
                    : `${prediction.best_bet.team} ML`
                  }
                </span>
                <span className="text-emerald-400 ml-1">({prediction.best_bet.value_score}%)</span>
              </div>
            )}

            {/* Key Factors */}
            {prediction.factors.length > 0 && (
              <ul className="text-xs text-[#94a3b8] space-y-0.5">
                {prediction.factors.map((factor, i) => (
                  <li key={i} className="flex items-start gap-1">
                    <span className="text-[#475569]">•</span>
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
