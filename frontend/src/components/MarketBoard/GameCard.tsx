import { useState } from 'react';
import { Link } from 'react-router-dom';
import { getTeamLogo, getTeamColor } from '@/lib/teamLogos';
import type { Market, Algorithm } from '@/types/market';
import type { TeamTrends, GamePrediction } from '@/lib/api';

interface GameCardProps {
  gameId: string;
  homeTeam: string;
  awayTeam: string;
  tipTime: string;
  markets: Market[];
  algorithm: Algorithm;
  homeTrends?: TeamTrends;
  awayTrends?: TeamTrends;
  prediction?: GamePrediction | null;
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
function getTeamSpreadValue(markets: Market[], _team: string, isHome: boolean, algorithm: Algorithm): { score: number; line: number | null; marketId: string } | null {
  const spreadMarkets = markets.filter(m =>
    m.market_type === 'spread' &&
    ((isHome && m.outcome_label.includes('home')) || (!isHome && m.outcome_label.includes('away')))
  );

  if (spreadMarkets.length === 0) return null;

  const best = spreadMarkets.reduce((a, b) => {
    const aScore = algorithm === 'a' ? a.algo_a_value_score : a.algo_b_value_score;
    const bScore = algorithm === 'a' ? b.algo_a_value_score : b.algo_b_value_score;
    return bScore > aScore ? b : a;
  });

  const score = algorithm === 'a' ? best.algo_a_value_score : best.algo_b_value_score;
  return { score, line: best.line, marketId: best.market_id };
}

// Get consensus spread line
function getConsensusLine(markets: Market[], type: string): number | null {
  const lines = markets
    .filter(m => m.market_type === type && m.line !== null)
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

// Value score badge matching Covers style
function ValueBadge({ score, size = 'md' }: { score: number; size?: 'sm' | 'md' | 'lg' }) {
  const getColor = (s: number) => {
    if (s >= 70) return 'bg-emerald-600';
    if (s >= 50) return 'bg-amber-500';
    return 'bg-slate-400';
  };

  const sizeClasses = {
    sm: 'text-xs px-2 py-0.5',
    md: 'text-sm px-3 py-1',
    lg: 'text-base px-4 py-1.5 font-bold',
  };

  return (
    <span className={`${getColor(score)} ${sizeClasses[size]} text-white rounded font-semibold`}>
      {Math.round(score)}%
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

export function GameCard({ homeTeam, awayTeam, tipTime, markets, algorithm, homeTrends, awayTrends, prediction }: GameCardProps) {
  const { date, time } = formatGameTime(tipTime);

  // Get spread values for each team
  const awaySpread = getTeamSpreadValue(markets, awayTeam, false, algorithm);
  const homeSpread = getTeamSpreadValue(markets, homeTeam, true, algorithm);

  // Get consensus lines
  const spreadLine = getConsensusLine(markets, 'spread');
  const totalLine = getTotalLine(markets);

  // Get best market for linking
  const bestMarketId = awaySpread?.marketId || homeSpread?.marketId || '';

  return (
    <Link
      to={bestMarketId ? `/bet/${bestMarketId}` : '#'}
      className="block bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden hover:shadow-md transition-shadow"
    >
      {/* Dark Header */}
      <div className="bg-slate-800 text-white px-4 py-2 text-sm font-medium tracking-wide">
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
              <div className="text-xl font-bold text-gray-900">{awayTeam}</div>
              <div className="flex items-center gap-2 mt-1">
                {awaySpread && (
                  <>
                    <ValueBadge score={awaySpread.score} size="lg" />
                    <span className="text-lg font-semibold text-gray-700">
                      {formatLine(spreadLine ? -spreadLine : awaySpread.line)}
                    </span>
                  </>
                )}
              </div>
            </div>
          </div>

          {/* Center - Date/Time/Total */}
          <div className="text-center px-4">
            <div className="text-sm font-medium text-gray-900">{date}</div>
            <div className="text-sm text-gray-500">{time}</div>
            {totalLine && (
              <div className="mt-2 text-sm">
                <span className="text-gray-500">o/u</span>{' '}
                <span className="font-semibold text-gray-900">{totalLine}</span>
              </div>
            )}
          </div>

          {/* Home Team */}
          <div className="flex items-center gap-3">
            <div className="text-right">
              <div className="text-xl font-bold text-gray-900">{homeTeam}</div>
              <div className="flex items-center justify-end gap-2 mt-1">
                {homeSpread && (
                  <>
                    <span className="text-lg font-semibold text-gray-700">
                      {formatLine(spreadLine)}
                    </span>
                    <ValueBadge score={homeSpread.score} size="lg" />
                  </>
                )}
              </div>
            </div>
            <TeamLogo team={homeTeam} size={56} />
          </div>
        </div>

        {/* Team Stats */}
        <div className="mt-4 pt-4 border-t border-gray-100">
          {/* Stats Table */}
          <div className="grid grid-cols-3 gap-2 text-xs">
            {/* Header Row */}
            <div className="font-medium text-gray-500">{awayTeam}</div>
            <div className="text-center font-medium text-gray-500"></div>
            <div className="text-right font-medium text-gray-500">{homeTeam}</div>

            {/* Overall Record */}
            <div className="font-semibold text-gray-800">{awayTrends?.record || '0-0'}</div>
            <div className="text-center text-gray-400">Record</div>
            <div className="text-right font-semibold text-gray-800">{homeTrends?.record || '0-0'}</div>

            {/* Home/Road Record */}
            <div className="text-gray-600">{awayTrends?.away_record || '0-0'} <span className="text-gray-400">Road</span></div>
            <div className="text-center text-gray-400">Split</div>
            <div className="text-right text-gray-600"><span className="text-gray-400">Home</span> {homeTrends?.home_record || '0-0'}</div>

            {/* L10 Record */}
            <div className="text-gray-600">{awayTrends?.l10_record || '0-0'}</div>
            <div className="text-center text-gray-400">L10</div>
            <div className="text-right text-gray-600">{homeTrends?.l10_record || '0-0'}</div>

            {/* Net Rating L10 */}
            <div className={`font-medium ${(awayTrends?.net_rtg_l10 ?? 0) > 0 ? 'text-green-600' : (awayTrends?.net_rtg_l10 ?? 0) < 0 ? 'text-red-500' : 'text-gray-500'}`}>
              {awayTrends?.net_rtg_l10 != null ? `${(awayTrends?.net_rtg_l10 ?? 0) > 0 ? '+' : ''}${(awayTrends?.net_rtg_l10 ?? 0).toFixed(1)}` : '-'}
            </div>
            <div className="text-center text-gray-400">Net Rtg</div>
            <div className={`text-right font-medium ${(homeTrends?.net_rtg_l10 ?? 0) > 0 ? 'text-green-600' : (homeTrends?.net_rtg_l10 ?? 0) < 0 ? 'text-red-500' : 'text-gray-500'}`}>
              {homeTrends?.net_rtg_l10 != null ? `${(homeTrends?.net_rtg_l10 ?? 0) > 0 ? '+' : ''}${(homeTrends?.net_rtg_l10 ?? 0).toFixed(1)}` : '-'}
            </div>
          </div>

          {/* B2B Indicators */}
          {(awayTrends?.is_b2b || homeTrends?.is_b2b) && (
            <div className="mt-2 flex justify-between text-xs">
              <div>{awayTrends?.is_b2b && <span className="text-amber-600 font-medium">B2B</span>}</div>
              <div>{homeTrends?.is_b2b && <span className="text-amber-600 font-medium">B2B</span>}</div>
            </div>
          )}
        </div>

        {/* Prediction Section */}
        {prediction && (
          <div className="mt-4 pt-4 border-t border-gray-200 bg-gradient-to-r from-slate-50 to-slate-100 -mx-4 -mb-4 px-4 py-3">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <span className="text-sm font-bold text-slate-700">OUR PICK</span>
                <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${
                  prediction.confidence === 'high'
                    ? 'bg-emerald-100 text-emerald-700'
                    : prediction.confidence === 'medium'
                    ? 'bg-amber-100 text-amber-700'
                    : 'bg-gray-100 text-gray-600'
                }`}>
                  {prediction.confidence === 'high' ? 'High Confidence' : prediction.confidence === 'medium' ? 'Medium' : 'Low'}
                </span>
              </div>
            </div>

            {/* Winner Prediction */}
            <div className="flex items-center gap-2 mb-2">
              <TeamLogo team={prediction.winner} size={24} />
              <span className="font-bold text-gray-900">{prediction.winner} to win</span>
              <span className="text-sm text-gray-500">({prediction.winner_prob}%)</span>
            </div>

            {/* Best Value Bet */}
            {prediction.best_bet && prediction.best_bet.value_score >= 50 && (
              <div className="text-sm mb-2">
                <span className="text-gray-600">Best Value: </span>
                <span className="font-semibold text-emerald-700">
                  {prediction.best_bet.team} {prediction.best_bet.line != null ? (prediction.best_bet.line > 0 ? `+${prediction.best_bet.line}` : prediction.best_bet.line) : ''}
                  {prediction.best_bet.type === 'moneyline' ? ' ML' : prediction.best_bet.type === 'total' ? ` (${prediction.best_bet.line})` : ''}
                </span>
                <span className="text-gray-500 ml-1">({prediction.best_bet.value_score}% value)</span>
              </div>
            )}

            {/* Key Factors */}
            {prediction.factors.length > 0 && (
              <ul className="text-xs text-gray-600 space-y-0.5">
                {prediction.factors.map((factor, i) => (
                  <li key={i} className="flex items-start gap-1">
                    <span className="text-slate-400">•</span>
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
