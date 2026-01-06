import { useState } from 'react';
import { Link } from 'react-router-dom';
import { getTeamLogo, getTeamColor } from '@/lib/teamLogos';
import type { Market, Algorithm } from '@/types/market';
import type { TeamTrends } from '@/lib/api';

interface GameCardProps {
  gameId: string;
  homeTeam: string;
  awayTeam: string;
  tipTime: string;
  markets: Market[];
  algorithm: Algorithm;
  homeTrends?: TeamTrends;
  awayTrends?: TeamTrends;
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

export function GameCard({ homeTeam, awayTeam, tipTime, markets, algorithm, homeTrends, awayTrends }: GameCardProps) {
  const { date, time } = formatGameTime(tipTime);
  const [showDetails, setShowDetails] = useState(false);

  // Get spread values for each team
  const awaySpread = getTeamSpreadValue(markets, awayTeam, false, algorithm);
  const homeSpread = getTeamSpreadValue(markets, homeTeam, true, algorithm);

  // Get consensus lines
  const spreadLine = getConsensusLine(markets, 'spread');
  const totalLine = getTotalLine(markets);

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
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
          <div className="grid grid-cols-3 gap-4 text-xs">
            {/* Away Stats */}
            <div className="space-y-1">
              {awayTrends && (
                <>
                  <div className="text-gray-500">
                    ({awayTrends.record} {awayTrends.is_b2b ? '' : 'Road'})
                  </div>
                  {awayTrends.win_pct_l10 !== null && (
                    <div className="text-gray-500">
                      L10: {Math.round(awayTrends.win_pct_l10 * 10)}-{10 - Math.round(awayTrends.win_pct_l10 * 10)}
                    </div>
                  )}
                  {awayTrends.is_b2b && (
                    <div className="text-amber-600 font-medium">B2B</div>
                  )}
                </>
              )}
            </div>

            {/* Center Stats Labels */}
            <div className="text-center space-y-1 text-gray-500 font-medium">
              <div>Win/Loss</div>
              <div>Last 10</div>
              <div>Net Rtg</div>
            </div>

            {/* Home Stats */}
            <div className="text-right space-y-1">
              {homeTrends && (
                <>
                  <div className="text-gray-500">
                    ({homeTrends.record} Home)
                  </div>
                  {homeTrends.win_pct_l10 !== null && (
                    <div className="text-gray-500">
                      L10: {Math.round(homeTrends.win_pct_l10 * 10)}-{10 - Math.round(homeTrends.win_pct_l10 * 10)}
                    </div>
                  )}
                  {homeTrends.is_b2b && (
                    <div className="text-amber-600 font-medium">B2B</div>
                  )}
                </>
              )}
            </div>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="mt-4 flex items-center gap-4">
          <button
            onClick={() => setShowDetails(!showDetails)}
            className="px-4 py-2 bg-orange-500 hover:bg-orange-600 text-white text-sm font-medium rounded transition-colors"
          >
            Matchup
          </button>

          <div className="flex items-center gap-4 text-sm">
            <button className="text-gray-600 hover:text-gray-900 font-medium">
              Consensus
            </button>
            <Link
              to={`/bet/${awaySpread?.marketId || homeSpread?.marketId || ''}`}
              className="text-gray-600 hover:text-gray-900 font-medium"
            >
              Picks
            </Link>
            <button
              onClick={() => setShowDetails(!showDetails)}
              className="text-gray-600 hover:text-gray-900 font-medium flex items-center gap-1"
            >
              Line Moves
              <svg className={`w-4 h-4 transition-transform ${showDetails ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </button>
          </div>
        </div>

        {/* Expanded Details */}
        {showDetails && (
          <div className="mt-4 pt-4 border-t border-gray-100">
            <h4 className="font-semibold text-gray-900 mb-3">Best Lines by Book</h4>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-gray-500 border-b">
                    <th className="pb-2 font-medium">Book</th>
                    <th className="pb-2 font-medium text-center">Spread</th>
                    <th className="pb-2 font-medium text-center">ML</th>
                    <th className="pb-2 font-medium text-center">Total</th>
                  </tr>
                </thead>
                <tbody>
                  {['draftkings', 'fanduel', 'betmgm', 'betus'].map(book => {
                    const bookMarkets = markets.filter(m => m.book === book);
                    if (bookMarkets.length === 0) return null;

                    const spread = bookMarkets.find(m => m.market_type === 'spread');
                    const ml = bookMarkets.find(m => m.market_type === 'moneyline');
                    const total = bookMarkets.find(m => m.market_type === 'total');

                    return (
                      <tr key={book} className="border-b border-gray-50">
                        <td className="py-2 font-medium capitalize">{book}</td>
                        <td className="py-2 text-center">
                          {spread && (
                            <Link to={`/bet/${spread.market_id}`} className="hover:text-blue-600">
                              {formatLine(spread.line)} ({(spread.odds_american ?? 0) > 0 ? '+' : ''}{spread.odds_american})
                            </Link>
                          )}
                        </td>
                        <td className="py-2 text-center">
                          {ml && (
                            <Link to={`/bet/${ml.market_id}`} className="hover:text-blue-600">
                              {(ml.odds_american ?? 0) > 0 ? '+' : ''}{ml.odds_american}
                            </Link>
                          )}
                        </td>
                        <td className="py-2 text-center">
                          {total && (
                            <Link to={`/bet/${total.market_id}`} className="hover:text-blue-600">
                              {total.outcome_label === 'over' ? 'O' : 'U'} {total.line}
                            </Link>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
