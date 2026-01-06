import { Link } from 'react-router-dom';
import { ValueScoreBadge } from '@/components/ui/ValueScoreBadge';
import type { Market, Algorithm } from '@/types/market';

// Books to display (in order)
const DISPLAY_BOOKS = [
  { key: 'draftkings', name: 'DraftKings' },
  { key: 'fanduel', name: 'FanDuel' },
  { key: 'betmgm', name: 'BetMGM' },
  { key: 'betus', name: 'BetUS' },
  { key: 'hardrock', name: 'Hard Rock' },
  { key: 'fanatics', name: 'Fanatics' },
];

interface GameCardProps {
  gameId: string;
  homeTeam: string;
  awayTeam: string;
  tipTime: string;
  markets: Market[];
  algorithm: Algorithm;
}

function formatTipTime(tipTime: string): string {
  const date = new Date(tipTime);
  return date.toLocaleTimeString('en-US', {
    hour: 'numeric',
    minute: '2-digit',
    hour12: true,
  });
}

function getMinutesToTip(tipTime: string): number {
  const now = new Date();
  const tip = new Date(tipTime);
  return Math.round((tip.getTime() - now.getTime()) / 60000);
}

function formatOdds(odds: number | undefined): string {
  if (!odds) return '';
  return odds > 0 ? `+${odds}` : `${odds}`;
}

function formatLine(line: number | null, showPlus = true): string {
  if (line === null) return '';
  if (showPlus && line > 0) return `+${line}`;
  return `${line}`;
}

interface BookValue {
  book: string;
  bookName: string;
  spread: { market: Market; score: number } | null;
  moneyline: { market: Market; score: number } | null;
  total: { market: Market; score: number } | null;
  bestScore: number;
}

// Get consensus line (most common value)
function getConsensusLine(markets: Market[], type: string): { line: number | null; count: number } {
  const lines = markets
    .filter(m => m.market_type === type && m.line !== null)
    .map(m => m.line as number);

  if (lines.length === 0) return { line: null, count: 0 };

  const counts = lines.reduce((acc, line) => {
    acc[line] = (acc[line] || 0) + 1;
    return acc;
  }, {} as Record<number, number>);

  const entries = Object.entries(counts);
  const [mostCommon] = entries.sort(([, a], [, b]) => b - a);
  return { line: parseFloat(mostCommon[0]), count: mostCommon[1] };
}

export function GameCard({ homeTeam, awayTeam, tipTime, markets, algorithm }: GameCardProps) {
  const minutesToTip = getMinutesToTip(tipTime);
  const tipTimeStr = formatTipTime(tipTime);

  // Get consensus spread and total
  const consensusSpread = getConsensusLine(markets, 'spread');
  const consensusTotal = getConsensusLine(markets, 'total');

  // Group markets by book and type, find best value for each
  const bookValues: BookValue[] = DISPLAY_BOOKS.map(({ key, name }) => {
    const bookMarkets = markets.filter((m) => m.book === key);

    const getBestMarket = (type: string) => {
      const typeMarkets = bookMarkets.filter((m) => m.market_type === type);
      if (typeMarkets.length === 0) return null;

      // Get highest value score market
      const best = typeMarkets.reduce((a, b) => {
        const aScore = algorithm === 'a' ? a.algo_a_value_score : a.algo_b_value_score;
        const bScore = algorithm === 'a' ? b.algo_a_value_score : b.algo_b_value_score;
        return bScore > aScore ? b : a;
      });

      const score = algorithm === 'a' ? best.algo_a_value_score : best.algo_b_value_score;
      return { market: best, score };
    };

    const spread = getBestMarket('spread');
    const moneyline = getBestMarket('moneyline');
    const total = getBestMarket('total');

    const scores = [spread?.score || 0, moneyline?.score || 0, total?.score || 0];
    const bestScore = Math.max(...scores);

    return { book: key, bookName: name, spread, moneyline, total, bestScore };
  });

  // Filter to only books that have markets
  const activeBooks = bookValues.filter(
    (b) => b.spread || b.moneyline || b.total
  );

  // Overall best value for this game
  const gameBestScore = Math.max(...activeBooks.map((b) => b.bestScore), 0);

  // Calculate edge range for this game
  const edges = markets.map(m => m.raw_edge * 100);
  const avgEdge = edges.length > 0 ? edges.reduce((a, b) => a + b, 0) / edges.length : 0;

  return (
    <div className="card">
      {/* Game Header */}
      <div className="flex justify-between items-start mb-3">
        <div>
          <h3 className="text-lg font-bold text-gray-900">
            {awayTeam} @ {homeTeam}
          </h3>
          <p className="text-sm text-gray-500">
            {tipTimeStr} • {minutesToTip > 0 ? `${Math.floor(minutesToTip / 60)}h ${minutesToTip % 60}m to tip` : 'In Progress'}
          </p>
        </div>
        <div className="text-right">
          <p className="text-xs text-gray-500 mb-1">Best Value</p>
          <ValueScoreBadge score={gameBestScore} size="lg" />
        </div>
      </div>

      {/* Game Summary - Consensus Lines */}
      <div className="flex gap-4 mb-4 p-2 bg-gray-50 rounded-lg text-sm">
        {consensusSpread.line !== null && (
          <div className="flex-1">
            <span className="text-gray-500">Spread:</span>{' '}
            <span className="font-semibold">
              {homeTeam} {formatLine(consensusSpread.line)}
            </span>
          </div>
        )}
        {consensusTotal.line !== null && (
          <div className="flex-1">
            <span className="text-gray-500">Total:</span>{' '}
            <span className="font-semibold">O/U {consensusTotal.line}</span>
          </div>
        )}
        <div className="flex-1">
          <span className="text-gray-500">Avg Edge:</span>{' '}
          <span className={`font-semibold ${avgEdge >= 10 ? 'text-green-600' : avgEdge >= 5 ? 'text-yellow-600' : 'text-gray-600'}`}>
            {avgEdge.toFixed(1)}%
          </span>
        </div>
      </div>

      {/* Books Table */}
      {activeBooks.length > 0 ? (
        <div className="overflow-x-auto -mx-4 px-4">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-200">
                <th className="text-left py-2 pr-2 text-xs font-medium text-gray-500 uppercase">
                  Book
                </th>
                <th className="text-center py-2 px-1 text-xs font-medium text-gray-500 uppercase">
                  Spread
                </th>
                <th className="text-center py-2 px-1 text-xs font-medium text-gray-500 uppercase">
                  ML
                </th>
                <th className="text-center py-2 px-1 text-xs font-medium text-gray-500 uppercase">
                  Total
                </th>
              </tr>
            </thead>
            <tbody>
              {activeBooks.map((book) => (
                <tr key={book.book} className="border-b border-gray-100 last:border-0">
                  <td className="py-2 pr-2 font-medium text-gray-900 text-xs">{book.bookName}</td>
                  <td className="py-2 px-1 text-center">
                    {book.spread ? (
                      <Link
                        to={`/bet/${book.spread.market.market_id}`}
                        className="inline-block hover:scale-105 transition-transform"
                      >
                        <ValueScoreBadge score={book.spread.score} size="sm" />
                        <div className="text-xs text-gray-600 mt-0.5">
                          <span className="font-medium">
                            {book.spread.market.outcome_label.includes('home') ? homeTeam : awayTeam}{' '}
                            {formatLine(book.spread.market.line)}
                          </span>
                          <span className="text-gray-400 ml-1">
                            ({formatOdds(book.spread.market.odds_american)})
                          </span>
                        </div>
                      </Link>
                    ) : (
                      <span className="text-gray-300">—</span>
                    )}
                  </td>
                  <td className="py-2 px-1 text-center">
                    {book.moneyline ? (
                      <Link
                        to={`/bet/${book.moneyline.market.market_id}`}
                        className="inline-block hover:scale-105 transition-transform"
                      >
                        <ValueScoreBadge score={book.moneyline.score} size="sm" />
                        <div className="text-xs text-gray-600 mt-0.5">
                          <span className="font-medium">
                            {book.moneyline.market.outcome_label.includes('home') ? homeTeam : awayTeam}
                          </span>
                          <span className="text-gray-400 ml-1">
                            ({formatOdds(book.moneyline.market.odds_american)})
                          </span>
                        </div>
                      </Link>
                    ) : (
                      <span className="text-gray-300">—</span>
                    )}
                  </td>
                  <td className="py-2 px-1 text-center">
                    {book.total ? (
                      <Link
                        to={`/bet/${book.total.market.market_id}`}
                        className="inline-block hover:scale-105 transition-transform"
                      >
                        <ValueScoreBadge score={book.total.score} size="sm" />
                        <div className="text-xs text-gray-600 mt-0.5">
                          <span className="font-medium">
                            {book.total.market.outcome_label === 'over' ? 'O' : 'U'}{' '}
                            {book.total.market.line}
                          </span>
                          <span className="text-gray-400 ml-1">
                            ({formatOdds(book.total.market.odds_american)})
                          </span>
                        </div>
                      </Link>
                    ) : (
                      <span className="text-gray-300">—</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="text-sm text-gray-500 text-center py-4">
          No markets available for selected books
        </p>
      )}
    </div>
  );
}
