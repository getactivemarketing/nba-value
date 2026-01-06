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

interface BookValue {
  book: string;
  bookName: string;
  spread: { market: Market; score: number } | null;
  moneyline: { market: Market; score: number } | null;
  total: { market: Market; score: number } | null;
  bestScore: number;
}

export function GameCard({ homeTeam, awayTeam, tipTime, markets, algorithm }: GameCardProps) {
  const minutesToTip = getMinutesToTip(tipTime);
  const tipTimeStr = formatTipTime(tipTime);

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

  return (
    <div className="card">
      {/* Game Header */}
      <div className="flex justify-between items-start mb-4">
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

      {/* Books Table */}
      {activeBooks.length > 0 ? (
        <div className="overflow-x-auto -mx-4 px-4">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-200">
                <th className="text-left py-2 pr-4 text-xs font-medium text-gray-500 uppercase">
                  Book
                </th>
                <th className="text-center py-2 px-2 text-xs font-medium text-gray-500 uppercase">
                  Spread
                </th>
                <th className="text-center py-2 px-2 text-xs font-medium text-gray-500 uppercase">
                  ML
                </th>
                <th className="text-center py-2 px-2 text-xs font-medium text-gray-500 uppercase">
                  Total
                </th>
              </tr>
            </thead>
            <tbody>
              {activeBooks.map((book) => (
                <tr key={book.book} className="border-b border-gray-100 last:border-0">
                  <td className="py-3 pr-4 font-medium text-gray-900">{book.bookName}</td>
                  <td className="py-3 px-2 text-center">
                    {book.spread ? (
                      <Link
                        to={`/bet/${book.spread.market.market_id}`}
                        className="inline-block hover:scale-105 transition-transform"
                      >
                        <ValueScoreBadge score={book.spread.score} size="sm" />
                        <p className="text-xs text-gray-500 mt-0.5">
                          {book.spread.market.outcome_label.includes('home') ? homeTeam : awayTeam}{' '}
                          {book.spread.market.line && book.spread.market.line > 0 ? '+' : ''}
                          {book.spread.market.line}
                        </p>
                      </Link>
                    ) : (
                      <span className="text-gray-300">—</span>
                    )}
                  </td>
                  <td className="py-3 px-2 text-center">
                    {book.moneyline ? (
                      <Link
                        to={`/bet/${book.moneyline.market.market_id}`}
                        className="inline-block hover:scale-105 transition-transform"
                      >
                        <ValueScoreBadge score={book.moneyline.score} size="sm" />
                        <p className="text-xs text-gray-500 mt-0.5">
                          {book.moneyline.market.outcome_label.includes('home') ? homeTeam : awayTeam}
                        </p>
                      </Link>
                    ) : (
                      <span className="text-gray-300">—</span>
                    )}
                  </td>
                  <td className="py-3 px-2 text-center">
                    {book.total ? (
                      <Link
                        to={`/bet/${book.total.market.market_id}`}
                        className="inline-block hover:scale-105 transition-transform"
                      >
                        <ValueScoreBadge score={book.total.score} size="sm" />
                        <p className="text-xs text-gray-500 mt-0.5">
                          {book.total.market.outcome_label === 'over' ? 'O' : 'U'}{' '}
                          {book.total.market.line}
                        </p>
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
