import { Link } from 'react-router-dom';
import { ValueScoreBadge } from '@/components/ui/ValueScoreBadge';
import type { Market, Algorithm } from '@/types/market';

interface MarketRowProps {
  market: Market;
  algorithm: Algorithm;
}

export function MarketRow({ market, algorithm }: MarketRowProps) {
  const valueScore =
    algorithm === 'a' ? market.algo_a_value_score : market.algo_b_value_score;
  const confidence =
    algorithm === 'a' ? market.algo_a_confidence : market.algo_b_confidence;

  const formatOdds = (odds: number) => {
    if (odds >= 2) {
      return `+${Math.round((odds - 1) * 100)}`;
    }
    return `-${Math.round(100 / (odds - 1))}`;
  };

  const formatLine = (line: number | null, marketType: string) => {
    if (line === null) return '-';
    if (marketType === 'spread') {
      return line > 0 ? `+${line}` : line.toString();
    }
    return line.toString();
  };

  const timeToTip =
    market.time_to_tip_minutes > 60
      ? `${Math.floor(market.time_to_tip_minutes / 60)}h`
      : `${market.time_to_tip_minutes}m`;

  return (
    <tr className="hover:bg-gray-50 transition-colors">
      <td className="px-6 py-4 whitespace-nowrap">
        <Link to={`/bet/${market.market_id}`} className="hover:text-blue-600">
          <div className="text-sm font-medium text-gray-900">
            {market.game?.away_team} @ {market.game?.home_team}
          </div>
          <div className="text-xs text-gray-500">{market.book}</div>
        </Link>
      </td>
      <td className="px-6 py-4 whitespace-nowrap">
        <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-800">
          {market.market_type}
        </span>
      </td>
      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
        {market.outcome_label}
      </td>
      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
        {formatLine(market.line, market.market_type)}
      </td>
      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
        {formatOdds(market.odds_decimal)}
      </td>
      <td className="px-6 py-4 whitespace-nowrap">
        <ValueScoreBadge score={valueScore} />
      </td>
      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
        {(market.raw_edge * 100).toFixed(1)}%
      </td>
      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
        {confidence.toFixed(2)}
      </td>
      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{timeToTip}</td>
    </tr>
  );
}
