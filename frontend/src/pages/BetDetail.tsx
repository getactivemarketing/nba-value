import { useParams, Link } from 'react-router-dom';
import { useBetDetail, useBetHistory } from '@/hooks/useMarkets';
import { ValueScoreBadge } from '@/components/ui/ValueScoreBadge';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { ErrorMessage } from '@/components/ui/ErrorMessage';

export function BetDetail() {
  const { marketId } = useParams<{ marketId: string }>();
  const { data: bet, isLoading, error } = useBetDetail(marketId || '');
  const { data: history } = useBetHistory(marketId || '');

  if (isLoading) {
    return (
      <div className="flex justify-center py-12">
        <LoadingSpinner size="lg" />
      </div>
    );
  }

  if (error) {
    return <ErrorMessage error={error as Error} />;
  }

  if (!bet) {
    return (
      <div className="text-center py-12">
        <p className="text-gray-500">Bet not found</p>
        <Link to="/" className="text-blue-600 hover:underline mt-2 inline-block">
          Back to Market Board
        </Link>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Back Link */}
      <Link to="/" className="text-sm text-gray-500 hover:text-gray-700">
        ← Back to Market Board
      </Link>

      {/* Header */}
      <div className="card">
        <div className="flex justify-between items-start">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">
              {bet.away_team} @ {bet.home_team}
            </h1>
            <p className="text-gray-500 mt-1">
              {bet.market_type} • {bet.outcome_label}
              {bet.line && ` (${bet.line > 0 ? '+' : ''}${bet.line})`}
            </p>
          </div>
          <div className="text-right">
            <ValueScoreBadge score={bet.recommended_score} size="lg" />
            <p className="text-sm text-gray-500 mt-1">
              Algorithm {bet.active_algorithm.toUpperCase()}
            </p>
          </div>
        </div>
      </div>

      {/* Probability Comparison */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="card">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">
            Probability Comparison
          </h2>
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <span className="text-gray-600">Model Probability</span>
              <span className="text-xl font-bold text-green-600">
                {(bet.p_true * 100).toFixed(1)}%
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-600">Market Probability</span>
              <span className="text-xl font-bold text-gray-600">
                {(bet.p_market * 100).toFixed(1)}%
              </span>
            </div>
            <div className="border-t pt-4 flex justify-between items-center">
              <span className="text-gray-600">Edge</span>
              <span
                className={`text-xl font-bold ${
                  bet.raw_edge > 0 ? 'text-green-600' : 'text-red-600'
                }`}
              >
                {bet.raw_edge > 0 ? '+' : ''}
                {(bet.raw_edge * 100).toFixed(1)}%
              </span>
            </div>
          </div>
        </div>

        <div className="card">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Odds Info</h2>
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <span className="text-gray-600">Decimal Odds</span>
              <span className="text-xl font-bold">{bet.odds_decimal.toFixed(2)}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-600">American Odds</span>
              <span className="text-xl font-bold">
                {bet.odds_american !== undefined && bet.odds_american > 0 ? '+' : ''}
                {bet.odds_american ?? '-'}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-600">Book</span>
              <span className="text-xl font-bold">{bet.book}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Algorithm Comparison */}
      <div className="card">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">
          Algorithm Comparison
        </h2>
        <div className="grid grid-cols-2 gap-8">
          {/* Algorithm A */}
          <div>
            <div className="flex items-center justify-between mb-4">
              <h3 className="font-medium text-gray-700">Algorithm A</h3>
              <ValueScoreBadge score={bet.algo_a.value_score} />
            </div>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-500">Edge Score</span>
                <span>{bet.algo_a.edge_score?.toFixed(3) || '-'}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">Confidence</span>
                <span>{bet.algo_a.confidence.final_multiplier.toFixed(2)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">Market Quality</span>
                <span>{bet.algo_a.market_quality.final_multiplier.toFixed(2)}</span>
              </div>
            </div>
          </div>

          {/* Algorithm B */}
          <div>
            <div className="flex items-center justify-between mb-4">
              <h3 className="font-medium text-gray-700">Algorithm B</h3>
              <ValueScoreBadge score={bet.algo_b.value_score} />
            </div>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-500">Combined Edge</span>
                <span>{bet.algo_b.combined_edge?.toFixed(3) || '-'}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">Confidence</span>
                <span>{bet.algo_b.confidence.final_multiplier.toFixed(2)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">Market Quality</span>
                <span>{bet.algo_b.market_quality.final_multiplier.toFixed(2)}</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* History placeholder */}
      {history && history.length > 0 && (
        <div className="card">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Score History</h2>
          <p className="text-sm text-gray-500">
            {history.length} snapshots over the last 24 hours
          </p>
        </div>
      )}
    </div>
  );
}
