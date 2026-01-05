import { useState } from 'react';
import { useMarkets } from '@/hooks/useMarkets';
import { MarketRow } from '@/components/MarketBoard/MarketRow';
import { MarketFilters } from '@/components/MarketBoard/MarketFilters';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { ErrorMessage } from '@/components/ui/ErrorMessage';
import type { MarketFilters as Filters } from '@/types/market';

export function MarketBoard() {
  const [filters, setFilters] = useState<Filters>({
    algorithm: 'a',
    marketType: null,
    minValueScore: 0,
    minConfidence: 0,
  });

  const { data: markets, isLoading, error } = useMarkets(filters);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Market Board</h1>
          <p className="text-sm text-gray-500 mt-1">
            Betting opportunities ranked by Value Score
          </p>
        </div>
        <div className="text-sm text-gray-500">
          {markets?.length || 0} opportunities
        </div>
      </div>

      {/* Filters */}
      <div className="card">
        <MarketFilters filters={filters} onChange={setFilters} />
      </div>

      {/* Loading State */}
      {isLoading && (
        <div className="flex justify-center py-12">
          <LoadingSpinner size="lg" />
        </div>
      )}

      {/* Error State */}
      {error && <ErrorMessage error={error as Error} />}

      {/* Empty State */}
      {!isLoading && !error && markets?.length === 0 && (
        <div className="text-center py-12">
          <p className="text-gray-500">No betting opportunities found</p>
          <p className="text-sm text-gray-400 mt-1">
            Try adjusting your filters or check back later
          </p>
        </div>
      )}

      {/* Market Table */}
      {!isLoading && markets && markets.length > 0 && (
        <div className="card overflow-hidden p-0">
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Game
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Type
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Pick
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Line
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Odds
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Value
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Edge
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Conf
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Tip
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {markets.map((market) => (
                  <MarketRow
                    key={market.market_id}
                    market={market}
                    algorithm={filters.algorithm}
                  />
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
