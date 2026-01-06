import { useState, useMemo } from 'react';
import { useMarkets } from '@/hooks/useMarkets';
import { useDebounce } from '@/hooks/useDebounce';
import { MarketRow } from '@/components/MarketBoard/MarketRow';
import { MarketFilters } from '@/components/MarketBoard/MarketFilters';
import { ErrorMessage } from '@/components/ui/ErrorMessage';
import type { MarketFilters as Filters } from '@/types/market';

type SortField = 'value' | 'edge' | 'confidence' | 'time';
type SortDirection = 'asc' | 'desc';

function MarketRowSkeleton() {
  return (
    <tr className="animate-pulse">
      <td className="px-6 py-4">
        <div className="h-4 bg-gray-200 rounded w-32" />
        <div className="h-3 bg-gray-100 rounded w-20 mt-1" />
      </td>
      <td className="px-6 py-4">
        <div className="h-5 bg-gray-200 rounded-full w-16" />
      </td>
      <td className="px-6 py-4">
        <div className="h-4 bg-gray-200 rounded w-24" />
      </td>
      <td className="px-6 py-4">
        <div className="h-4 bg-gray-200 rounded w-12" />
      </td>
      <td className="px-6 py-4">
        <div className="h-4 bg-gray-200 rounded w-14" />
      </td>
      <td className="px-6 py-4">
        <div className="h-6 bg-gray-200 rounded-full w-12" />
      </td>
      <td className="px-6 py-4">
        <div className="h-4 bg-gray-200 rounded w-10" />
      </td>
      <td className="px-6 py-4">
        <div className="h-4 bg-gray-200 rounded w-10" />
      </td>
      <td className="px-6 py-4">
        <div className="h-4 bg-gray-200 rounded w-8" />
      </td>
    </tr>
  );
}

function SortableHeader({
  field,
  label,
  currentField,
  direction,
  onSort,
}: {
  field: SortField;
  label: string;
  currentField: SortField;
  direction: SortDirection;
  onSort: (field: SortField) => void;
}) {
  const isActive = currentField === field;
  return (
    <th
      className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100 select-none"
      onClick={() => onSort(field)}
    >
      <div className="flex items-center space-x-1">
        <span>{label}</span>
        {isActive && (
          <span className="text-blue-600">{direction === 'desc' ? '↓' : '↑'}</span>
        )}
      </div>
    </th>
  );
}

export function MarketBoard() {
  const [filters, setFilters] = useState<Filters>({
    algorithm: 'a',
    marketType: null,
    minValueScore: 0,
    minConfidence: 0,
  });

  const [sortField, setSortField] = useState<SortField>('value');
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc');

  // Debounce filters to reduce API calls
  const debouncedFilters = useDebounce(filters, 300);

  const { data: markets, isLoading, error, isFetching } = useMarkets(debouncedFilters);

  // Sort markets client-side
  const sortedMarkets = useMemo(() => {
    if (!markets) return [];

    return [...markets].sort((a, b) => {
      const algorithm = filters.algorithm;
      let aVal: number, bVal: number;

      switch (sortField) {
        case 'value':
          aVal = algorithm === 'a' ? a.algo_a_value_score : a.algo_b_value_score;
          bVal = algorithm === 'a' ? b.algo_a_value_score : b.algo_b_value_score;
          break;
        case 'edge':
          aVal = a.raw_edge;
          bVal = b.raw_edge;
          break;
        case 'confidence':
          aVal = algorithm === 'a' ? a.algo_a_confidence : a.algo_b_confidence;
          bVal = algorithm === 'a' ? b.algo_a_confidence : b.algo_b_confidence;
          break;
        case 'time':
          aVal = a.time_to_tip_minutes;
          bVal = b.time_to_tip_minutes;
          break;
        default:
          return 0;
      }

      return sortDirection === 'desc' ? bVal - aVal : aVal - bVal;
    });
  }, [markets, sortField, sortDirection, filters.algorithm]);

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDirection((prev) => (prev === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortField(field);
      setSortDirection('desc');
    }
  };

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
        <div className="flex items-center space-x-3">
          {isFetching && !isLoading && (
            <div className="w-4 h-4 border-2 border-blue-600 border-t-transparent rounded-full animate-spin" />
          )}
          <span className="text-sm text-gray-500">{markets?.length || 0} opportunities</span>
        </div>
      </div>

      {/* Filters */}
      <div className="card">
        <MarketFilters filters={filters} onChange={setFilters} />
      </div>

      {/* Error State */}
      {error && <ErrorMessage error={error as Error} />}

      {/* Loading State - Skeleton */}
      {isLoading && (
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
                {[...Array(8)].map((_, i) => (
                  <MarketRowSkeleton key={i} />
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Empty State */}
      {!isLoading && !error && sortedMarkets.length === 0 && (
        <div className="text-center py-12">
          <p className="text-gray-500">No betting opportunities found</p>
          <p className="text-sm text-gray-400 mt-1">
            Try adjusting your filters or check back later
          </p>
        </div>
      )}

      {/* Market Table */}
      {!isLoading && sortedMarkets.length > 0 && (
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
                  <SortableHeader
                    field="value"
                    label="Value"
                    currentField={sortField}
                    direction={sortDirection}
                    onSort={handleSort}
                  />
                  <SortableHeader
                    field="edge"
                    label="Edge"
                    currentField={sortField}
                    direction={sortDirection}
                    onSort={handleSort}
                  />
                  <SortableHeader
                    field="confidence"
                    label="Conf"
                    currentField={sortField}
                    direction={sortDirection}
                    onSort={handleSort}
                  />
                  <SortableHeader
                    field="time"
                    label="Tip"
                    currentField={sortField}
                    direction={sortDirection}
                    onSort={handleSort}
                  />
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {sortedMarkets.map((market) => (
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
