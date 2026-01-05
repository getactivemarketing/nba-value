import type { MarketFilters as Filters, MarketType } from '@/types/market';

interface MarketFiltersProps {
  filters: Filters;
  onChange: (filters: Filters) => void;
}

const marketTypes: { value: MarketType | ''; label: string }[] = [
  { value: '', label: 'All Markets' },
  { value: 'spread', label: 'Spreads' },
  { value: 'moneyline', label: 'Moneyline' },
  { value: 'total', label: 'Totals' },
  { value: 'prop', label: 'Player Props' },
];

export function MarketFilters({ filters, onChange }: MarketFiltersProps) {
  return (
    <div className="flex flex-wrap gap-4 items-center">
      {/* Algorithm Toggle */}
      <div className="flex items-center space-x-2">
        <span className="text-sm text-gray-600">Algorithm:</span>
        <div className="flex rounded-lg overflow-hidden border border-gray-200">
          <button
            onClick={() => onChange({ ...filters, algorithm: 'a' })}
            className={`px-3 py-1.5 text-sm font-medium ${
              filters.algorithm === 'a'
                ? 'bg-blue-600 text-white'
                : 'bg-white text-gray-700 hover:bg-gray-50'
            }`}
          >
            A
          </button>
          <button
            onClick={() => onChange({ ...filters, algorithm: 'b' })}
            className={`px-3 py-1.5 text-sm font-medium ${
              filters.algorithm === 'b'
                ? 'bg-blue-600 text-white'
                : 'bg-white text-gray-700 hover:bg-gray-50'
            }`}
          >
            B
          </button>
        </div>
      </div>

      {/* Market Type */}
      <div className="flex items-center space-x-2">
        <label htmlFor="marketType" className="text-sm text-gray-600">
          Market:
        </label>
        <select
          id="marketType"
          value={filters.marketType || ''}
          onChange={(e) =>
            onChange({
              ...filters,
              marketType: (e.target.value as MarketType) || null,
            })
          }
          className="rounded-lg border border-gray-200 px-3 py-1.5 text-sm"
        >
          {marketTypes.map((type) => (
            <option key={type.value} value={type.value}>
              {type.label}
            </option>
          ))}
        </select>
      </div>

      {/* Min Value Score */}
      <div className="flex items-center space-x-2">
        <label htmlFor="minScore" className="text-sm text-gray-600">
          Min Score:
        </label>
        <input
          id="minScore"
          type="number"
          min="0"
          max="100"
          step="10"
          value={filters.minValueScore}
          onChange={(e) =>
            onChange({ ...filters, minValueScore: Number(e.target.value) })
          }
          className="w-20 rounded-lg border border-gray-200 px-3 py-1.5 text-sm"
        />
      </div>

      {/* Min Confidence */}
      <div className="flex items-center space-x-2">
        <label htmlFor="minConf" className="text-sm text-gray-600">
          Min Confidence:
        </label>
        <input
          id="minConf"
          type="number"
          min="0"
          max="2"
          step="0.1"
          value={filters.minConfidence}
          onChange={(e) =>
            onChange({ ...filters, minConfidence: Number(e.target.value) })
          }
          className="w-20 rounded-lg border border-gray-200 px-3 py-1.5 text-sm"
        />
      </div>
    </div>
  );
}
