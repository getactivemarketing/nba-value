import { useState } from 'react';
import { EmptyChart } from '@/components/Charts';

type TrendCategory = 'situational' | 'market' | 'team';

const categories: { value: TrendCategory; label: string; description: string; icon: string }[] = [
  {
    value: 'situational',
    label: 'Situational Edge',
    description: 'Back-to-backs, rest days, travel distance',
    icon: 'M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z',
  },
  {
    value: 'market',
    label: 'Market Type',
    description: 'Performance by spread, ML, totals',
    icon: 'M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z',
  },
  {
    value: 'team',
    label: 'Team-Based',
    description: 'Historical edge by team matchup',
    icon: 'M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z',
  },
];

const upcomingFeatures = [
  'Situational patterns (back-to-backs, rest advantages, home/away)',
  'Market type performance breakdown (spreads vs ML vs totals)',
  'Team-specific edge analysis (which teams the model predicts well)',
  'Time-of-day and day-of-week patterns',
  'Line movement correlation with value scores',
];

export function Trends() {
  const [category, setCategory] = useState<TrendCategory>('situational');

  // Backend returns empty array for trends until we have historical data
  const trends: unknown[] = [];
  const isLoading = false;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Trend Explorer</h1>
        <p className="text-sm text-gray-500 mt-1">
          Analyze where the model historically finds the most edge
        </p>
      </div>

      {/* Category Selector Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {categories.map((cat) => (
          <button
            key={cat.value}
            onClick={() => setCategory(cat.value)}
            className={`card text-left transition-all hover:shadow-md ${
              category === cat.value
                ? 'ring-2 ring-blue-500 border-blue-500'
                : 'hover:border-gray-300'
            }`}
          >
            <div className="flex items-start space-x-3">
              <div
                className={`p-2 rounded-lg ${
                  category === cat.value ? 'bg-blue-100' : 'bg-gray-100'
                }`}
              >
                <svg
                  className={`w-5 h-5 ${
                    category === cat.value ? 'text-blue-600' : 'text-gray-500'
                  }`}
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={1.5}
                    d={cat.icon}
                  />
                </svg>
              </div>
              <div>
                <h3 className="font-medium text-gray-900">{cat.label}</h3>
                <p className="text-sm text-gray-500 mt-0.5">{cat.description}</p>
              </div>
            </div>
          </button>
        ))}
      </div>

      {/* Main Content */}
      <div className="card">
        {isLoading ? (
          <div className="flex justify-center py-12">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600" />
          </div>
        ) : trends.length === 0 ? (
          <div className="text-center py-8">
            <EmptyChart message="Trend analysis requires historical data" height={180} />
            <div className="mt-6 max-w-lg mx-auto">
              <h3 className="text-lg font-medium text-gray-900">Coming Soon</h3>
              <p className="text-sm text-gray-500 mt-2">
                Once we accumulate enough betting results, this page will reveal powerful
                insights about where and when our model finds the most value.
              </p>
              <div className="mt-6 text-left">
                <h4 className="text-sm font-medium text-gray-700 mb-3">
                  Upcoming features:
                </h4>
                <ul className="space-y-2">
                  {upcomingFeatures.map((feature, i) => (
                    <li key={i} className="flex items-start text-sm text-gray-500">
                      <svg
                        className="w-4 h-4 text-blue-500 mr-2 mt-0.5 flex-shrink-0"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                        />
                      </svg>
                      {feature}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        ) : (
          <div>
            {/* Future: Render actual trend data tables/charts */}
            <p className="text-gray-500">Trend data will be displayed here</p>
          </div>
        )}
      </div>
    </div>
  );
}
