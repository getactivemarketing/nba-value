import { useState, useEffect } from 'react';
import { api, TopPicksResponse, TopPick } from '@/lib/api';

function ValueBadge({ score }: { score: number }) {
  const getColor = (s: number) => {
    if (s >= 90) return 'bg-emerald-600';
    if (s >= 70) return 'bg-emerald-500';
    if (s >= 55) return 'bg-amber-500';
    return 'bg-slate-400';
  };

  return (
    <span className={`${getColor(score)} text-white text-xs px-2 py-0.5 rounded font-semibold`}>
      {Math.round(score)}%
    </span>
  );
}

function EdgeBadge({ edge }: { edge: number }) {
  const getColor = (e: number) => {
    if (e >= 15) return 'text-emerald-600';
    if (e >= 10) return 'text-emerald-500';
    if (e >= 5) return 'text-amber-600';
    return 'text-gray-600';
  };

  return (
    <span className={`${getColor(edge)} font-semibold`}>
      +{edge.toFixed(1)}%
    </span>
  );
}

function BestEdgesCard({ picks }: { picks: TopPick[] }) {
  if (picks.length === 0) return null;

  const top3 = picks.slice(0, 3);

  return (
    <div className="bg-gradient-to-br from-emerald-50 to-emerald-100 rounded-lg border border-emerald-200 p-4">
      <h3 className="font-bold text-emerald-800 mb-3 flex items-center gap-2">
        <span className="text-lg">Best Edge Plays</span>
      </h3>
      <div className="space-y-2">
        {top3.map((pick, i) => (
          <div key={i} className="flex items-center justify-between bg-white rounded-lg px-3 py-2 shadow-sm">
            <div>
              <span className="font-bold text-gray-900">{pick.pick}</span>
              <span className="text-gray-500 text-sm ml-2">({pick.game})</span>
            </div>
            <div className="flex items-center gap-3">
              <EdgeBadge edge={pick.edge} />
              <span className="text-sm text-gray-500">
                Model: {pick.model_prob.toFixed(0)}% vs Market: {pick.market_prob.toFixed(0)}%
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export function TopPicks() {
  const [data, setData] = useState<TopPicksResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'spreads' | 'moneylines' | 'totals'>('spreads');

  useEffect(() => {
    async function fetchData() {
      try {
        setLoading(true);
        const response = await api.getTopPicks(55, 'a');
        setData(response);
        setError(null);
      } catch (err) {
        setError('Failed to load top picks');
        console.error(err);
      } finally {
        setLoading(false);
      }
    }

    fetchData();
  }, []);

  if (loading) {
    return (
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-8 text-center">
        <div className="animate-pulse text-gray-500">Loading top picks...</div>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="bg-red-50 rounded-lg border border-red-200 p-4 text-red-700">
        {error || 'No data available'}
      </div>
    );
  }

  const hasPicks = data.spreads.length > 0 || data.moneylines.length > 0 || data.totals.length > 0;

  if (!hasPicks) {
    return (
      <div className="bg-gray-50 rounded-lg border border-gray-200 p-8 text-center text-gray-500">
        No high-value picks available right now. Check back closer to game time.
      </div>
    );
  }

  const tabs = [
    { key: 'spreads' as const, label: 'Spreads', count: data.spreads.length },
    { key: 'moneylines' as const, label: 'Moneylines', count: data.moneylines.length },
    { key: 'totals' as const, label: 'Totals', count: data.totals.length },
  ];

  const activePicks = data[activeTab];

  return (
    <div className="space-y-4">
      {/* Best Edges Summary */}
      <BestEdgesCard picks={data.best_edges} />

      {/* Tabbed Picks Tables */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
        {/* Tab Header */}
        <div className="bg-slate-800 px-4 py-2 flex items-center justify-between">
          <span className="text-white font-semibold text-sm">TOP VALUE PICKS</span>
          <div className="flex gap-1">
            {tabs.map((tab) => (
              <button
                key={tab.key}
                onClick={() => setActiveTab(tab.key)}
                className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
                  activeTab === tab.key
                    ? 'bg-white text-slate-800'
                    : 'text-slate-300 hover:text-white hover:bg-slate-700'
                }`}
              >
                {tab.label} ({tab.count})
              </button>
            ))}
          </div>
        </div>

        {/* Table */}
        {activePicks.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-gray-50 border-b border-gray-200">
                <tr>
                  <th className="text-left px-4 py-2 font-medium text-gray-600">Game</th>
                  <th className="text-left px-4 py-2 font-medium text-gray-600">Pick</th>
                  <th className="text-center px-4 py-2 font-medium text-gray-600">Value</th>
                  <th className="text-center px-4 py-2 font-medium text-gray-600">Edge</th>
                  <th className="text-center px-4 py-2 font-medium text-gray-600">Model %</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {activePicks.map((pick, i) => (
                  <tr key={i} className="hover:bg-gray-50">
                    <td className="px-4 py-2.5 font-medium text-gray-800">{pick.game}</td>
                    <td className="px-4 py-2.5">
                      <span className="font-bold text-gray-900">{pick.pick}</span>
                    </td>
                    <td className="px-4 py-2.5 text-center">
                      <ValueBadge score={pick.value_score} />
                    </td>
                    <td className="px-4 py-2.5 text-center">
                      <EdgeBadge edge={pick.edge} />
                    </td>
                    <td className="px-4 py-2.5 text-center text-gray-600">
                      {pick.model_prob.toFixed(0)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="p-8 text-center text-gray-500">
            No {activeTab} picks available
          </div>
        )}
      </div>
    </div>
  );
}
