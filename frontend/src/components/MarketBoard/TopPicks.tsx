import { useState, useEffect } from 'react';
import { api, TopPicksResponse, TopPick } from '@/lib/api';
import { ValueBadge, EdgeBadge } from '@/components/ui/ValueBadge';

function BestEdgesCard({ picks }: { picks: TopPick[] }) {
  if (picks.length === 0) return null;

  const top3 = picks.slice(0, 3);

  return (
    <div className="bg-gradient-to-br from-amber-900/20 to-yellow-900/10 rounded-xl border border-amber-800/30 p-4">
      <h3 className="font-bold text-amber-400 mb-3 flex items-center gap-2">
        <svg className="w-5 h-5 text-amber-500" fill="currentColor" viewBox="0 0 20 20">
          <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
        </svg>
        <span className="text-lg">Best Edge Plays</span>
      </h3>
      <div className="space-y-2">
        {top3.map((pick, i) => {
          const isElite = pick.value_score >= 75;
          const isStrong = pick.value_score >= 70;
          return (
            <div
              key={i}
              className={`flex items-center justify-between rounded-lg px-3 py-2 ${
                isElite
                  ? 'bg-gradient-to-r from-amber-900/30 to-yellow-900/20 ring-1 ring-amber-700/50'
                  : isStrong
                    ? 'bg-gradient-to-r from-amber-900/20 to-[#191c22] ring-1 ring-amber-800/30'
                    : 'bg-[#191c22]'
              }`}
            >
              <div className="flex items-center gap-2">
                {isStrong && (
                  <svg className="w-4 h-4 text-amber-500" fill="currentColor" viewBox="0 0 20 20">
                    <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                  </svg>
                )}
                <div>
                  <span className="font-bold text-[#f1f5f9]">{pick.pick}</span>
                  <span className="text-[#64748b] text-sm ml-2">({pick.game})</span>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <ValueBadge score={pick.value_score} size="sm" />
                <EdgeBadge edge={pick.edge} />
              </div>
            </div>
          );
        })}
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
      <div className="bg-[#191c22] rounded-xl border border-[#1e293b] p-8 text-center">
        <div className="animate-pulse text-[#64748b]">Loading top picks...</div>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="bg-red-900/20 rounded-xl border border-red-800/30 p-4 text-red-400">
        {error || 'No data available'}
      </div>
    );
  }

  const hasPicks = data.spreads.length > 0 || data.moneylines.length > 0 || data.totals.length > 0;

  if (!hasPicks) {
    return (
      <div className="bg-[#191c22] rounded-xl border border-[#1e293b] p-8 text-center text-[#64748b]">
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
      <div className="bg-[#191c22] rounded-xl border border-[#1e293b] overflow-hidden">
        {/* Tab Header */}
        <div className="bg-[#0b0e14] px-4 py-2 flex items-center justify-between">
          <span className="text-[#a4e6ff] font-semibold text-sm">TOP VALUE PICKS</span>
          <div className="flex gap-1">
            {tabs.map((tab) => (
              <button
                key={tab.key}
                onClick={() => setActiveTab(tab.key)}
                className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
                  activeTab === tab.key
                    ? 'bg-[#a4e6ff] text-[#0b0e14]'
                    : 'text-[#64748b] hover:text-[#f1f5f9] hover:bg-[#191c22]'
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
              <thead className="bg-[#0b0e14] border-b border-[#1e293b]">
                <tr>
                  <th className="text-left px-4 py-2 font-medium text-[#64748b]">Game</th>
                  <th className="text-left px-4 py-2 font-medium text-[#64748b]">Pick</th>
                  <th className="text-center px-4 py-2 font-medium text-[#64748b]">Value</th>
                  <th className="text-center px-4 py-2 font-medium text-[#64748b]">Edge</th>
                  <th className="text-center px-4 py-2 font-medium text-[#64748b]">Model %</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#1e293b]">
                {activePicks.map((pick, i) => {
                  const isElite = pick.value_score >= 75;
                  const isStrong = pick.value_score >= 70;
                  return (
                    <tr
                      key={i}
                      className={`${
                        isElite
                          ? 'bg-gradient-to-r from-amber-900/20 to-yellow-900/10 hover:from-amber-900/30 hover:to-yellow-900/20'
                          : isStrong
                            ? 'bg-amber-900/10 hover:bg-amber-900/20'
                            : 'hover:bg-[#1d2026]'
                      }`}
                    >
                      <td className="px-4 py-2.5 font-medium text-[#f1f5f9]">
                        <div className="flex items-center gap-2">
                          {isStrong && (
                            <svg className="w-4 h-4 text-amber-500 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                              <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                            </svg>
                          )}
                          {pick.game}
                        </div>
                      </td>
                      <td className="px-4 py-2.5">
                        <span className={`font-bold ${isStrong ? 'text-amber-400' : 'text-[#f1f5f9]'}`}>
                          {pick.pick}
                        </span>
                        {isElite && (
                          <span className="ml-2 text-xs font-bold text-amber-600 uppercase">Elite</span>
                        )}
                      </td>
                      <td className="px-4 py-2.5 text-center">
                        <ValueBadge score={pick.value_score} size="sm" />
                      </td>
                      <td className="px-4 py-2.5 text-center">
                        <EdgeBadge edge={pick.edge} />
                      </td>
                      <td className="px-4 py-2.5 text-center text-[#94a3b8]">
                        {pick.model_prob.toFixed(0)}%
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="p-8 text-center text-[#64748b]">
            No {activeTab} picks available
          </div>
        )}
      </div>
    </div>
  );
}
