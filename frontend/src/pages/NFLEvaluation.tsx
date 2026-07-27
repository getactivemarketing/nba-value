import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { nflApi, type NFLMarketRecord } from '@/lib/nflApi';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';

function StatCard({ label, value, color, small }: {
  label: string;
  value: string;
  color?: 'green' | 'red' | 'cyan';
  small?: boolean;
}) {
  const colorClass = color === 'green' ? 'text-[#66f796]'
    : color === 'red' ? 'text-[#ef4444]'
    : color === 'cyan' ? 'text-[#a4e6ff]'
    : 'text-[#f1f5f9]';

  return (
    <div className="bg-[#0b0e14] rounded-lg p-4 border border-[#1e293b]">
      <div className="text-[10px] text-[#64748b] uppercase font-bold tracking-widest mb-2">{label}</div>
      <div className={`${small ? 'text-xl' : 'text-2xl'} font-black font-mono ${colorClass}`}>{value}</div>
    </div>
  );
}

function ShadowMarketTile({ label, record }: { label: string; record?: NFLMarketRecord }) {
  const d = (record?.wins ?? 0) + (record?.losses ?? 0);
  const wr = record?.win_rate != null ? `${(record.win_rate * 100).toFixed(0)}%` : d > 0 ? `${((record!.wins / d) * 100).toFixed(0)}%` : '-';
  const profit = record?.profit ?? 0;

  return (
    <div className="bg-[#0b0e14]/60 rounded-lg p-3 border border-[#1e293b]/60 opacity-70">
      <div className="flex items-center justify-between mb-2">
        <div className="text-[10px] text-[#64748b] uppercase font-bold tracking-widest">{label}</div>
        <span className="text-[9px] text-[#64748b] uppercase font-bold tracking-widest bg-[#191c22] border border-[#1e293b] rounded px-1.5 py-0.5">
          Shadow
        </span>
      </div>
      <div className="text-base font-black font-mono text-[#94a3b8]">
        {record ? `${record.wins}-${record.losses}${record.pushes > 0 ? `-${record.pushes}` : ''}` : '0-0'}
      </div>
      <div className="text-xs font-mono text-[#64748b] mt-1">
        {wr}{record && ` · ${profit >= 0 ? '+' : ''}${profit.toFixed(0)}u`}
      </div>
      <div className="text-[10px] text-[#64748b] font-mono mt-1.5 italic">
        tracked, not bet
      </div>
    </div>
  );
}

function EmptyStatePanel() {
  return (
    <div className="rounded-xl bg-[#191c22] border border-[#1e293b] p-10 text-center">
      <p className="text-3xl mb-3">🏈</p>
      <h3 className="text-lg font-black font-mono text-[#f1f5f9] tracking-tight mb-2">
        No graded NFL games yet
      </h3>
      <p className="text-sm text-[#64748b] font-mono max-w-md mx-auto leading-relaxed">
        Results appear once the season starts. The model is totals-forward for launch — spread
        &amp; moneyline picks stay shadow-tracked until they show an edge over the market.
      </p>
    </div>
  );
}

export function NFLEvaluation() {
  const [dailyDays, setDailyDays] = useState(14);

  const { data: summary, isLoading: summaryLoading } = useQuery({
    queryKey: ['nfl-eval-summary'],
    queryFn: () => nflApi.getEvaluationSummary(),
  });

  const { data: daily, isLoading: dailyLoading } = useQuery({
    queryKey: ['nfl-eval-daily', dailyDays],
    queryFn: () => nflApi.getDailyEvaluation(dailyDays),
  });

  const selectClass =
    'text-sm bg-[#0b0e14] border border-[#1e293b] text-[#f1f5f9] rounded px-2 py-1 font-mono focus:outline-none focus:border-[#a4e6ff]';

  const bestBet = summary?.by_market?.best_bet;
  const decided = (bestBet?.wins ?? 0) + (bestBet?.losses ?? 0);
  const winRate = bestBet?.win_rate != null
    ? `${(bestBet.win_rate * 100).toFixed(0)}%`
    : decided > 0
    ? `${((bestBet!.wins / decided) * 100).toFixed(0)}%`
    : '-';
  const roi = decided > 0 && bestBet ? ((bestBet.profit / (decided * 100)) * 100).toFixed(1) : null;

  const isEmpty = !summaryLoading && (!summary || summary.graded === 0);

  return (
    <div className="max-w-6xl mx-auto px-4 py-6 space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-[#f1f5f9] font-display tracking-tight">
          NFL <span className="text-[#a4e6ff]">PERFORMANCE</span>
        </h1>
        <p className="text-sm text-[#64748b] mt-1 font-mono">
          Totals-forward best-bet record — spread &amp; moneyline shadow-tracked, not bet
        </p>
      </div>

      {summaryLoading && (
        <div className="flex justify-center py-8"><LoadingSpinner /></div>
      )}

      {isEmpty && <EmptyStatePanel />}

      {!summaryLoading && summary && summary.graded > 0 && (
        <>
          {/* Headline: best_bet (totals) record */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <StatCard
              label="Best Bet Record"
              value={`${bestBet?.wins ?? 0}-${bestBet?.losses ?? 0}${(bestBet?.pushes ?? 0) > 0 ? `-${bestBet?.pushes}` : ''}`}
            />
            <StatCard label="Win %" value={winRate} color="cyan" />
            <StatCard
              label="P/L (units)"
              value={`${(bestBet?.profit ?? 0) >= 0 ? '+' : ''}${(bestBet?.profit ?? 0).toFixed(0)}u`}
              color={(bestBet?.profit ?? 0) >= 0 ? 'green' : 'red'}
            />
            <StatCard
              label="ROI"
              value={roi ? `${Number(roi) >= 0 ? '+' : ''}${roi}%` : '-'}
              color={roi && Number(roi) >= 0 ? 'green' : 'red'}
            />
          </div>

          {/* Shadow markets: spread + ml */}
          <div className="bg-[#191c22] rounded-xl border border-[#1e293b] p-5">
            <h2 className="text-[10px] text-[#64748b] uppercase font-bold tracking-widest mb-1">
              Spread &amp; Moneyline
            </h2>
            <p className="text-xs text-[#64748b] font-mono mb-4">
              Shadow markets — tracked for accuracy, not part of the live best-bet feed.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <ShadowMarketTile label="Spread" record={summary.by_market?.spread} />
              <ShadowMarketTile label="Moneyline" record={summary.by_market?.ml} />
            </div>
          </div>

          {/* Daily Results */}
          <div className="bg-[#191c22] rounded-xl border border-[#1e293b] p-5">
            <div className="flex justify-between items-center mb-4 flex-wrap gap-3">
              <h2 className="text-[10px] text-[#64748b] uppercase font-bold tracking-widest">Daily Results</h2>
              <select value={dailyDays} onChange={(e) => setDailyDays(Number(e.target.value))} className={selectClass}>
                <option value={7}>7 days</option>
                <option value={14}>14 days</option>
                <option value={30}>30 days</option>
              </select>
            </div>

            {dailyLoading && (
              <div className="flex justify-center py-8"><LoadingSpinner /></div>
            )}

            {!dailyLoading && daily && daily.length > 0 && (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left text-[10px] text-[#64748b] uppercase font-bold tracking-widest border-b border-[#1e293b]">
                      <th className="pb-2">Date</th>
                      <th className="pb-2 text-center">Picks</th>
                      <th className="pb-2 text-center">Record</th>
                      <th className="pb-2 text-center">Win %</th>
                      <th className="pb-2 text-right">P/L</th>
                    </tr>
                  </thead>
                  <tbody>
                    {daily.map((day) => {
                      const record = `${day.wins}-${day.losses}${day.pushes > 0 ? `-${day.pushes}` : ''}`;
                      const wr = day.win_rate ? `${(day.win_rate * 100).toFixed(0)}%` : '-';
                      return (
                        <tr key={day.date} className="border-b border-[#1e293b]">
                          <td className="py-2 text-[#94a3b8] font-mono">{day.date}</td>
                          <td className="py-2 text-center font-mono text-[#f1f5f9]">{day.predictions}</td>
                          <td className="py-2 text-center font-mono text-[#f1f5f9]">{record}</td>
                          <td className="py-2 text-center font-mono text-[#94a3b8]">{wr}</td>
                          <td className={`py-2 text-right font-mono font-bold ${day.profit >= 0 ? 'text-[#66f796]' : 'text-[#ef4444]'}`}>
                            {day.profit >= 0 ? '+' : ''}{day.profit.toFixed(0)}u
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}

            {!dailyLoading && (!daily || daily.length === 0) && (
              <div className="text-center py-8 text-[#64748b] font-mono text-sm">
                No daily results available yet
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}
