import { useState } from 'react';
import { useEvaluationSummary, usePerformanceByBucket, useDailyResults, usePredictionPerformance } from '@/hooks/useEvaluation';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { CumulativePLChart } from '@/components/Charts';

export function Evaluation() {
  const [days, setDays] = useState(14);
  const [minValue, setMinValue] = useState(65);
  const [predDays, setPredDays] = useState(14);

  const { data: summary, isLoading: summaryLoading } = useEvaluationSummary(days, minValue);
  const { data: performance, isLoading: perfLoading } = usePerformanceByBucket(days);
  const { data: dailyResults, isLoading: dailyLoading } = useDailyResults(days, minValue);
  const { data: predictions, isLoading: predictionsLoading } = usePredictionPerformance(predDays);

  const selectClass =
    'text-sm bg-[#0b0e14] border border-[#1e293b] text-[#f1f5f9] rounded px-2 py-1 font-mono focus:outline-none focus:border-[#a4e6ff]';

  return (
    <div className="max-w-6xl mx-auto px-4 py-6 space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-[#f1f5f9] font-display tracking-tight">
          MODEL <span className="text-[#a4e6ff]">PERFORMANCE</span>
        </h1>
        <p className="text-sm text-[#64748b] mt-1 font-mono">
          Track prediction accuracy and betting performance
        </p>
      </div>

      {/* Summary Stats */}
      <div className="bg-[#191c22] rounded-xl border border-[#1e293b] p-5">
        <div className="flex justify-between items-center mb-4 flex-wrap gap-3">
          <h2 className="text-[10px] text-[#64748b] uppercase font-bold tracking-widest">Performance Summary</h2>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <label className="text-[10px] text-[#64748b] uppercase font-bold tracking-widest">Days</label>
              <select
                value={days}
                onChange={(e) => setDays(Number(e.target.value))}
                className={selectClass}
              >
                <option value={7}>7</option>
                <option value={14}>14</option>
                <option value={30}>30</option>
              </select>
            </div>
            <div className="flex items-center gap-2">
              <label className="text-[10px] text-[#64748b] uppercase font-bold tracking-widest">Min Value</label>
              <select
                value={minValue}
                onChange={(e) => setMinValue(Number(e.target.value))}
                className={selectClass}
              >
                <option value={0}>All</option>
                <option value={50}>50+</option>
                <option value={60}>60+</option>
                <option value={65}>65+</option>
                <option value={70}>70+</option>
              </select>
            </div>
          </div>
        </div>

        {summaryLoading && (
          <div className="flex justify-center py-8">
            <LoadingSpinner />
          </div>
        )}

        {!summaryLoading && summary && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-[#0b0e14] rounded-lg p-4 border border-[#1e293b]">
              <div className="text-[10px] text-[#64748b] uppercase font-bold tracking-widest mb-2">Total Bets</div>
              <div className="text-2xl font-black font-mono text-[#a4e6ff]">{summary.total_bets}</div>
            </div>

            <div className="bg-[#0b0e14] rounded-lg p-4 border border-[#1e293b]">
              <div className="text-[10px] text-[#64748b] uppercase font-bold tracking-widest mb-2">Record</div>
              <div className="text-2xl font-black font-mono text-[#f1f5f9]">
                {summary.wins}-{summary.losses}
                {summary.pushes > 0 && `-${summary.pushes}`}
              </div>
              <div className="text-[11px] text-[#64748b] font-mono mt-1">
                {summary.win_rate ? `${summary.win_rate}%` : '-'}
              </div>
            </div>

            <div className="bg-[#0b0e14] rounded-lg p-4 border border-[#1e293b]">
              <div className="text-[10px] text-[#64748b] uppercase font-bold tracking-widest mb-2">Profit</div>
              <div className={`text-2xl font-black font-mono ${summary.profit >= 0 ? 'text-[#66f796]' : 'text-[#ef4444]'}`}>
                {summary.profit >= 0 ? '+' : ''}${summary.profit.toFixed(0)}
              </div>
            </div>

            <div className="bg-[#0b0e14] rounded-lg p-4 border border-[#1e293b]">
              <div className="text-[10px] text-[#64748b] uppercase font-bold tracking-widest mb-2">ROI</div>
              <div className={`text-2xl font-black font-mono ${(summary.roi ?? 0) >= 0 ? 'text-[#66f796]' : 'text-[#ef4444]'}`}>
                {summary.roi ? `${summary.roi >= 0 ? '+' : ''}${summary.roi}%` : '-'}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Cumulative P/L Chart */}
      {!dailyLoading && dailyResults && dailyResults.length > 0 && (
        <CumulativePLChart dailyResults={dailyResults} />
      )}

      {/* Performance by Value Bucket */}
      <div className="bg-[#191c22] rounded-xl border border-[#1e293b] p-5">
        <h2 className="text-[10px] text-[#64748b] uppercase font-bold tracking-widest mb-4">Performance by Value Score</h2>

        {perfLoading && (
          <div className="flex justify-center py-8">
            <LoadingSpinner />
          </div>
        )}

        {!perfLoading && performance && performance.length > 0 && (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-[10px] text-[#64748b] uppercase font-bold tracking-widest border-b border-[#1e293b]">
                  <th className="pb-2">Bucket</th>
                  <th className="pb-2 text-center">Bets</th>
                  <th className="pb-2 text-center">Record</th>
                  <th className="pb-2 text-center">Win Rate</th>
                  <th className="pb-2 text-right">Profit</th>
                  <th className="pb-2 text-right">ROI</th>
                </tr>
              </thead>
              <tbody>
                {performance.map((bucket) => (
                  <tr key={bucket.bucket} className="border-b border-[#1e293b]">
                    <td className="py-2 font-medium text-[#f1f5f9]">{bucket.bucket}</td>
                    <td className="py-2 text-center font-mono text-[#f1f5f9]">{bucket.bet_count}</td>
                    <td className="py-2 text-center font-mono text-[#94a3b8]">{bucket.wins}-{bucket.losses}</td>
                    <td className="py-2 text-center font-mono text-[#94a3b8]">
                      {bucket.win_rate ? `${bucket.win_rate}%` : '-'}
                    </td>
                    <td className={`py-2 text-right font-mono font-bold ${bucket.profit >= 0 ? 'text-[#66f796]' : 'text-[#ef4444]'}`}>
                      {bucket.profit >= 0 ? '+' : ''}${bucket.profit.toFixed(0)}
                    </td>
                    <td className={`py-2 text-right font-mono ${bucket.roi && bucket.roi >= 0 ? 'text-[#66f796]' : 'text-[#ef4444]'}`}>
                      {bucket.roi ? `${bucket.roi >= 0 ? '+' : ''}${bucket.roi}%` : '-'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {!perfLoading && (!performance || performance.length === 0) && (
          <div className="text-center py-8 text-[#64748b] font-mono text-sm">
            No performance data available
          </div>
        )}
      </div>

      {/* Daily Results */}
      <div className="bg-[#191c22] rounded-xl border border-[#1e293b] p-5">
        <h2 className="text-[10px] text-[#64748b] uppercase font-bold tracking-widest mb-4">Daily Results</h2>

        {dailyLoading && (
          <div className="flex justify-center py-8">
            <LoadingSpinner />
          </div>
        )}

        {!dailyLoading && dailyResults && dailyResults.length > 0 && (
          <div className="space-y-4">
            {dailyResults.map((day) => (
              <div key={day.date} className="border border-[#1e293b] rounded-lg overflow-hidden">
                <div className="bg-[#0b0e14] px-4 py-3">
                  <div className="flex justify-between items-center flex-wrap gap-2">
                    <span className="font-bold text-[#f1f5f9] text-sm">
                      {new Date(day.date + 'T12:00:00').toLocaleDateString('en-US', {
                        weekday: 'short',
                        month: 'short',
                        day: 'numeric',
                      })}
                    </span>
                    <div className="flex items-center gap-4 text-sm">
                      <span className="font-mono text-[#94a3b8]">{day.record}</span>
                      <span
                        className={`font-bold font-mono ${day.profit >= 0 ? 'text-[#66f796]' : 'text-[#ef4444]'}`}
                      >
                        {day.profit >= 0 ? '+' : ''}${day.profit.toFixed(0)}
                      </span>
                      <span className={`font-mono ${day.roi >= 0 ? 'text-[#66f796]' : 'text-[#ef4444]'}`}>
                        ({day.roi >= 0 ? '+' : ''}{day.roi}%)
                      </span>
                    </div>
                  </div>

                  {/* By Type Breakdown */}
                  {day.by_type && (
                    <div className="flex gap-2 mt-2 text-[10px] font-mono flex-wrap">
                      {day.by_type.spread.wins + day.by_type.spread.losses > 0 && (
                        <span className="bg-[#a4e6ff]/10 text-[#a4e6ff] px-2 py-0.5 rounded uppercase tracking-wider">
                          Spread: {day.by_type.spread.record} ({day.by_type.spread.profit >= 0 ? '+' : ''}${day.by_type.spread.profit.toFixed(0)})
                        </span>
                      )}
                      {day.by_type.total.wins + day.by_type.total.losses > 0 && (
                        <span className="bg-[#a4e6ff]/10 text-[#a4e6ff] px-2 py-0.5 rounded uppercase tracking-wider">
                          Total: {day.by_type.total.record} ({day.by_type.total.profit >= 0 ? '+' : ''}${day.by_type.total.profit.toFixed(0)})
                        </span>
                      )}
                      {day.by_type.moneyline.wins + day.by_type.moneyline.losses > 0 && (
                        <span className="bg-[#f59e0b]/10 text-[#f59e0b] px-2 py-0.5 rounded uppercase tracking-wider">
                          ML: {day.by_type.moneyline.record} ({day.by_type.moneyline.profit >= 0 ? '+' : ''}${day.by_type.moneyline.profit.toFixed(0)})
                        </span>
                      )}
                    </div>
                  )}
                </div>
                <div className="divide-y divide-[#1e293b]">
                  {day.bets.map((bet, i) => (
                    <div
                      key={i}
                      className="px-4 py-2 flex justify-between items-center text-sm bg-[#191c22]"
                    >
                      <div className="flex items-center gap-3">
                        <span
                          className="w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-black font-mono text-white"
                          style={{
                            backgroundColor:
                              bet.result === 'win'
                                ? '#10b981'
                                : bet.result === 'loss'
                                ? '#ef4444'
                                : '#64748b',
                          }}
                        >
                          {bet.result === 'win' ? 'W' : bet.result === 'loss' ? 'L' : 'P'}
                        </span>
                        <div>
                          <div className="font-medium text-[#f1f5f9]">{bet.bet}</div>
                          <div className="text-[11px] text-[#64748b] font-mono">
                            {bet.matchup} - {bet.final_score}
                          </div>
                        </div>
                      </div>
                      <div className="flex items-center gap-4">
                        <span className="text-[10px] bg-[#a4e6ff]/10 text-[#a4e6ff] px-2 py-0.5 rounded font-mono font-bold">
                          {bet.value_score}
                        </span>
                        <span
                          className={`font-bold font-mono ${
                            bet.profit >= 0 ? 'text-[#66f796]' : 'text-[#ef4444]'
                          }`}
                        >
                          {bet.profit >= 0 ? '+' : ''}${bet.profit.toFixed(0)}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}

        {!dailyLoading && (!dailyResults || dailyResults.length === 0) && (
          <div className="text-center py-8 text-[#64748b] font-mono text-sm">
            No betting results available for the selected criteria
          </div>
        )}
      </div>

      {/* Recent Predictions */}
      <div className="bg-[#191c22] rounded-xl border border-[#1e293b] p-5">
        <div className="flex justify-between items-center mb-4 flex-wrap gap-3">
          <h2 className="text-[10px] text-[#64748b] uppercase font-bold tracking-widest">Recent Predictions</h2>
          <div className="flex items-center gap-2">
            <label className="text-[10px] text-[#64748b] uppercase font-bold tracking-widest">Days</label>
            <select
              value={predDays}
              onChange={(e) => setPredDays(Number(e.target.value))}
              className={selectClass}
            >
              <option value={7}>7</option>
              <option value={14}>14</option>
              <option value={30}>30</option>
            </select>
          </div>
        </div>

        {predictionsLoading && (
          <div className="flex justify-center py-8">
            <LoadingSpinner />
          </div>
        )}

        {!predictionsLoading && predictions && predictions.recent_predictions.length > 0 && (
          <div className="space-y-2 max-h-96 overflow-y-auto">
            {predictions.recent_predictions.slice(0, 20).map((pred, i) => (
              <div
                key={i}
                className="p-3 rounded-lg border border-[#1e293b] bg-[#0b0e14]"
              >
                <div className="flex justify-between items-start gap-3">
                  <div className="min-w-0">
                    <div className="font-bold text-[#f1f5f9] text-sm">{pred.matchup}</div>
                    <div className="text-[11px] text-[#94a3b8] font-mono mt-0.5">
                      Picked: <span className="font-bold text-[#f1f5f9]">{pred.predicted_winner}</span> ({pred.winner_prob}%)
                      {pred.best_bet && (
                        <span className="ml-2">
                          | {pred.best_bet.team} {pred.best_bet.type}
                          {pred.best_bet.line ? ` ${pred.best_bet.line > 0 ? '+' : ''}${pred.best_bet.line}` : ''}
                          {pred.best_bet.value_score && <span className="text-[#a4e6ff]"> ({pred.best_bet.value_score})</span>}
                        </span>
                      )}
                    </div>
                  </div>
                  <div className="text-right flex-shrink-0">
                    <div className={`font-black font-mono text-sm ${pred.winner_correct ? 'text-[#66f796]' : 'text-[#ef4444]'}`}>
                      {pred.winner_correct ? 'WIN' : 'LOSS'}
                    </div>
                    <div className="text-[11px] text-[#64748b] font-mono">{pred.final_score}</div>
                    {pred.bet_profit !== null && (
                      <div className={`text-[11px] font-bold font-mono ${pred.bet_profit >= 0 ? 'text-[#66f796]' : 'text-[#ef4444]'}`}>
                        {pred.bet_profit >= 0 ? '+' : ''}${pred.bet_profit.toFixed(0)}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {!predictionsLoading && (!predictions || predictions.recent_predictions.length === 0) && (
          <div className="text-center py-8 text-[#64748b] font-mono text-sm">
            No predictions available
          </div>
        )}
      </div>
    </div>
  );
}
