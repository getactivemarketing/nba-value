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

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Model Performance</h1>
        <p className="text-sm text-gray-500 mt-1">
          Track prediction accuracy and betting performance
        </p>
      </div>

      {/* Summary Stats */}
      <div className="card">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-lg font-semibold text-gray-900">Performance Summary</h2>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <label className="text-sm text-gray-600">Days:</label>
              <select
                value={days}
                onChange={(e) => setDays(Number(e.target.value))}
                className="text-sm border rounded px-2 py-1"
              >
                <option value={7}>7</option>
                <option value={14}>14</option>
                <option value={30}>30</option>
              </select>
            </div>
            <div className="flex items-center gap-2">
              <label className="text-sm text-gray-600">Min Value:</label>
              <select
                value={minValue}
                onChange={(e) => setMinValue(Number(e.target.value))}
                className="text-sm border rounded px-2 py-1"
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
            <div className="bg-blue-50 rounded-lg p-4 text-center">
              <div className="text-sm text-blue-600 font-medium">Total Bets</div>
              <div className="text-2xl font-bold text-blue-700">{summary.total_bets}</div>
            </div>

            <div className="bg-emerald-50 rounded-lg p-4 text-center">
              <div className="text-sm text-emerald-600 font-medium">Record</div>
              <div className="text-2xl font-bold text-emerald-700">
                {summary.wins}-{summary.losses}
                {summary.pushes > 0 && `-${summary.pushes}`}
              </div>
              <div className="text-sm text-emerald-600">
                {summary.win_rate ? `${summary.win_rate}%` : '-'}
              </div>
            </div>

            <div className={`rounded-lg p-4 text-center ${summary.profit >= 0 ? 'bg-green-50' : 'bg-red-50'}`}>
              <div className={`text-sm font-medium ${summary.profit >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                Profit
              </div>
              <div className={`text-2xl font-bold ${summary.profit >= 0 ? 'text-green-700' : 'text-red-700'}`}>
                {summary.profit >= 0 ? '+' : ''}${summary.profit.toFixed(0)}
              </div>
            </div>

            <div className={`rounded-lg p-4 text-center ${(summary.roi ?? 0) >= 0 ? 'bg-green-50' : 'bg-red-50'}`}>
              <div className={`text-sm font-medium ${(summary.roi ?? 0) >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                ROI
              </div>
              <div className={`text-2xl font-bold ${(summary.roi ?? 0) >= 0 ? 'text-green-700' : 'text-red-700'}`}>
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
      <div className="card">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">Performance by Value Score</h2>

        {perfLoading && (
          <div className="flex justify-center py-8">
            <LoadingSpinner />
          </div>
        )}

        {!perfLoading && performance && performance.length > 0 && (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-gray-500 border-b">
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
                  <tr key={bucket.bucket} className="border-b border-gray-100">
                    <td className="py-2 font-medium">{bucket.bucket}</td>
                    <td className="py-2 text-center">{bucket.bet_count}</td>
                    <td className="py-2 text-center">{bucket.wins}-{bucket.losses}</td>
                    <td className="py-2 text-center">
                      {bucket.win_rate ? `${bucket.win_rate}%` : '-'}
                    </td>
                    <td className={`py-2 text-right font-medium ${bucket.profit >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                      {bucket.profit >= 0 ? '+' : ''}${bucket.profit.toFixed(0)}
                    </td>
                    <td className={`py-2 text-right ${bucket.roi && bucket.roi >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                      {bucket.roi ? `${bucket.roi >= 0 ? '+' : ''}${bucket.roi}%` : '-'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {!perfLoading && (!performance || performance.length === 0) && (
          <div className="text-center py-8 text-gray-500">
            No performance data available
          </div>
        )}
      </div>

      {/* Daily Results */}
      <div className="card">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">Daily Results</h2>

        {dailyLoading && (
          <div className="flex justify-center py-8">
            <LoadingSpinner />
          </div>
        )}

        {!dailyLoading && dailyResults && dailyResults.length > 0 && (
          <div className="space-y-4">
            {dailyResults.map((day) => (
              <div key={day.date} className="border rounded-lg overflow-hidden">
                <div className="bg-gray-50 px-4 py-3">
                  <div className="flex justify-between items-center">
                    <span className="font-medium">
                      {new Date(day.date + 'T12:00:00').toLocaleDateString('en-US', {
                        weekday: 'short',
                        month: 'short',
                        day: 'numeric',
                      })}
                    </span>
                    <div className="flex items-center gap-4 text-sm">
                      <span className="font-medium">{day.record}</span>
                      <span
                        className={`font-bold ${day.profit >= 0 ? 'text-green-600' : 'text-red-600'}`}
                      >
                        {day.profit >= 0 ? '+' : ''}${day.profit.toFixed(0)}
                      </span>
                      <span className={day.roi >= 0 ? 'text-green-600' : 'text-red-600'}>
                        ({day.roi >= 0 ? '+' : ''}{day.roi}%)
                      </span>
                    </div>
                  </div>

                  {/* By Type Breakdown */}
                  {day.by_type && (
                    <div className="flex gap-4 mt-2 text-xs">
                      {day.by_type.spread.wins + day.by_type.spread.losses > 0 && (
                        <span className="bg-blue-100 text-blue-700 px-2 py-0.5 rounded">
                          Spread: {day.by_type.spread.record} ({day.by_type.spread.profit >= 0 ? '+' : ''}${day.by_type.spread.profit.toFixed(0)})
                        </span>
                      )}
                      {day.by_type.total.wins + day.by_type.total.losses > 0 && (
                        <span className="bg-purple-100 text-purple-700 px-2 py-0.5 rounded">
                          Total: {day.by_type.total.record} ({day.by_type.total.profit >= 0 ? '+' : ''}${day.by_type.total.profit.toFixed(0)})
                        </span>
                      )}
                      {day.by_type.moneyline.wins + day.by_type.moneyline.losses > 0 && (
                        <span className="bg-amber-100 text-amber-700 px-2 py-0.5 rounded">
                          ML: {day.by_type.moneyline.record} ({day.by_type.moneyline.profit >= 0 ? '+' : ''}${day.by_type.moneyline.profit.toFixed(0)})
                        </span>
                      )}
                    </div>
                  )}
                </div>
                <div className="divide-y divide-gray-100">
                  {day.bets.map((bet, i) => (
                    <div
                      key={i}
                      className={`px-4 py-2 flex justify-between items-center text-sm ${
                        bet.result === 'win'
                          ? 'bg-green-50'
                          : bet.result === 'loss'
                          ? 'bg-red-50'
                          : 'bg-gray-50'
                      }`}
                    >
                      <div className="flex items-center gap-3">
                        <span
                          className={`w-5 h-5 rounded-full flex items-center justify-center text-xs font-bold ${
                            bet.result === 'win'
                              ? 'bg-green-500 text-white'
                              : bet.result === 'loss'
                              ? 'bg-red-500 text-white'
                              : 'bg-gray-400 text-white'
                          }`}
                        >
                          {bet.result === 'win' ? 'W' : bet.result === 'loss' ? 'L' : 'P'}
                        </span>
                        <div>
                          <div className="font-medium">{bet.bet}</div>
                          <div className="text-xs text-gray-500">
                            {bet.matchup} - {bet.final_score}
                          </div>
                        </div>
                      </div>
                      <div className="flex items-center gap-4">
                        <span className="text-xs bg-blue-100 text-blue-700 px-2 py-0.5 rounded">
                          {bet.value_score}
                        </span>
                        <span
                          className={`font-medium ${
                            bet.profit >= 0 ? 'text-green-600' : 'text-red-600'
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
          <div className="text-center py-8 text-gray-500">
            No betting results available for the selected criteria
          </div>
        )}
      </div>

      {/* Recent Predictions */}
      <div className="card">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-lg font-semibold text-gray-900">Recent Predictions</h2>
          <div className="flex items-center gap-2">
            <label className="text-sm text-gray-600">Days:</label>
            <select
              value={predDays}
              onChange={(e) => setPredDays(Number(e.target.value))}
              className="text-sm border rounded px-2 py-1"
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
                className={`p-3 rounded-lg border ${
                  pred.winner_correct
                    ? 'bg-green-50 border-green-200'
                    : 'bg-red-50 border-red-200'
                }`}
              >
                <div className="flex justify-between items-start">
                  <div>
                    <div className="font-medium">{pred.matchup}</div>
                    <div className="text-sm text-gray-600">
                      Picked: <span className="font-medium">{pred.predicted_winner}</span> ({pred.winner_prob}%)
                      {pred.best_bet && (
                        <span className="ml-2">
                          | {pred.best_bet.team} {pred.best_bet.type}
                          {pred.best_bet.line ? ` ${pred.best_bet.line > 0 ? '+' : ''}${pred.best_bet.line}` : ''}
                          {pred.best_bet.value_score && <span className="text-blue-600"> ({pred.best_bet.value_score})</span>}
                        </span>
                      )}
                    </div>
                  </div>
                  <div className="text-right">
                    <div className={`font-bold ${pred.winner_correct ? 'text-green-600' : 'text-red-600'}`}>
                      {pred.winner_correct ? 'WIN' : 'LOSS'}
                    </div>
                    <div className="text-sm text-gray-600">{pred.final_score}</div>
                    {pred.bet_profit !== null && (
                      <div className={`text-sm font-medium ${pred.bet_profit >= 0 ? 'text-green-600' : 'text-red-600'}`}>
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
          <div className="text-center py-8 text-gray-500">
            No predictions available
          </div>
        )}
      </div>
    </div>
  );
}
