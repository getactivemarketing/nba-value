import { useState } from 'react';
import { useAlgorithmComparison, usePerformanceByBucket, useDailyResults, usePredictionPerformance } from '@/hooks/useEvaluation';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { PerformanceBucketsChart, EmptyChart } from '@/components/Charts';
import type { Algorithm } from '@/types/market';

type Metric = 'win_rate' | 'roi' | 'clv_avg';

export function Evaluation() {
  const [metric, setMetric] = useState<Metric>('win_rate');
  const [dailyAlgo, setDailyAlgo] = useState<Algorithm>('b');
  const [minValue, setMinValue] = useState(50);
  const [predDays, setPredDays] = useState(14);

  const { data: comparison, isLoading: comparisonLoading } = useAlgorithmComparison();
  const { data: performanceA, isLoading: perfALoading } = usePerformanceByBucket('a', 'score');
  const { data: performanceB, isLoading: perfBLoading } = usePerformanceByBucket('b', 'score');
  const { data: dailyResults, isLoading: dailyLoading } = useDailyResults(14, dailyAlgo, minValue);
  const { data: predictions, isLoading: predictionsLoading } = usePredictionPerformance(predDays);

  const isLoadingPerformance = perfALoading || perfBLoading;
  const hasPerformanceData =
    performanceA && performanceA.length > 0 && performanceB && performanceB.length > 0;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Model Performance</h1>
        <p className="text-sm text-gray-500 mt-1">
          Track prediction accuracy and betting performance
        </p>
      </div>

      {/* Prediction Performance - Real-time tracking from snapshots */}
      <div className="card">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-lg font-semibold text-gray-900">
            Prediction Tracking
          </h2>
          <div className="flex items-center gap-2">
            <label className="text-sm text-gray-600">Days:</label>
            <select
              value={predDays}
              onChange={(e) => setPredDays(Number(e.target.value))}
              className="text-sm border rounded px-2 py-1"
            >
              <option value={7}>7 days</option>
              <option value={14}>14 days</option>
              <option value={30}>30 days</option>
            </select>
          </div>
        </div>

        {predictionsLoading && (
          <div className="flex justify-center py-8">
            <LoadingSpinner />
          </div>
        )}

        {!predictionsLoading && predictions && (
          <div className="space-y-4">
            {/* Summary Stats */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="bg-blue-50 rounded-lg p-4 text-center">
                <div className="text-sm text-blue-600 font-medium">Winner Picks</div>
                <div className="text-2xl font-bold text-blue-700">
                  {predictions.summary.winner_accuracy.wins}-{predictions.summary.winner_accuracy.losses}
                </div>
                <div className="text-sm text-blue-600">
                  {predictions.summary.winner_accuracy.rate ? `${predictions.summary.winner_accuracy.rate}%` : '-'}
                </div>
              </div>

              <div className="bg-emerald-50 rounded-lg p-4 text-center">
                <div className="text-sm text-emerald-600 font-medium">Best Bets</div>
                <div className="text-2xl font-bold text-emerald-700">
                  {predictions.summary.best_bet_performance.wins}-{predictions.summary.best_bet_performance.losses}
                  {predictions.summary.best_bet_performance.pushes > 0 && `-${predictions.summary.best_bet_performance.pushes}`}
                </div>
                <div className="text-sm text-emerald-600">
                  {predictions.summary.best_bet_performance.win_rate ? `${predictions.summary.best_bet_performance.win_rate}%` : '-'}
                </div>
              </div>

              <div className={`rounded-lg p-4 text-center ${predictions.summary.best_bet_performance.profit >= 0 ? 'bg-green-50' : 'bg-red-50'}`}>
                <div className={`text-sm font-medium ${predictions.summary.best_bet_performance.profit >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                  Profit
                </div>
                <div className={`text-2xl font-bold ${predictions.summary.best_bet_performance.profit >= 0 ? 'text-green-700' : 'text-red-700'}`}>
                  {predictions.summary.best_bet_performance.profit >= 0 ? '+' : ''}${predictions.summary.best_bet_performance.profit.toFixed(0)}
                </div>
                <div className={`text-sm ${predictions.summary.best_bet_performance.profit >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                  {predictions.summary.best_bet_performance.roi ? `${predictions.summary.best_bet_performance.roi >= 0 ? '+' : ''}${predictions.summary.best_bet_performance.roi}% ROI` : '-'}
                </div>
              </div>

              <div className="bg-gray-50 rounded-lg p-4 text-center">
                <div className="text-sm text-gray-600 font-medium">Total Picks</div>
                <div className="text-2xl font-bold text-gray-700">
                  {predictions.summary.total_predictions}
                </div>
                {predictions.summary.pending_grading > 0 && (
                  <div className="text-sm text-amber-600">
                    {predictions.summary.pending_grading} pending
                  </div>
                )}
              </div>
            </div>

            {/* Performance by Value Bucket */}
            {predictions.by_value_bucket.length > 0 && (
              <div className="mt-4">
                <h3 className="text-sm font-semibold text-gray-700 mb-2">By Value Score</h3>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="text-left text-gray-500 border-b">
                        <th className="pb-2">Bucket</th>
                        <th className="pb-2 text-center">Record</th>
                        <th className="pb-2 text-center">Win Rate</th>
                        <th className="pb-2 text-right">Profit</th>
                        <th className="pb-2 text-right">ROI</th>
                      </tr>
                    </thead>
                    <tbody>
                      {predictions.by_value_bucket.map((bucket) => (
                        <tr key={bucket.bucket} className="border-b border-gray-100">
                          <td className="py-2 font-medium">{bucket.bucket}</td>
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
              </div>
            )}

            {/* Recent Predictions */}
            {predictions.recent_predictions.length > 0 && (
              <div className="mt-4">
                <h3 className="text-sm font-semibold text-gray-700 mb-2">Recent Predictions</h3>
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
                                • Best bet: {pred.best_bet.team} {pred.best_bet.type}
                                {pred.best_bet.line ? ` ${pred.best_bet.line > 0 ? '+' : ''}${pred.best_bet.line}` : ''}
                                {pred.best_bet.value_score && <span className="text-blue-600"> ({pred.best_bet.value_score}%)</span>}
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
              </div>
            )}

            {predictions.summary.total_predictions === 0 && (
              <div className="text-center py-8 text-gray-500">
                No graded predictions yet. Results will appear after games complete.
              </div>
            )}
          </div>
        )}
      </div>

      {comparisonLoading && (
        <div className="flex justify-center py-12">
          <LoadingSpinner size="lg" />
        </div>
      )}

      {comparison && (
        <>
          {/* Summary Card */}
          <div className="card">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">
              Performance Summary
            </h2>
            <div className="grid grid-cols-2 gap-8">
              <div>
                <h3 className="font-medium text-gray-700 mb-3">Algorithm A</h3>
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-500">Brier Score</span>
                    <span>{comparison.algo_a_metrics.brier_score?.toFixed(4) || '-'}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-500">CLV Avg</span>
                    <span>
                      {comparison.algo_a_metrics.clv_avg
                        ? `${(comparison.algo_a_metrics.clv_avg * 100).toFixed(2)}%`
                        : '-'}
                    </span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-500">ROI</span>
                    <span>
                      {comparison.algo_a_metrics.roi
                        ? `${(comparison.algo_a_metrics.roi * 100).toFixed(2)}%`
                        : '-'}
                    </span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-500">Win Rate</span>
                    <span>
                      {comparison.algo_a_metrics.win_rate
                        ? `${(comparison.algo_a_metrics.win_rate * 100).toFixed(1)}%`
                        : '-'}
                    </span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-500">Bet Count</span>
                    <span>{comparison.algo_a_metrics.bet_count || 0}</span>
                  </div>
                </div>
              </div>

              <div>
                <h3 className="font-medium text-gray-700 mb-3">Algorithm B</h3>
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-500">Brier Score</span>
                    <span>{comparison.algo_b_metrics.brier_score?.toFixed(4) || '-'}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-500">CLV Avg</span>
                    <span>
                      {comparison.algo_b_metrics.clv_avg
                        ? `${(comparison.algo_b_metrics.clv_avg * 100).toFixed(2)}%`
                        : '-'}
                    </span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-500">ROI</span>
                    <span>
                      {comparison.algo_b_metrics.roi
                        ? `${(comparison.algo_b_metrics.roi * 100).toFixed(2)}%`
                        : '-'}
                    </span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-500">Win Rate</span>
                    <span>
                      {comparison.algo_b_metrics.win_rate
                        ? `${(comparison.algo_b_metrics.win_rate * 100).toFixed(1)}%`
                        : '-'}
                    </span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-500">Bet Count</span>
                    <span>{comparison.algo_b_metrics.bet_count || 0}</span>
                  </div>
                </div>
              </div>
            </div>

            {comparison.recommendation !== 'insufficient_data' && (
              <div className="mt-6 pt-4 border-t">
                <p className="text-sm">
                  <span className="font-medium">Recommendation: </span>
                  <span
                    className={
                      comparison.recommendation === 'algo_a'
                        ? 'text-blue-600'
                        : comparison.recommendation === 'algo_b'
                        ? 'text-green-600'
                        : 'text-gray-600'
                    }
                  >
                    {comparison.recommendation === 'algo_a' && 'Use Algorithm A'}
                    {comparison.recommendation === 'algo_b' && 'Use Algorithm B'}
                    {comparison.recommendation === 'no_difference' &&
                      'No significant difference'}
                  </span>
                </p>
              </div>
            )}

            {comparison.recommendation === 'insufficient_data' && (
              <div className="mt-6 pt-4 border-t">
                <p className="text-sm text-gray-500">
                  Insufficient data for comparison. Need more completed bets.
                </p>
              </div>
            )}
          </div>

          {/* Performance by Bucket */}
          <div className="card">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-lg font-semibold text-gray-900">
                Performance by Value Score Bucket
              </h2>

              <div className="flex rounded-lg overflow-hidden border border-gray-200">
                {(['win_rate', 'roi', 'clv_avg'] as const).map((m) => (
                  <button
                    key={m}
                    onClick={() => setMetric(m)}
                    className={`px-3 py-1.5 text-sm font-medium transition-colors ${
                      metric === m
                        ? 'bg-blue-600 text-white'
                        : 'bg-white text-gray-700 hover:bg-gray-50'
                    }`}
                  >
                    {m === 'win_rate' ? 'Win Rate' : m === 'roi' ? 'ROI' : 'CLV'}
                  </button>
                ))}
              </div>
            </div>

            {isLoadingPerformance && (
              <div className="flex justify-center py-8">
                <LoadingSpinner />
              </div>
            )}

            {!isLoadingPerformance && hasPerformanceData && (
              <PerformanceBucketsChart
                dataAlgoA={performanceA}
                dataAlgoB={performanceB}
                metric={metric}
              />
            )}

            {!isLoadingPerformance && !hasPerformanceData && (
              <EmptyChart
                message="Need more completed bets to show performance breakdown"
                height={300}
              />
            )}
          </div>

          {/* Daily Results */}
          <div className="card">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-lg font-semibold text-gray-900">
                Daily Results
              </h2>

              <div className="flex items-center gap-4">
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
                    <option value={70}>70+</option>
                  </select>
                </div>
                <div className="flex rounded-lg overflow-hidden border border-gray-200">
                  {(['a', 'b'] as const).map((algo) => (
                    <button
                      key={algo}
                      onClick={() => setDailyAlgo(algo)}
                      className={`px-3 py-1.5 text-sm font-medium transition-colors ${
                        dailyAlgo === algo
                          ? 'bg-blue-600 text-white'
                          : 'bg-white text-gray-700 hover:bg-gray-50'
                      }`}
                    >
                      Algo {algo.toUpperCase()}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {dailyLoading && (
              <div className="flex justify-center py-8">
                <LoadingSpinner />
              </div>
            )}

            {!dailyLoading && dailyResults && dailyResults.length > 0 && (
              <div className="space-y-4">
                {/* Summary totals */}
                <div className="bg-gray-50 rounded-lg p-4">
                  <div className="grid grid-cols-4 gap-4 text-center">
                    <div>
                      <div className="text-sm text-gray-500">Total Bets</div>
                      <div className="text-xl font-bold">
                        {dailyResults.reduce((sum, d) => sum + d.wins + d.losses, 0)}
                      </div>
                    </div>
                    <div>
                      <div className="text-sm text-gray-500">Record</div>
                      <div className="text-xl font-bold">
                        {dailyResults.reduce((sum, d) => sum + d.wins, 0)}-
                        {dailyResults.reduce((sum, d) => sum + d.losses, 0)}
                        {dailyResults.reduce((sum, d) => sum + d.pushes, 0) > 0 &&
                          `-${dailyResults.reduce((sum, d) => sum + d.pushes, 0)}`}
                      </div>
                    </div>
                    <div>
                      <div className="text-sm text-gray-500">P/L</div>
                      <div
                        className={`text-xl font-bold ${
                          dailyResults.reduce((sum, d) => sum + d.profit, 0) >= 0
                            ? 'text-green-600'
                            : 'text-red-600'
                        }`}
                      >
                        {dailyResults.reduce((sum, d) => sum + d.profit, 0) >= 0 ? '+' : ''}$
                        {dailyResults.reduce((sum, d) => sum + d.profit, 0).toFixed(0)}
                      </div>
                    </div>
                    <div>
                      <div className="text-sm text-gray-500">ROI</div>
                      <div
                        className={`text-xl font-bold ${
                          (() => {
                            const totalBets = dailyResults.reduce(
                              (sum, d) => sum + d.wins + d.losses,
                              0
                            );
                            const totalProfit = dailyResults.reduce((sum, d) => sum + d.profit, 0);
                            return totalBets > 0 ? totalProfit / (totalBets * 100) : 0;
                          })() >= 0
                            ? 'text-green-600'
                            : 'text-red-600'
                        }`}
                      >
                        {(() => {
                          const totalBets = dailyResults.reduce(
                            (sum, d) => sum + d.wins + d.losses,
                            0
                          );
                          const totalProfit = dailyResults.reduce((sum, d) => sum + d.profit, 0);
                          const roi = totalBets > 0 ? (totalProfit / (totalBets * 100)) * 100 : 0;
                          return `${roi >= 0 ? '+' : ''}${roi.toFixed(1)}%`;
                        })()}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Daily breakdown */}
                {dailyResults.map((day) => (
                  <div key={day.date} className="border rounded-lg overflow-hidden">
                    <div className="bg-gray-50 px-4 py-2 flex justify-between items-center">
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
                                {bet.matchup} • {bet.final_score}
                              </div>
                            </div>
                          </div>
                          <div className="flex items-center gap-4">
                            <span className="text-xs bg-blue-100 text-blue-700 px-2 py-0.5 rounded">
                              {bet.value_score}%
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
        </>
      )}
    </div>
  );
}
