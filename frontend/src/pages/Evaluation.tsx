import { useState } from 'react';
import { useAlgorithmComparison, usePerformanceByBucket, useDailyResults } from '@/hooks/useEvaluation';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { PerformanceBucketsChart, EmptyChart } from '@/components/Charts';
import type { Algorithm } from '@/types/market';

type Metric = 'win_rate' | 'roi' | 'clv_avg';

export function Evaluation() {
  const [metric, setMetric] = useState<Metric>('win_rate');
  const [dailyAlgo, setDailyAlgo] = useState<Algorithm>('b');
  const [minValue, setMinValue] = useState(50);

  const { data: comparison, isLoading: comparisonLoading } = useAlgorithmComparison();
  const { data: performanceA, isLoading: perfALoading } = usePerformanceByBucket('a', 'score');
  const { data: performanceB, isLoading: perfBLoading } = usePerformanceByBucket('b', 'score');
  const { data: dailyResults, isLoading: dailyLoading } = useDailyResults(14, dailyAlgo, minValue);

  const isLoadingPerformance = perfALoading || perfBLoading;
  const hasPerformanceData =
    performanceA && performanceA.length > 0 && performanceB && performanceB.length > 0;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Algorithm Evaluation</h1>
        <p className="text-sm text-gray-500 mt-1">
          Compare performance between Algorithm A and Algorithm B
        </p>
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
