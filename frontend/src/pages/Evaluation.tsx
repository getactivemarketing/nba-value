import { useState } from 'react';
import { useAlgorithmComparison, usePerformanceByBucket } from '@/hooks/useEvaluation';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { PerformanceBucketsChart, EmptyChart } from '@/components/Charts';

type Metric = 'win_rate' | 'roi' | 'clv_avg';

export function Evaluation() {
  const [metric, setMetric] = useState<Metric>('win_rate');

  const { data: comparison, isLoading: comparisonLoading } = useAlgorithmComparison();
  const { data: performanceA, isLoading: perfALoading } = usePerformanceByBucket('a', 'score');
  const { data: performanceB, isLoading: perfBLoading } = usePerformanceByBucket('b', 'score');

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
        </>
      )}
    </div>
  );
}
