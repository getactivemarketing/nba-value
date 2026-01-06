import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import type { PerformanceBucket } from '@/types/evaluation';

interface PerformanceBucketsChartProps {
  dataAlgoA: PerformanceBucket[];
  dataAlgoB: PerformanceBucket[];
  metric: 'win_rate' | 'roi' | 'clv_avg';
}

const metricLabels: Record<string, string> = {
  win_rate: 'Win Rate (%)',
  roi: 'ROI (%)',
  clv_avg: 'CLV (%)',
};

export function PerformanceBucketsChart({
  dataAlgoA,
  dataAlgoB,
  metric,
}: PerformanceBucketsChartProps) {
  const chartData = dataAlgoA.map((bucket, index) => {
    const algoAValue = bucket[metric] * 100;
    const algoBValue = dataAlgoB[index] ? dataAlgoB[index][metric] * 100 : 0;

    return {
      bucket: `${bucket.bucket_start}-${bucket.bucket_end}`,
      algoA: algoAValue,
      algoB: algoBValue,
      countA: bucket.bet_count,
      countB: dataAlgoB[index]?.bet_count || 0,
    };
  });

  return (
    <ResponsiveContainer width="100%" height={350}>
      <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 25 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
        <XAxis
          dataKey="bucket"
          tick={{ fontSize: 12 }}
          stroke="#9CA3AF"
          label={{ value: 'Value Score Range', position: 'insideBottom', offset: -15, style: { fontSize: 12 } }}
        />
        <YAxis
          tick={{ fontSize: 12 }}
          stroke="#9CA3AF"
          label={{ value: metricLabels[metric], angle: -90, position: 'insideLeft', style: { fontSize: 12 } }}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: '#fff',
            border: '1px solid #E5E7EB',
            borderRadius: '8px',
          }}
          formatter={(value: number, name: string, props) => {
            const count = name === 'algoA' ? props.payload.countA : props.payload.countB;
            return [
              `${value.toFixed(1)}% (n=${count})`,
              name === 'algoA' ? 'Algorithm A' : 'Algorithm B',
            ];
          }}
        />
        <Legend
          formatter={(value) => (value === 'algoA' ? 'Algorithm A' : 'Algorithm B')}
        />
        <Bar dataKey="algoA" name="algoA" fill="#3B82F6" radius={[4, 4, 0, 0]} />
        <Bar dataKey="algoB" name="algoB" fill="#10B981" radius={[4, 4, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}
