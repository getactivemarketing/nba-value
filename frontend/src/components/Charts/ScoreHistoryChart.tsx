import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { format } from 'date-fns';
import type { BetHistory } from '@/types/market';

interface ScoreHistoryChartProps {
  data: BetHistory[];
  showAlgoA?: boolean;
  showAlgoB?: boolean;
}

export function ScoreHistoryChart({
  data,
  showAlgoA = true,
  showAlgoB = true,
}: ScoreHistoryChartProps) {
  const chartData = data
    .map((point) => ({
      time: format(new Date(point.calc_time), 'MMM d HH:mm'),
      timestamp: new Date(point.calc_time).getTime(),
      algoA: point.algo_a_value_score,
      algoB: point.algo_b_value_score,
      odds: point.odds_decimal,
      edge: point.raw_edge * 100,
    }))
    .sort((a, b) => a.timestamp - b.timestamp);

  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
        <XAxis
          dataKey="time"
          tick={{ fontSize: 12 }}
          stroke="#9CA3AF"
        />
        <YAxis
          domain={[0, 100]}
          tick={{ fontSize: 12 }}
          stroke="#9CA3AF"
          label={{ value: 'Value Score', angle: -90, position: 'insideLeft', style: { fontSize: 12 } }}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: '#fff',
            border: '1px solid #E5E7EB',
            borderRadius: '8px',
          }}
          formatter={(value: number, name: string) => [
            value.toFixed(1),
            name === 'algoA' ? 'Algorithm A' : 'Algorithm B',
          ]}
        />
        <Legend
          formatter={(value) => (value === 'algoA' ? 'Algorithm A' : 'Algorithm B')}
        />
        {showAlgoA && (
          <Line
            type="monotone"
            dataKey="algoA"
            stroke="#3B82F6"
            name="algoA"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4 }}
          />
        )}
        {showAlgoB && (
          <Line
            type="monotone"
            dataKey="algoB"
            stroke="#10B981"
            name="algoB"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4 }}
          />
        )}
      </LineChart>
    </ResponsiveContainer>
  );
}
