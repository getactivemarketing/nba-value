import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import type { DailyResult } from '@/hooks/useEvaluation';

interface CumulativePLChartProps {
  dailyResults: DailyResult[];
}

interface ChartDataPoint {
  date: string;
  displayDate: string;
  dailyProfit: number;
  cumulativeProfit: number;
  record: string;
}

export function CumulativePLChart({ dailyResults }: CumulativePLChartProps) {
  if (!dailyResults || dailyResults.length === 0) {
    return (
      <div className="bg-gray-50 rounded-lg p-8 text-center text-gray-500">
        No data available for chart
      </div>
    );
  }

  // Sort by date ascending and calculate cumulative profit
  const sortedResults = [...dailyResults].sort(
    (a, b) => new Date(a.date).getTime() - new Date(b.date).getTime()
  );

  let runningTotal = 0;
  const chartData: ChartDataPoint[] = sortedResults.map((day) => {
    runningTotal += day.profit;
    return {
      date: day.date,
      displayDate: new Date(day.date + 'T12:00:00').toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
      }),
      dailyProfit: day.profit,
      cumulativeProfit: runningTotal,
      record: day.record,
    };
  });

  const minValue = Math.min(...chartData.map((d) => d.cumulativeProfit));
  const maxValue = Math.max(...chartData.map((d) => d.cumulativeProfit));
  const padding = Math.max(Math.abs(maxValue - minValue) * 0.1, 50);

  const finalProfit = chartData[chartData.length - 1]?.cumulativeProfit ?? 0;
  const isPositive = finalProfit >= 0;

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900">Cumulative P/L</h3>
        <div
          className={`text-2xl font-bold ${
            isPositive ? 'text-green-600' : 'text-red-600'
          }`}
        >
          {isPositive ? '+' : ''}${finalProfit.toFixed(0)}
        </div>
      </div>

      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart
            data={chartData}
            margin={{ top: 10, right: 10, left: 0, bottom: 0 }}
          >
            <defs>
              <linearGradient id="colorProfit" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#10b981" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
              </linearGradient>
              <linearGradient id="colorLoss" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#ef4444" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#ef4444" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis
              dataKey="displayDate"
              tick={{ fontSize: 12, fill: '#6b7280' }}
              tickLine={false}
              axisLine={{ stroke: '#e5e7eb' }}
            />
            <YAxis
              tick={{ fontSize: 12, fill: '#6b7280' }}
              tickLine={false}
              axisLine={{ stroke: '#e5e7eb' }}
              tickFormatter={(value) => `$${value}`}
              domain={[minValue - padding, maxValue + padding]}
            />
            <Tooltip
              content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  const data = payload[0].payload as ChartDataPoint;
                  return (
                    <div className="bg-white border border-gray-200 rounded-lg shadow-lg p-3">
                      <p className="font-semibold text-gray-900">{data.displayDate}</p>
                      <p className="text-sm text-gray-500">Record: {data.record}</p>
                      <p
                        className={`text-sm font-medium ${
                          data.dailyProfit >= 0 ? 'text-green-600' : 'text-red-600'
                        }`}
                      >
                        Daily: {data.dailyProfit >= 0 ? '+' : ''}${data.dailyProfit.toFixed(0)}
                      </p>
                      <p
                        className={`text-sm font-bold ${
                          data.cumulativeProfit >= 0 ? 'text-green-600' : 'text-red-600'
                        }`}
                      >
                        Total: {data.cumulativeProfit >= 0 ? '+' : ''}${data.cumulativeProfit.toFixed(0)}
                      </p>
                    </div>
                  );
                }
                return null;
              }}
            />
            <ReferenceLine y={0} stroke="#9ca3af" strokeDasharray="3 3" />
            <Area
              type="monotone"
              dataKey="cumulativeProfit"
              stroke={isPositive ? '#10b981' : '#ef4444'}
              strokeWidth={2}
              fill={isPositive ? 'url(#colorProfit)' : 'url(#colorLoss)'}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Quick stats */}
      <div className="mt-4 grid grid-cols-3 gap-4 pt-4 border-t border-gray-100">
        <div className="text-center">
          <div className="text-sm text-gray-500">Best Day</div>
          <div className="font-semibold text-green-600">
            +${Math.max(...chartData.map((d) => d.dailyProfit)).toFixed(0)}
          </div>
        </div>
        <div className="text-center">
          <div className="text-sm text-gray-500">Worst Day</div>
          <div className="font-semibold text-red-600">
            ${Math.min(...chartData.map((d) => d.dailyProfit)).toFixed(0)}
          </div>
        </div>
        <div className="text-center">
          <div className="text-sm text-gray-500">Avg/Day</div>
          <div
            className={`font-semibold ${
              finalProfit / chartData.length >= 0 ? 'text-green-600' : 'text-red-600'
            }`}
          >
            {finalProfit / chartData.length >= 0 ? '+' : ''}$
            {(finalProfit / chartData.length).toFixed(0)}
          </div>
        </div>
      </div>
    </div>
  );
}
