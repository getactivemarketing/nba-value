import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';

export interface LineMovementPoint {
  snapshot_time: string;
  minutes_to_tip: number;
  home_spread: number | null;
  away_spread: number | null;
  total_line: number | null;
  home_spread_odds: number | null;
  over_odds: number | null;
}

interface LineMovementChartProps {
  snapshots: LineMovementPoint[];
  homeTeam: string;
  awayTeam: string;
}

interface ChartDataPoint {
  time: string;
  displayTime: string;
  minutesToTip: number;
  spread: number | null;
  total: number | null;
}

export function LineMovementChart({ snapshots, homeTeam, awayTeam: _awayTeam }: LineMovementChartProps) {
  if (!snapshots || snapshots.length === 0) {
    return (
      <div className="bg-gray-50 rounded-lg p-6 text-center text-gray-500 text-sm">
        No line movement data available yet
      </div>
    );
  }

  // Transform data for chart
  const chartData: ChartDataPoint[] = snapshots.map((s) => {
    const hoursToTip = Math.round(s.minutes_to_tip / 60);

    return {
      time: s.snapshot_time,
      displayTime: hoursToTip > 24
        ? `${Math.round(hoursToTip / 24)}d`
        : hoursToTip > 0
          ? `${hoursToTip}h`
          : `${s.minutes_to_tip}m`,
      minutesToTip: s.minutes_to_tip,
      spread: s.home_spread,
      total: s.total_line,
    };
  });

  // Calculate opening and current values
  const openingSpread = chartData[0]?.spread;
  const currentSpread = chartData[chartData.length - 1]?.spread;
  const spreadMovement = openingSpread !== null && currentSpread !== null
    ? currentSpread - openingSpread
    : null;

  const openingTotal = chartData[0]?.total;
  const currentTotal = chartData[chartData.length - 1]?.total;
  const totalMovement = openingTotal !== null && currentTotal !== null
    ? currentTotal - openingTotal
    : null;

  // Get spread range for Y axis
  const spreads = chartData.map(d => d.spread).filter((s): s is number => s !== null);
  const minSpread = Math.min(...spreads);
  const maxSpread = Math.max(...spreads);
  const spreadPadding = Math.max((maxSpread - minSpread) * 0.3, 0.5);

  // Get total range for Y axis
  const totals = chartData.map(d => d.total).filter((t): t is number => t !== null);
  const minTotal = Math.min(...totals);
  const maxTotal = Math.max(...totals);
  const totalPadding = Math.max((maxTotal - minTotal) * 0.3, 1);

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-4">
      <div className="flex items-center justify-between mb-3">
        <h4 className="text-sm font-semibold text-gray-900">Line Movement</h4>
        <div className="flex gap-4 text-xs">
          {spreadMovement !== null && (
            <div className={`${spreadMovement < 0 ? 'text-blue-600' : spreadMovement > 0 ? 'text-orange-600' : 'text-gray-500'}`}>
              Spread: {currentSpread !== null ? currentSpread > 0 ? `+${currentSpread}` : currentSpread : '-'}
              {spreadMovement !== 0 && (
                <span className="ml-1">
                  ({spreadMovement > 0 ? '+' : ''}{spreadMovement.toFixed(1)})
                </span>
              )}
            </div>
          )}
          {totalMovement !== null && (
            <div className={`${totalMovement > 0 ? 'text-green-600' : totalMovement < 0 ? 'text-red-600' : 'text-gray-500'}`}>
              Total: {currentTotal}
              {totalMovement !== 0 && (
                <span className="ml-1">
                  ({totalMovement > 0 ? '+' : ''}{totalMovement.toFixed(1)})
                </span>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Spread Chart */}
      {spreads.length > 0 && (
        <div className="mb-4">
          <div className="text-xs text-gray-500 mb-1">Spread ({homeTeam})</div>
          <div className="h-32">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 5, right: 5, left: -10, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis
                  dataKey="displayTime"
                  tick={{ fontSize: 10, fill: '#9ca3af' }}
                  tickLine={false}
                  axisLine={{ stroke: '#e5e7eb' }}
                  reversed
                />
                <YAxis
                  domain={[minSpread - spreadPadding, maxSpread + spreadPadding]}
                  tick={{ fontSize: 10, fill: '#9ca3af' }}
                  tickLine={false}
                  axisLine={{ stroke: '#e5e7eb' }}
                  tickFormatter={(v) => v > 0 ? `+${v}` : v.toString()}
                />
                <Tooltip
                  content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      const data = payload[0].payload as ChartDataPoint;
                      return (
                        <div className="bg-white border border-gray-200 rounded shadow-lg p-2 text-xs">
                          <p className="text-gray-500">{Math.round(data.minutesToTip / 60)}h before tip</p>
                          <p className="font-semibold">
                            {homeTeam}: {data.spread !== null ? (data.spread > 0 ? `+${data.spread}` : data.spread) : '-'}
                          </p>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                {openingSpread !== null && (
                  <ReferenceLine y={openingSpread} stroke="#9ca3af" strokeDasharray="3 3" />
                )}
                <Line
                  type="stepAfter"
                  dataKey="spread"
                  stroke="#3b82f6"
                  strokeWidth={2}
                  dot={false}
                  activeDot={{ r: 4, fill: '#3b82f6' }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Total Chart */}
      {totals.length > 0 && (
        <div>
          <div className="text-xs text-gray-500 mb-1">Total</div>
          <div className="h-32">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 5, right: 5, left: -10, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis
                  dataKey="displayTime"
                  tick={{ fontSize: 10, fill: '#9ca3af' }}
                  tickLine={false}
                  axisLine={{ stroke: '#e5e7eb' }}
                  reversed
                />
                <YAxis
                  domain={[minTotal - totalPadding, maxTotal + totalPadding]}
                  tick={{ fontSize: 10, fill: '#9ca3af' }}
                  tickLine={false}
                  axisLine={{ stroke: '#e5e7eb' }}
                />
                <Tooltip
                  content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      const data = payload[0].payload as ChartDataPoint;
                      return (
                        <div className="bg-white border border-gray-200 rounded shadow-lg p-2 text-xs">
                          <p className="text-gray-500">{Math.round(data.minutesToTip / 60)}h before tip</p>
                          <p className="font-semibold">Total: {data.total ?? '-'}</p>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                {openingTotal !== null && (
                  <ReferenceLine y={openingTotal} stroke="#9ca3af" strokeDasharray="3 3" />
                )}
                <Line
                  type="stepAfter"
                  dataKey="total"
                  stroke="#10b981"
                  strokeWidth={2}
                  dot={false}
                  activeDot={{ r: 4, fill: '#10b981' }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Movement Summary */}
      <div className="mt-3 pt-3 border-t border-gray-100 grid grid-cols-2 gap-4 text-xs">
        <div>
          <span className="text-gray-500">Opening:</span>
          <span className="ml-2 font-medium">
            {openingSpread !== null ? (openingSpread > 0 ? `+${openingSpread}` : openingSpread) : '-'} / {openingTotal ?? '-'}
          </span>
        </div>
        <div>
          <span className="text-gray-500">Current:</span>
          <span className="ml-2 font-medium">
            {currentSpread !== null ? (currentSpread > 0 ? `+${currentSpread}` : currentSpread) : '-'} / {currentTotal ?? '-'}
          </span>
        </div>
      </div>
    </div>
  );
}
