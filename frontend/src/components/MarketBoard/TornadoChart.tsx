interface TornadoFactor {
  factor: string;
  label: string;
  home_value: number | string;
  away_value: number | string;
  diff: number;
  home_better: boolean | null;
  expected_pace?: number;
}

interface TornadoChartProps {
  factors: TornadoFactor[];
  homeTeam: string;
  awayTeam: string;
}

export function TornadoChart({ factors, homeTeam, awayTeam }: TornadoChartProps) {
  if (!factors || factors.length === 0) {
    return null;
  }

  return (
    <div className="space-y-2">
      {/* Header */}
      <div className="flex justify-between text-xs font-semibold text-gray-500 px-1">
        <span>{awayTeam}</span>
        <span>Factor</span>
        <span>{homeTeam}</span>
      </div>

      {/* Factors */}
      {factors.map((f) => (
        <div key={f.factor} className="flex items-center gap-2">
          {/* Away value */}
          <div className="w-16 text-xs text-right font-medium text-gray-700">
            {f.away_value}
          </div>

          {/* Bar container */}
          <div className="flex-1 relative h-5">
            {/* Center line */}
            <div className="absolute left-1/2 top-0 bottom-0 w-px bg-gray-300" />

            {/* Label in center */}
            <div className="absolute inset-0 flex items-center justify-center">
              <span className="text-[10px] text-gray-500 bg-white px-1 z-10">
                {f.label}
              </span>
            </div>

            {/* Bar */}
            {f.diff !== 0 && (
              <div
                className={`absolute top-0.5 bottom-0.5 rounded transition-all ${
                  f.diff > 0
                    ? 'bg-blue-500/70 left-1/2'
                    : 'bg-red-500/70 right-1/2'
                }`}
                style={{
                  width: `${Math.abs(f.diff) / 2}%`,
                }}
              />
            )}

            {/* Pace special case - show expected value */}
            {f.factor === 'Pace' && f.expected_pace && (
              <div className="absolute inset-0 flex items-center justify-center">
                <span className="text-[10px] text-gray-500 bg-white px-1 z-10">
                  {f.label} ~{f.expected_pace}
                </span>
              </div>
            )}
          </div>

          {/* Home value */}
          <div className="w-16 text-xs text-left font-medium text-gray-700">
            {f.home_value}
          </div>
        </div>
      ))}

      {/* Legend */}
      <div className="flex justify-center gap-4 text-[10px] text-gray-400 pt-1">
        <span className="flex items-center gap-1">
          <span className="w-2 h-2 rounded bg-red-500/70" />
          {awayTeam} advantage
        </span>
        <span className="flex items-center gap-1">
          <span className="w-2 h-2 rounded bg-blue-500/70" />
          {homeTeam} advantage
        </span>
      </div>
    </div>
  );
}
