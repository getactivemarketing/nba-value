import { clsx } from 'clsx';

interface ExplainabilityCardProps {
  // Value components
  valueScore: number;
  edge: number;  // percentage (e.g., 3.5 for 3.5%)
  confidence: number;  // multiplier (e.g., 1.2)
  marketQuality: number;  // multiplier (e.g., 0.95)

  // Probabilities
  modelProb: number;  // percentage (e.g., 55.2)
  marketProb: number;  // percentage (e.g., 48.5)

  // Context
  marketType: string;
  team: string;
  line: number | null;
}

function getEdgeBand(edge: number): { label: string; color: string; description: string } {
  const absEdge = Math.abs(edge);
  if (absEdge < 2) {
    return {
      label: 'Tiny',
      color: 'text-gray-500 bg-gray-100',
      description: 'Very small edge, lower confidence'
    };
  } else if (absEdge < 5) {
    return {
      label: 'Small',
      color: 'text-blue-600 bg-blue-50',
      description: 'Moderate edge detected'
    };
  } else if (absEdge < 10) {
    return {
      label: 'Medium',
      color: 'text-green-600 bg-green-50',
      description: 'Strong edge detected'
    };
  } else {
    return {
      label: 'Large',
      color: 'text-amber-600 bg-amber-50',
      description: 'Very large edge (verify carefully)'
    };
  }
}

function getConfidenceLevel(confidence: number): { label: string; color: string } {
  if (confidence >= 1.3) {
    return { label: 'High', color: 'text-green-600' };
  } else if (confidence >= 1.0) {
    return { label: 'Normal', color: 'text-blue-600' };
  } else if (confidence >= 0.8) {
    return { label: 'Reduced', color: 'text-yellow-600' };
  } else {
    return { label: 'Low', color: 'text-red-600' };
  }
}

export function ExplainabilityCard({
  valueScore,
  edge,
  confidence,
  marketQuality,
  modelProb,
  marketProb,
  marketType,
  team,
  line,
}: ExplainabilityCardProps) {
  const edgeBand = getEdgeBand(edge);
  const confLevel = getConfidenceLevel(confidence);

  // Build factors list
  const factors: { icon: string; text: string; good: boolean }[] = [];

  // Edge factor
  if (edge > 0) {
    factors.push({
      icon: '+',
      text: `Model sees ${edge.toFixed(1)}% edge (${edgeBand.label})`,
      good: true,
    });
  }

  // Confidence factor
  if (confidence >= 1.1) {
    factors.push({
      icon: '+',
      text: `${confLevel.label} confidence (${confidence.toFixed(2)}x boost)`,
      good: true,
    });
  } else if (confidence < 0.9) {
    factors.push({
      icon: '-',
      text: `${confLevel.label} confidence (${confidence.toFixed(2)}x penalty)`,
      good: false,
    });
  }

  // Market quality factor
  if (marketQuality >= 1.0) {
    factors.push({
      icon: '+',
      text: 'Good market conditions',
      good: true,
    });
  } else if (marketQuality < 0.85) {
    factors.push({
      icon: '-',
      text: 'Sub-optimal market timing',
      good: false,
    });
  }

  // Probability comparison
  const probDiff = modelProb - marketProb;
  if (probDiff > 3) {
    factors.push({
      icon: '+',
      text: `Model: ${modelProb.toFixed(0)}% vs Market: ${marketProb.toFixed(0)}%`,
      good: true,
    });
  }

  return (
    <div className="bg-gradient-to-br from-slate-50 to-white rounded-lg border border-slate-200 p-4">
      <div className="flex items-center gap-2 mb-3">
        <svg className="w-5 h-5 text-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
        </svg>
        <h4 className="text-sm font-semibold text-slate-800">Why We Like This Bet</h4>
      </div>

      {/* Main pick summary */}
      <div className="mb-3 p-2 bg-white rounded border border-slate-100">
        <div className="flex items-center justify-between">
          <div className="text-sm font-medium text-slate-700">
            {team} {marketType === 'total' ? '' : marketType === 'spread' ? (line !== null ? (line > 0 ? `+${line}` : line) : '') : 'ML'}
          </div>
          <div className={clsx(
            'px-2 py-0.5 rounded text-xs font-semibold',
            valueScore >= 70 ? 'bg-amber-100 text-amber-700' :
            valueScore >= 60 ? 'bg-emerald-100 text-emerald-700' :
            'bg-slate-100 text-slate-600'
          )}>
            {valueScore.toFixed(0)}% Value
          </div>
        </div>
      </div>

      {/* Factors */}
      <div className="space-y-1.5 mb-3">
        {factors.map((factor, i) => (
          <div key={i} className="flex items-start gap-2 text-xs">
            <span className={clsx(
              'w-4 h-4 rounded-full flex items-center justify-center text-xs font-bold shrink-0',
              factor.good ? 'bg-green-100 text-green-600' : 'bg-red-100 text-red-600'
            )}>
              {factor.icon}
            </span>
            <span className="text-slate-600">{factor.text}</span>
          </div>
        ))}
      </div>

      {/* Value Score Breakdown */}
      <div className="pt-3 border-t border-slate-100">
        <div className="text-[10px] text-slate-400 uppercase tracking-wide mb-2">Score Breakdown</div>
        <div className="grid grid-cols-3 gap-2 text-center">
          <div className="bg-white rounded p-2 border border-slate-100">
            <div className="text-xs text-slate-500">Edge</div>
            <div className={clsx('text-sm font-semibold', edgeBand.color.split(' ')[0])}>
              {edge > 0 ? '+' : ''}{edge.toFixed(1)}%
            </div>
          </div>
          <div className="bg-white rounded p-2 border border-slate-100">
            <div className="text-xs text-slate-500">Confidence</div>
            <div className={clsx('text-sm font-semibold', confLevel.color)}>
              {confidence.toFixed(2)}x
            </div>
          </div>
          <div className="bg-white rounded p-2 border border-slate-100">
            <div className="text-xs text-slate-500">Mkt Quality</div>
            <div className="text-sm font-semibold text-slate-700">
              {marketQuality.toFixed(2)}x
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// Compact version for inline use in GameCard
export function ValueBreakdownTooltip({
  edge,
  confidence,
  marketQuality,
  modelProb,
  marketProb,
}: {
  edge: number;
  confidence: number;
  marketQuality: number;
  modelProb: number;
  marketProb: number;
}) {
  return (
    <div className="text-xs space-y-1 p-2">
      <div className="font-medium mb-1">Value Breakdown:</div>
      <div>Model: {modelProb.toFixed(0)}% vs Market: {marketProb.toFixed(0)}%</div>
      <div>Edge: {edge > 0 ? '+' : ''}{edge.toFixed(1)}%</div>
      <div>Confidence: {confidence.toFixed(2)}x</div>
      <div>Market Quality: {marketQuality.toFixed(2)}x</div>
    </div>
  );
}
