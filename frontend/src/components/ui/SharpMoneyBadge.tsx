import { clsx } from 'clsx';

interface SharpMoneyBadgeProps {
  signal: 'sharp_home' | 'sharp_away' | 'neutral';
  homeTeam: string;
  awayTeam: string;
  spreadMovement: number;
  showDetails?: boolean;
  size?: 'sm' | 'md';
}

export function SharpMoneyBadge({
  signal,
  homeTeam,
  awayTeam,
  spreadMovement,
  showDetails = true,
  size = 'md',
}: SharpMoneyBadgeProps) {
  if (signal === 'neutral') {
    return null; // Don't show badge for neutral signals
  }

  const isSharpHome = signal === 'sharp_home';
  const team = isSharpHome ? homeTeam : awayTeam;
  const movement = Math.abs(spreadMovement);

  const sizeClasses = {
    sm: 'text-xs px-1.5 py-0.5 gap-1',
    md: 'text-xs px-2 py-1 gap-1.5',
  };

  return (
    <span
      className={clsx(
        'inline-flex items-center rounded-full font-medium',
        sizeClasses[size],
        isSharpHome
          ? 'bg-blue-50 text-blue-700 ring-1 ring-blue-200'
          : 'bg-orange-50 text-orange-700 ring-1 ring-orange-200'
      )}
      title={`Line moved ${movement.toFixed(1)} pts toward ${team}, suggesting sharp money`}
    >
      {/* Target icon */}
      <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <circle cx="12" cy="12" r="10" strokeWidth="2" />
        <circle cx="12" cy="12" r="6" strokeWidth="2" />
        <circle cx="12" cy="12" r="2" fill="currentColor" />
      </svg>
      <span>Sharp: {team}</span>
      {showDetails && (
        <span className="text-[10px] opacity-75">
          ({spreadMovement > 0 ? '+' : ''}{spreadMovement.toFixed(1)})
        </span>
      )}
    </span>
  );
}

// Compact version for inline use
export function SharpMoneyIndicator({
  signal,
  homeTeam,
  awayTeam,
}: {
  signal: 'sharp_home' | 'sharp_away' | 'neutral';
  homeTeam: string;
  awayTeam: string;
}) {
  if (signal === 'neutral') {
    return null;
  }

  const isSharpHome = signal === 'sharp_home';
  const team = isSharpHome ? homeTeam : awayTeam;

  return (
    <span
      className={clsx(
        'inline-flex items-center gap-0.5 text-[10px] font-medium',
        isSharpHome ? 'text-blue-600' : 'text-orange-600'
      )}
      title={`Sharp money on ${team}`}
    >
      <svg className="w-2.5 h-2.5" fill="currentColor" viewBox="0 0 20 20">
        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
      </svg>
      Sharp
    </span>
  );
}
