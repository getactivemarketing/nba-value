import { clsx } from 'clsx';

interface ValueBadgeProps {
  score: number;
  size?: 'sm' | 'md' | 'lg';
  showLabel?: boolean;
}

export function ValueBadge({ score, size = 'md', showLabel = false }: ValueBadgeProps) {
  const isElite = score >= 75;
  const isStrong = score >= 70;

  const getColorClass = () => {
    if (isElite) return 'bg-gradient-to-r from-amber-400 to-yellow-300 text-amber-900';
    if (isStrong) return 'bg-gradient-to-r from-amber-500 to-amber-400 text-white';
    if (score >= 50) return 'bg-emerald-500 text-white';
    return 'bg-slate-400 text-white';
  };

  const sizeClasses = {
    sm: 'text-xs px-2 py-0.5',
    md: 'text-sm px-3 py-1',
    lg: 'text-base px-4 py-1.5 font-bold',
  };

  return (
    <span className="inline-flex items-center gap-1.5">
      <span
        className={clsx(
          'inline-flex items-center gap-1 rounded font-semibold shadow-sm',
          getColorClass(),
          sizeClasses[size],
          isStrong && 'ring-1 ring-amber-400/50'
        )}
      >
        {isStrong && (
          <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
            <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
          </svg>
        )}
        {Math.round(score)}%
      </span>
      {showLabel && isElite && (
        <span className="text-xs font-bold text-amber-600 uppercase tracking-wide">
          Elite
        </span>
      )}
    </span>
  );
}

// Edge badge component for consistency
export function EdgeBadge({ edge }: { edge: number }) {
  const getColor = (e: number) => {
    if (e >= 15) return 'text-emerald-600';
    if (e >= 10) return 'text-emerald-500';
    if (e >= 5) return 'text-amber-600';
    return 'text-gray-600';
  };

  return (
    <span className={`${getColor(edge)} font-semibold`}>
      +{edge.toFixed(1)}%
    </span>
  );
}
