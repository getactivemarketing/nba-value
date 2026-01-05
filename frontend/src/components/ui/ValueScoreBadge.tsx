import { clsx } from 'clsx';

interface ValueScoreBadgeProps {
  score: number;
  size?: 'sm' | 'md' | 'lg';
}

export function ValueScoreBadge({ score, size = 'md' }: ValueScoreBadgeProps) {
  const getColorClass = (score: number) => {
    if (score >= 80) return 'bg-green-500 text-white';
    if (score >= 60) return 'bg-green-400 text-white';
    if (score >= 40) return 'bg-yellow-400 text-gray-900';
    if (score >= 20) return 'bg-orange-400 text-white';
    return 'bg-gray-300 text-gray-700';
  };

  const sizeClasses = {
    sm: 'px-2 py-0.5 text-xs',
    md: 'px-3 py-1 text-sm',
    lg: 'px-4 py-2 text-base font-semibold',
  };

  return (
    <span
      className={clsx(
        'inline-flex items-center rounded-full font-medium',
        getColorClass(score),
        sizeClasses[size]
      )}
    >
      {score.toFixed(1)}
    </span>
  );
}
