import type { PitcherInfo } from '@/lib/mlbApi';

interface PitcherMatchupProps {
  homePitcher: PitcherInfo | null;
  awayPitcher: PitcherInfo | null;
}

function PitcherCard({ pitcher, isHome }: { pitcher: PitcherInfo | null; isHome: boolean }) {
  if (!pitcher) {
    return (
      <div className={`flex-1 p-3 bg-gray-50 rounded-lg ${isHome ? 'text-right' : 'text-left'}`}>
        <p className="text-sm text-gray-500">TBD</p>
      </div>
    );
  }

  const qualityColor = pitcher.quality_score
    ? pitcher.quality_score >= 70
      ? 'text-green-600'
      : pitcher.quality_score >= 50
      ? 'text-yellow-600'
      : 'text-red-600'
    : 'text-gray-600';

  return (
    <div className={`flex-1 p-3 bg-gray-50 rounded-lg ${isHome ? 'text-right' : 'text-left'}`}>
      <div className="flex items-center gap-2 justify-between">
        {!isHome && (
          <span className="text-xs font-medium text-gray-500 uppercase">
            {pitcher.throws === 'L' ? 'LHP' : 'RHP'}
          </span>
        )}
        <span className="font-semibold text-gray-900">{pitcher.name}</span>
        {isHome && (
          <span className="text-xs font-medium text-gray-500 uppercase">
            {pitcher.throws === 'L' ? 'LHP' : 'RHP'}
          </span>
        )}
      </div>

      <div className={`flex items-center gap-4 mt-2 ${isHome ? 'justify-end' : 'justify-start'}`}>
        {pitcher.era !== null && (
          <div className="text-center">
            <span className="text-lg font-bold text-gray-900">
              {pitcher.era.toFixed(2)}
            </span>
            <p className="text-xs text-gray-500">ERA</p>
          </div>
        )}
        {pitcher.whip !== null && (
          <div className="text-center">
            <span className="text-lg font-bold text-gray-900">
              {pitcher.whip.toFixed(2)}
            </span>
            <p className="text-xs text-gray-500">WHIP</p>
          </div>
        )}
        {pitcher.k_per_9 !== null && (
          <div className="text-center">
            <span className="text-lg font-bold text-gray-900">
              {pitcher.k_per_9.toFixed(1)}
            </span>
            <p className="text-xs text-gray-500">K/9</p>
          </div>
        )}
        {pitcher.quality_score !== null && (
          <div className="text-center">
            <span className={`text-lg font-bold ${qualityColor}`}>
              {pitcher.quality_score.toFixed(0)}
            </span>
            <p className="text-xs text-gray-500">QS</p>
          </div>
        )}
      </div>
    </div>
  );
}

export function PitcherMatchup({ homePitcher, awayPitcher }: PitcherMatchupProps) {
  // Calculate edge
  let edge: string | null = null;
  if (homePitcher?.quality_score && awayPitcher?.quality_score) {
    const diff = homePitcher.quality_score - awayPitcher.quality_score;
    if (Math.abs(diff) >= 5) {
      edge = diff > 0 ? 'HOME' : 'AWAY';
    }
  }

  return (
    <div className="border-t border-gray-100 pt-3 mt-3">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-semibold text-gray-500 uppercase tracking-wide">
          Pitcher Matchup
        </span>
        {edge && (
          <span className={`text-xs font-bold px-2 py-0.5 rounded ${
            edge === 'HOME' ? 'bg-blue-100 text-blue-700' : 'bg-red-100 text-red-700'
          }`}>
            {edge} EDGE
          </span>
        )}
      </div>

      <div className="flex items-stretch gap-3">
        <PitcherCard pitcher={awayPitcher} isHome={false} />
        <div className="flex items-center">
          <span className="text-sm font-medium text-gray-400">vs</span>
        </div>
        <PitcherCard pitcher={homePitcher} isHome={true} />
      </div>
    </div>
  );
}
