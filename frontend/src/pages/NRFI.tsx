import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { mlbApi, getTeamInfo, type FirstInningStats } from '@/lib/mlbApi';
import { getMLBLogo } from '@/lib/mlbLogos';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';

function MLBLogoCircle({ team, size = 32 }: { team: string; size?: number }) {
  const [imgError, setImgError] = useState(false);
  const teamInfo = getTeamInfo(team);
  const logoUrl = getMLBLogo(team);

  if (imgError || !logoUrl) {
    return (
      <div
        className="rounded-full flex items-center justify-center text-[10px] font-bold text-white flex-shrink-0"
        style={{ width: size, height: size, backgroundColor: teamInfo.color }}
      >
        {team}
      </div>
    );
  }

  return (
    <div
      className="rounded-full bg-white/5 flex items-center justify-center flex-shrink-0 overflow-hidden p-0.5"
      style={{ width: size, height: size }}
    >
      <img
        src={logoUrl}
        alt={team}
        className="w-full h-full object-contain"
        onError={() => setImgError(true)}
      />
    </div>
  );
}

function StatCard({ label, value, sub, accent }: { label: string; value: string; sub?: string; accent?: string }) {
  return (
    <div className="bg-[#191c22] rounded-xl border border-[#1e293b] p-4">
      <div className="text-[10px] text-[#64748b] uppercase font-bold tracking-widest mb-2">{label}</div>
      <div className={`text-2xl font-black font-mono ${accent || 'text-[#f1f5f9]'}`}>{value}</div>
      {sub && <div className="text-[11px] text-[#64748b] font-mono mt-1">{sub}</div>}
    </div>
  );
}

function TeamRow({ stat, mode }: { stat: FirstInningStats; mode: 'nrfi' | 'yrfi' }) {
  const pct = stat.score_pct * 100;
  const teamInfo = getTeamInfo(stat.team);
  const barPct = Math.min(100, Math.max(0, pct));

  return (
    <div className="px-4 py-3 border-b border-[#1e293b] last:border-b-0 hover:bg-[#0b0e14] transition-colors">
      <div className="flex items-center gap-3">
        <MLBLogoCircle team={stat.team} size={32} />
        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between mb-1.5">
            <span className="text-sm font-bold text-[#f1f5f9]">{teamInfo.name}</span>
            <span className={`text-sm font-mono font-black ${mode === 'nrfi' ? 'text-[#66f796]' : 'text-[#ef4444]'}`}>
              {pct.toFixed(1)}%
            </span>
          </div>
          <div className="h-1.5 bg-[#0b0e14] rounded-full overflow-hidden mb-1.5">
            <div
              className="h-full rounded-full"
              style={{
                width: `${barPct}%`,
                background:
                  mode === 'nrfi'
                    ? 'linear-gradient(90deg, #66f796 0%, #a4e6ff 100%)'
                    : 'linear-gradient(90deg, #f59e0b 0%, #ef4444 100%)',
              }}
            />
          </div>
          <div className="flex items-center justify-between text-[10px] text-[#64748b] font-mono">
            <span>{stat.games} GP</span>
            <span>
              <span className="text-[#ef4444]">{stat.scored} SCORED</span>
              {' / '}
              <span className="text-[#66f796]">{stat.scoreless} NRFI</span>
            </span>
            <span>{stat.avg_runs.toFixed(2)} AVG</span>
          </div>
        </div>
      </div>
    </div>
  );
}

export function NRFI() {
  const { data, isLoading } = useQuery({
    queryKey: ['mlb-first-inning-stats'],
    queryFn: () => mlbApi.getFirstInningStats(),
  });

  const stats = data || [];

  // Summary calculations
  const totalGames = Math.round(stats.reduce((s, t) => s + t.games, 0) / 2);
  const totalTeamGames = stats.reduce((s, t) => s + t.games, 0);
  const totalScoreless = stats.reduce((s, t) => s + t.scoreless, 0);
  const leagueNrfiRate = totalTeamGames > 0 ? (totalScoreless / totalTeamGames) * 100 : 0;

  const sortedAsc = [...stats].sort((a, b) => a.score_pct - b.score_pct);
  const sortedDesc = [...stats].sort((a, b) => b.score_pct - a.score_pct);

  const topNrfi = sortedAsc[0];
  const worstNrfi = sortedDesc[0];

  return (
    <div className="max-w-6xl mx-auto px-4 py-6">
      {/* Header */}
      <div className="mb-6">
        <div className="flex items-center gap-3 mb-1">
          <h1 className="text-3xl font-bold text-[#f1f5f9] font-display tracking-tight">
            NRFI <span className="text-[#a4e6ff]">LEADERBOARD</span>
          </h1>
          <div className="flex items-center gap-1.5">
            <div className="w-1.5 h-1.5 rounded-full bg-[#66f796] animate-pulse" />
            <span className="text-[10px] text-[#64748b] font-bold uppercase tracking-widest font-mono">Season</span>
          </div>
        </div>
        <p className="text-[#64748b] text-sm font-mono">
          Team first inning scoring rates
        </p>
      </div>

      {isLoading ? (
        <div className="flex justify-center py-16">
          <LoadingSpinner />
        </div>
      ) : (
        <>
          {/* Summary stats row */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <StatCard
              label="Games Tracked"
              value={totalGames.toString()}
              sub="Completed games"
            />
            <StatCard
              label="League NRFI Rate"
              value={`${leagueNrfiRate.toFixed(1)}%`}
              sub="Scoreless first innings"
              accent="text-[#a4e6ff]"
            />
            <StatCard
              label="Best NRFI"
              value={topNrfi ? `${(topNrfi.score_pct * 100).toFixed(1)}%` : '-'}
              sub={topNrfi ? getTeamInfo(topNrfi.team).name : ''}
              accent="text-[#66f796]"
            />
            <StatCard
              label="Worst NRFI"
              value={worstNrfi ? `${(worstNrfi.score_pct * 100).toFixed(1)}%` : '-'}
              sub={worstNrfi ? getTeamInfo(worstNrfi.team).name : ''}
              accent="text-[#ef4444]"
            />
          </div>

          {/* Two-column leaderboard */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Best NRFI */}
            <div className="bg-[#191c22] rounded-xl border border-[#1e293b] overflow-hidden">
              <div className="bg-[#66f796]/5 border-b border-[#1e293b] px-4 py-3">
                <h3 className="text-[10px] text-[#66f796] font-bold uppercase tracking-widest">
                  Best NRFI (Hardest to Score On)
                </h3>
              </div>
              <div>
                {sortedAsc.map((stat) => (
                  <TeamRow key={stat.team} stat={stat} mode="nrfi" />
                ))}
              </div>
            </div>

            {/* Worst NRFI */}
            <div className="bg-[#191c22] rounded-xl border border-[#1e293b] overflow-hidden">
              <div className="bg-[#ef4444]/5 border-b border-[#1e293b] px-4 py-3">
                <h3 className="text-[10px] text-[#ef4444] font-bold uppercase tracking-widest">
                  Worst NRFI (Always Scoring)
                </h3>
              </div>
              <div>
                {sortedDesc.map((stat) => (
                  <TeamRow key={stat.team} stat={stat} mode="yrfi" />
                ))}
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
