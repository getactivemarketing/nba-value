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

type Metric = 'opp_score_pct' | 'score_pct';
type Tone = 'good' | 'bad';

function TeamRow({
  stat,
  metric,
  tone,
}: {
  stat: FirstInningStats;
  metric: Metric;
  tone: Tone;
}) {
  // Defensive null-safety — backend may not have deployed the new fields yet
  const scorePct = stat.score_pct ?? 0;
  const oppScorePct = stat.opp_score_pct ?? 0;
  const avgRuns = stat.avg_runs ?? 0;
  const avgRunsAllowed = stat.avg_runs_allowed ?? 0;

  const rawPct = (metric === 'opp_score_pct' ? oppScorePct : scorePct) * 100;
  const teamInfo = getTeamInfo(stat.team);
  const barPct = Math.min(100, Math.max(0, rawPct));

  const pctColor = tone === 'good' ? 'text-[#66f796]' : 'text-[#ef4444]';
  const gradient =
    tone === 'good'
      ? 'linear-gradient(90deg, #66f796 0%, #a4e6ff 100%)'
      : 'linear-gradient(90deg, #f59e0b 0%, #ef4444 100%)';

  const subStat =
    metric === 'opp_score_pct'
      ? `${avgRunsAllowed.toFixed(2)} R/1st allowed`
      : `${avgRuns.toFixed(2)} R/1st scored`;

  return (
    <div className="px-4 py-3 border-b border-[#1e293b] last:border-b-0 hover:bg-[#0b0e14] transition-colors">
      <div className="flex items-center gap-3">
        <MLBLogoCircle team={stat.team} size={32} />
        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between mb-1.5">
            <span className="text-sm font-bold text-[#f1f5f9]">{teamInfo.name}</span>
            <span className={`text-sm font-mono font-black ${pctColor}`}>
              {rawPct.toFixed(1)}%
            </span>
          </div>
          <div className="h-1.5 bg-[#0b0e14] rounded-full overflow-hidden mb-1.5">
            <div
              className="h-full rounded-full"
              style={{ width: `${barPct}%`, background: gradient }}
            />
          </div>
          <div className="flex items-center justify-between text-[10px] text-[#64748b] font-mono">
            <span>{stat.games} GP</span>
            <span>{subStat}</span>
          </div>
        </div>
      </div>
    </div>
  );
}

function Section({
  title,
  accent,
  stats,
  metric,
  tone,
}: {
  title: string;
  accent: string;
  stats: FirstInningStats[];
  metric: Metric;
  tone: Tone;
}) {
  return (
    <div className="bg-[#191c22] rounded-xl border border-[#1e293b] overflow-hidden">
      <div className={`border-b border-[#1e293b] px-4 py-3`} style={{ backgroundColor: `${accent}0D` }}>
        <h3 className="text-[10px] font-bold uppercase tracking-widest" style={{ color: accent }}>
          {title}
        </h3>
      </div>
      <div>
        {stats.map((stat) => (
          <TeamRow key={stat.team} stat={stat} metric={metric} tone={tone} />
        ))}
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
  const totalGames = Math.round(stats.reduce((s, t) => s + (t.games ?? 0), 0) / 2);
  const totalTeamGames = stats.reduce((s, t) => s + (t.games ?? 0), 0);
  const totalScoreless = stats.reduce((s, t) => s + (t.scoreless ?? 0), 0);
  const leagueNrfiRate = totalTeamGames > 0 ? (totalScoreless / totalTeamGames) * 100 : 0;

  // Sorts — use nullish coalescing in case backend hasn't deployed new fields yet
  const bestPitching = [...stats].sort((a, b) => (a.opp_score_pct ?? 0) - (b.opp_score_pct ?? 0));
  const worstPitching = [...stats].sort((a, b) => (b.opp_score_pct ?? 0) - (a.opp_score_pct ?? 0));
  const coldestBats = [...stats].sort((a, b) => (a.score_pct ?? 0) - (b.score_pct ?? 0));
  const hottestBats = [...stats].sort((a, b) => (b.score_pct ?? 0) - (a.score_pct ?? 0));

  const bestPitchingTeam = bestPitching[0];
  const coldestBatsTeam = coldestBats[0];

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
          Team first inning scoring rates: offense and defense
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
              label="Best Pitching"
              value={bestPitchingTeam ? `${((bestPitchingTeam.opp_score_pct ?? 0) * 100).toFixed(1)}%` : '-'}
              sub={bestPitchingTeam ? `${getTeamInfo(bestPitchingTeam.team).name} opp 1st` : ''}
              accent="text-[#66f796]"
            />
            <StatCard
              label="Coldest Bats"
              value={coldestBatsTeam ? `${((coldestBatsTeam.score_pct ?? 0) * 100).toFixed(1)}%` : '-'}
              sub={coldestBatsTeam ? `${getTeamInfo(coldestBatsTeam.team).name} 1st scoring` : ''}
              accent="text-[#a4e6ff]"
            />
          </div>

          {/* 2x2 leaderboard */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Section
              title="Best NRFI Pitching (Fewest 1st Inning Runs Allowed)"
              accent="#66f796"
              stats={bestPitching}
              metric="opp_score_pct"
              tone="good"
            />
            <Section
              title="Worst NRFI Pitching (Most 1st Inning Runs Allowed)"
              accent="#ef4444"
              stats={worstPitching}
              metric="opp_score_pct"
              tone="bad"
            />
            <Section
              title="Coldest 1st Inning Bats"
              accent="#a4e6ff"
              stats={coldestBats}
              metric="score_pct"
              tone="good"
            />
            <Section
              title="Hottest 1st Inning Bats"
              accent="#f59e0b"
              stats={hottestBats}
              metric="score_pct"
              tone="bad"
            />
          </div>
        </>
      )}
    </div>
  );
}
