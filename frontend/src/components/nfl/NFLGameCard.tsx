import { useState } from 'react';
import type { NFLGameSummary } from '@/lib/nflApi';
import { getTeamInfo } from '@/lib/nflApi';
import { getTeamLogo } from '@/lib/nflLogos';

interface NFLGameCardProps {
  game: NFLGameSummary;
  /** No NFL market is live: show the model's lean as a tracked measurement, never as a pick. */
  trackingOnly?: boolean;
}

function formatSpread(line: number) {
  return line > 0 ? `+${line}` : line === 0 ? 'PK' : `${line}`;
}

function LeanRow({ label, lean, detail }: { label: string; lean: string | null; detail?: string | null }) {
  return (
    <div className="flex items-baseline justify-between gap-3">
      <span className="text-[10px] text-slate-500 uppercase font-bold tracking-widest w-14 flex-shrink-0">{label}</span>
      <span className="flex-1 text-sm font-semibold font-mono text-slate-300">
        {lean ?? <span className="text-slate-600 font-normal">No lean</span>}
      </span>
      {detail && <span className="text-[11px] text-slate-500 font-mono text-right">{detail}</span>}
    </div>
  );
}

function NFLLogoCircle({ abbr, size = 32 }: { abbr: string; size?: number }) {
  const [imgError, setImgError] = useState(false);
  const teamInfo = getTeamInfo(abbr);
  const logoUrl = getTeamLogo(abbr);

  if (imgError || !logoUrl) {
    return (
      <span
        className="rounded-full flex items-center justify-center text-[10px] font-bold text-white flex-shrink-0"
        style={{
          width: size,
          height: size,
          background: `linear-gradient(135deg, ${teamInfo.primary} 0 50%, ${teamInfo.secondary} 50% 100%)`,
        }}
      >
        {abbr}
      </span>
    );
  }

  return (
    <div
      className="rounded-full bg-white/5 flex items-center justify-center flex-shrink-0 overflow-hidden p-0.5"
      style={{ width: size, height: size }}
    >
      <img
        src={logoUrl}
        alt={abbr}
        className="w-full h-full object-contain"
        onError={() => setImgError(true)}
      />
    </div>
  );
}

function getValueTier(score: number) {
  if (score >= 70) {
    return {
      text: 'text-[#66f796]',
      bg: 'bg-[#66f796]/10 border-[#66f796]/30',
      label: 'STRONG',
    };
  }
  if (score >= 60) {
    return {
      text: 'text-[#a4e6ff]',
      bg: 'bg-[#a4e6ff]/10 border-[#a4e6ff]/30',
      label: 'MODERATE',
    };
  }
  return {
    text: 'text-slate-400',
    bg: 'bg-[#32353c]/50 border-[#32353c]',
    label: 'LOW',
  };
}

export function NFLGameCard({ game, trackingOnly = false }: NFLGameCardProps) {
  const kickoff = game.kickoff_utc ? new Date(game.kickoff_utc) : null;
  const timeDisplay = kickoff
    ? `${kickoff.toLocaleDateString('en-US', { weekday: 'short' })} ${kickoff.toLocaleTimeString('en-US', {
        hour: 'numeric',
        minute: '2-digit',
      })}`
    : 'TBD';

  const awayTeam = getTeamInfo(game.away_team);
  const homeTeam = getTeamInfo(game.home_team);

  // While tracking, a value score would read as pick strength, so none is shown.
  const valueScore = trackingOnly ? null : game.best_bet_value_score;
  const hasHighValue = valueScore != null && valueScore >= 65;
  const tier = valueScore != null ? getValueTier(valueScore) : null;

  const hasBestBet = game.best_bet_type === 'total' && valueScore != null;
  // Totals never set best_bet_team (the scorer only sets `team` for spread/ML), so the
  // real over/under direction comes from the snapshot's best_total_direction column.
  // Never guess a specific side when it's missing — fall back to a neutral O/U label.
  const rawDirection = game.best_total_direction?.toLowerCase();
  const direction = rawDirection === 'over' ? 'OVER' : rawDirection === 'under' ? 'UNDER' : null;
  const bestBetLabel = direction
    ? `${direction} ${game.best_bet_line ?? '-'}`
    : `O/U ${game.best_bet_line ?? '-'}`;

  // Tracked leans. predicted_margin is home minus away.
  const snapshotted = game.predicted_total != null;
  const totalLean = direction && game.best_total_line != null ? `${direction} ${game.best_total_line}` : null;
  const projTotal = game.predicted_total != null ? `proj ${game.predicted_total.toFixed(1)}` : null;
  const spreadLean = game.spread_lean_team && game.spread_lean_line != null
    ? `${game.spread_lean_team} ${formatSpread(game.spread_lean_line)}`
    : null;
  const margin = game.predicted_margin;
  const projMargin = margin == null
    ? null
    : Math.abs(margin) < 0.5
      ? 'proj even'
      : `proj ${margin > 0 ? game.home_team : game.away_team} by ${Math.abs(margin).toFixed(1)}`;

  return (
    <div className="rounded-xl bg-[#191c22] border border-[#1e293b] hover:border-[#a4e6ff]/30 relative overflow-hidden transition-colors">
      {/* Left edge glow for high-value games */}
      {hasHighValue && (
        <div className="absolute left-0 top-0 bottom-0 w-1 bg-[#a4e6ff] shadow-[0_0_8px_rgba(164,230,255,0.4)]" />
      )}

      <div className="p-5 pb-3">
        {/* Header: matchup label + kickoff + pills */}
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <span className="text-[10px] text-slate-500 uppercase font-bold tracking-widest">
              {game.away_team} @ {game.home_team}
            </span>
            {game.week != null && (
              <span className="text-[10px] bg-[#32353c] px-2 py-0.5 rounded text-slate-400 font-mono">
                WK {game.week}
              </span>
            )}
            {game.is_primetime && (
              <span className="text-[10px] bg-[#f59e0b]/10 text-[#f59e0b] border border-[#f59e0b]/30 px-2 py-0.5 rounded font-bold font-mono uppercase tracking-widest">
                PRIME
              </span>
            )}
            {game.is_divisional && (
              <span className="text-[10px] bg-[#32353c] px-2 py-0.5 rounded text-slate-400 font-bold font-mono uppercase tracking-widest">
                DIV
              </span>
            )}
          </div>
          <span className="text-[10px] text-[#a4e6ff] font-bold font-mono uppercase tracking-widest">
            {timeDisplay}
          </span>
        </div>

        {/* Teams + value badge */}
        <div className="flex justify-between items-start mb-5">
          <div className="flex flex-col gap-1.5">
            <div className="flex items-center gap-2.5">
              <NFLLogoCircle abbr={game.away_team} />
              <span className="text-base font-bold text-txt-primary">{awayTeam.name}</span>
            </div>
            <div className="flex items-center gap-2.5">
              <NFLLogoCircle abbr={game.home_team} />
              <span className="text-base font-bold text-txt-primary">{homeTeam.name}</span>
            </div>
          </div>

          {valueScore != null && tier && (
            <div className="flex flex-col items-end flex-shrink-0">
              <span className="text-[10px] text-slate-500 uppercase font-bold tracking-widest mb-1">
                Value Score
              </span>
              <div className={`${tier.bg} border px-3 py-1 rounded-full flex items-center gap-2`}>
                <span className={`${tier.text} font-black font-mono text-lg`}>{valueScore.toFixed(0)}</span>
                <span className={`text-[10px] ${tier.text} font-bold tracking-widest`}>{tier.label}</span>
              </div>
            </div>
          )}
        </div>

        {trackingOnly ? (
          <div className="mt-4 pt-4 border-t border-slate-700/30">
            <div className="flex items-center justify-between mb-2.5">
              <span className="text-[10px] text-slate-500 uppercase font-bold tracking-widest">Model Lean</span>
            </div>
            {snapshotted ? (
              <div className="flex flex-col gap-2">
                <LeanRow label="Total" lean={totalLean} detail={projTotal} />
                <LeanRow label="Spread" lean={spreadLean} detail={projMargin} />
              </div>
            ) : (
              <span className="text-sm text-slate-500 font-mono">Lean posts ~90 min before kickoff</span>
            )}
          </div>
        ) : (
        <div className="mt-4 pt-4 border-t border-slate-700/30">
          <div className="flex items-center justify-between mb-2">
            <span className="text-[10px] text-slate-500 uppercase font-bold tracking-widest">Best Bet</span>
          </div>
          {hasBestBet && tier ? (
            <div className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-lg border ${tier.bg}`}>
              <span className={`font-semibold text-sm ${tier.text}`}>
                {bestBetLabel}
              </span>
              <span className={`font-bold font-mono text-sm ${tier.text}`}>{valueScore!.toFixed(0)}</span>
            </div>
          ) : (
            <span className="text-sm text-slate-500 font-mono">No value pick</span>
          )}
        </div>
        )}
      </div>

      {/* SHADOW strip: what is tracked but not bet */}
      <div className="flex items-center justify-between bg-[#0b0e14] border-t border-[#1e293b] px-5 py-2.5">
        <span className="text-[9px] font-bold uppercase tracking-widest text-slate-600 bg-[#32353c]/50 px-2 py-0.5 rounded">
          {trackingOnly ? 'Tracking' : 'Shadow'}
        </span>
        <span className="text-[10px] text-slate-600 font-mono">
          {trackingOnly ? 'Measured vs the closing line · not a bet' : 'Spread & ML tracked, not bet'}
        </span>
      </div>
    </div>
  );
}
