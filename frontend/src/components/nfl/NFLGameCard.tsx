import { useState } from 'react';
import type { NFLGameSummary } from '@/lib/nflApi';
import { getTeamInfo } from '@/lib/nflApi';
import { getTeamLogo } from '@/lib/nflLogos';

interface NFLGameCardProps {
  game: NFLGameSummary;
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

export function NFLGameCard({ game }: NFLGameCardProps) {
  const kickoff = game.kickoff_utc ? new Date(game.kickoff_utc) : null;
  const timeDisplay = kickoff
    ? `${kickoff.toLocaleDateString('en-US', { weekday: 'short' })} ${kickoff.toLocaleTimeString('en-US', {
        hour: 'numeric',
        minute: '2-digit',
      })}`
    : 'TBD';

  const awayTeam = getTeamInfo(game.away_team);
  const homeTeam = getTeamInfo(game.home_team);

  const valueScore = game.best_bet_value_score;
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

        {/* Best-bet row (highlighted, totals-forward) */}
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
      </div>

      {/* SHADOW strip: spread + ML tracked, not bet */}
      <div className="flex items-center justify-between bg-[#0b0e14] border-t border-[#1e293b] px-5 py-2.5">
        <span className="text-[9px] font-bold uppercase tracking-widest text-slate-600 bg-[#32353c]/50 px-2 py-0.5 rounded">
          Shadow
        </span>
        <span className="text-[10px] text-slate-600 font-mono">Spread &amp; ML tracked, not bet</span>
      </div>
    </div>
  );
}
