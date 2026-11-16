// frontend/src/pages/BlindPlay.tsx
import { Link, useParams } from 'react-router-dom';
import { useState } from 'react';
import { useBlindPlay } from '@/hooks/useEdge';
import { EdgeFooter } from '@/components/edge/EdgeFooter';
import { EmailSignup } from '@/components/edge/EmailSignup';
import { bookLabel, edgeBase, etDateTime, fmtAmerican, fmtEv, ordinal, type Sport } from '@/lib/edgeFormat';
import { NotFound, ResultBadge } from '@/components/edge/blind';
import type { BlindPanel, BlindPlayData } from '@/types/edge';

function Logo({ url, name }: { url: string | null; name: string | null }) {
  const label = name ?? '?';
  return url
    ? <img src={url} alt={label} className="w-12 h-12 object-contain" />
    : <div className="w-12 h-12 grid place-items-center rounded bg-[#191c22] border border-[#1e293b] text-[10px] font-mono font-bold text-[#94a3b8] text-center leading-tight p-1">{label.split(' ').pop()}</div>;
}

function Headshot({ url, name }: { url: string | null; name: string }) {
  const [broken, setBroken] = useState(false);
  return url && !broken
    ? <img src={url} alt={name} onError={() => setBroken(true)} className="w-full h-full object-cover" />
    : <Silhouette />;
}

function Silhouette() {
  return (
    <svg viewBox="0 0 24 24" role="img" aria-label="No photo" className="w-full h-full text-[#334155]" fill="currentColor">
      <circle cx="12" cy="8" r="4" /><path d="M4 21c0-4.4 3.6-7 8-7s8 2.6 8 7z" />
    </svg>
  );
}

type StatRow = [label: string, value: string, note?: string];

// Ranks match the X card. NBA vs-position and defensive-rating ranks run 1..30, so the direction is spelled out.
function panelStats(sport: Sport, p: BlindPanel | null): StatRow[] {
  if (!p) return [];
  const rows: (StatRow | null)[] = sport === 'nfl'
    ? [p.opp_rank != null ? ['Opponent vs position', `${ordinal(p.opp_rank)} most yards allowed`] : null,
       p.share_l4 != null ? ['Share, last 4 games', `${Math.round(p.share_l4 * 100)}%`] : null,
       p.snap_share != null ? ['Snap share', `${Math.round(p.snap_share * 100)}%`] : null]
    : [p.opp_vs_pos_rank != null ? ['Opponent vs position', `${ordinal(p.opp_vs_pos_rank)} of 30`, '30th = allows the most'] : null,
       p.drtg_rank != null ? ['Opponent defensive rating', `${ordinal(p.drtg_rank)} of 30`, '1st = best defense'] : null,
       p.pace_rank != null ? ['Opponent pace', `${ordinal(p.pace_rank)} fastest`] : null,
       p.usage_l10 != null ? ['Usage, last 10', `${p.usage_l10.toFixed(1)}%`] : null,
       p.minutes_l5 != null ? ['Minutes, last 5', p.minutes_l5.toFixed(1)] : null];
  return rows.filter((r): r is StatRow => r != null);
}

function Stat({ k, v, note }: { k: string; v: string; note?: string }) {
  return (
    <div className="bg-[#0b0e14] rounded-lg p-3 border border-[#1e293b]">
      <div className="text-[10px] text-[#64748b] uppercase font-bold tracking-widest">{k}</div>
      <div className="text-base font-black font-mono text-[#f1f5f9]">{v}</div>
      {note && <div className="text-[10px] font-mono text-[#64748b]">{note}</div>}
    </div>
  );
}

function Reveal({ play, sport }: { play: BlindPlayData; sport: Sport }) {
  const stats = panelStats(sport, play.panel);
  const form = play.form;
  return (
    <div>
      <Link to="/blind" className="text-[10px] font-mono uppercase tracking-widest text-[#64748b] hover:text-[#a4e6ff]">
        All blind plays
      </Link>
      <div className="mt-3 bg-[#0b0e14] rounded-xl border border-[#1e293b] p-5">
        <div className="flex items-center gap-4">
          <div className="w-24 h-24 shrink-0 rounded-full overflow-hidden bg-[#191c22] border border-[#1e293b]">
            <Headshot url={play.headshot_url} name={play.player} />
          </div>
          <div className="min-w-0">
            <h1 className="text-2xl font-black text-[#f1f5f9] font-display tracking-tight">{play.player}</h1>
            <p className="text-xs font-mono text-[#94a3b8] mt-1">
              {play.position ? `${play.position} · ` : ''}{play.team ?? 'Team unconfirmed'}
            </p>
          </div>
        </div>
        <div className="mt-4 flex items-center justify-center gap-3">
          <Logo url={play.away_logo_url} name={play.away_team} />
          <span className="text-xs font-mono text-[#64748b]">@</span>
          <Logo url={play.home_logo_url} name={play.home_team} />
        </div>
        <p className="mt-2 text-center text-[11px] font-mono text-[#64748b]">{etDateTime(play.commence_time)}</p>
        <div className="mt-4 flex items-center justify-between gap-2 border-t border-[#1e293b] pt-4">
          <div className="text-lg font-black font-mono text-[#a4e6ff]">{play.bet}</div>
          <ResultBadge result={play.result} units={play.profit_units} />
        </div>
      </div>
      <div className="mt-3 grid grid-cols-2 sm:grid-cols-4 gap-2">
        <Stat k="Best price" v={`${fmtAmerican(play.price)} ${bookLabel(play.book)}`} />
        <Stat k="Edge (EV)" v={fmtEv(play.ev)} />
        <Stat k="Books compared" v={String(play.n_books)} />
        {form?.hits != null && form.games ? <Stat k="Last 10" v={`${form.hits}/${form.games} hit`} /> : null}
      </div>
      <p className="mt-4 text-sm font-mono text-[#cbd5e1] leading-relaxed">{play.explanation}</p>
      {stats.length > 0 && (
        <section className="mt-6" aria-label="Matchup">
          <h2 className="text-[10px] text-[#64748b] uppercase font-bold tracking-widest mb-2">Why the matchup helps</h2>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
            {stats.map(([k, v, note]) => <Stat key={k} k={k} v={v} note={note} />)}
          </div>
        </section>
      )}
      <div className="mt-6 flex flex-wrap gap-3 text-sm font-mono">
        <Link to={`${edgeBase(sport)}/odds/${play.event_id}?market=${play.market}`}
              className="text-[#a4e6ff] hover:underline">Compare every book on the odds screen</Link>
        {play.bdl_player_id != null && (
          <Link to={`${edgeBase(sport)}/player/${play.bdl_player_id}?market=${play.market}`}
                className="text-[#a4e6ff] hover:underline">Game log and trends for {play.player}</Link>
        )}
      </div>
      <EmailSignup source="blind_reveal" />
      <EdgeFooter />
    </div>
  );
}

export function BlindPlay() {
  const { sport: rawSport = '', n = '' } = useParams();
  const sport = rawSport.toLowerCase();
  const valid = (sport === 'nba' || sport === 'nfl') && /^[1-9]\d*$/.test(n);
  const { data, isLoading, isError } = useBlindPlay(valid ? (sport as Sport) : 'nba', valid ? Number(n) : 0);
  if (!valid || isError) return <NotFound />;
  if (isLoading || !data) return <div className="text-center py-8 text-[#64748b] font-mono text-sm">Loading…</div>;
  return <Reveal play={data} sport={sport as Sport} />;
}
