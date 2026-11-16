// frontend/src/pages/BlindPlay.tsx
import { Link, useParams } from 'react-router-dom';
import { useBlindPlay, useEdgeEv } from '@/hooks/useEdge';
import { EdgeFooter } from '@/components/edge/EdgeFooter';
import { EmailSignup } from '@/components/edge/EmailSignup';
import { bookLabel, edgeBase, fmtAmerican, fmtEv, type Sport } from '@/lib/edgeFormat';
import type { BlindPanel, BlindPlayData, BlindResult } from '@/types/edge';

const ordinal = (n: number) => {
  const r = n % 100;
  const suffix = r >= 11 && r <= 13 ? 'th' : ({ 1: 'st', 2: 'nd', 3: 'rd' } as Record<number, string>)[n % 10] ?? 'th';
  return `${n}${suffix}`;
};

// eslint-disable-next-line react-refresh/only-export-components
export const etDateTime = (iso: string) =>
  `${new Date(iso).toLocaleString('en-US', { timeZone: 'America/New_York', weekday: 'short', month: 'short',
    day: 'numeric', hour: 'numeric', minute: '2-digit' })} ET`;

const RESULT_STYLE: Record<BlindResult, string> = {
  win: 'text-[#4ade80] border-[#4ade80]/40', loss: 'text-[#f87171] border-[#f87171]/40',
  push: 'text-[#94a3b8] border-[#94a3b8]/40', void: 'text-[#94a3b8] border-[#94a3b8]/40',
};

export function ResultBadge({ result, units }: { result: BlindResult | null; units: number | null }) {
  if (!result) return <span className="text-[10px] font-mono uppercase text-[#64748b]">Pending</span>;
  const u = units == null || result === 'void' ? '' : ` ${units >= 0 ? '+' : ''}${units.toFixed(2)}u`;
  return (
    <span className={`text-xs font-mono font-bold uppercase border rounded px-2 py-0.5 ${RESULT_STYLE[result]}`}>
      {result}{u}
    </span>
  );
}

function Logo({ url, name }: { url: string | null; name: string | null }) {
  const label = name ?? '?';
  return url
    ? <img src={url} alt={label} className="w-12 h-12 object-contain" />
    : <div className="w-12 h-12 grid place-items-center rounded bg-[#191c22] border border-[#1e293b] text-[10px] font-mono font-bold text-[#94a3b8] text-center leading-tight p-1">{label.split(' ').pop()}</div>;
}

function Silhouette() {
  return (
    <svg viewBox="0 0 24 24" role="img" aria-label="No photo" className="w-full h-full text-[#334155]" fill="currentColor">
      <circle cx="12" cy="8" r="4" /><path d="M4 21c0-4.4 3.6-7 8-7s8 2.6 8 7z" />
    </svg>
  );
}

function panelStats(sport: Sport, p: BlindPanel | null): [string, string][] {
  if (!p) return [];
  const out: [string, string | null][] = sport === 'nfl'
    ? [['Opponent vs position', p.opp_rank != null ? `${ordinal(p.opp_rank)} most yards allowed` : null],
       ['Share, last 4 games', p.share_l4 != null ? `${Math.round(p.share_l4 * 100)}%` : null],
       ['Snap share', p.snap_share != null ? `${Math.round(p.snap_share * 100)}%` : null]]
    : [['Opponent vs position', p.opp_vs_pos_rank != null ? `${ordinal(p.opp_vs_pos_rank)} most allowed` : null],
       ['Opponent defense', p.drtg_rank != null ? `${ordinal(p.drtg_rank)} defensive rating` : null],
       ['Opponent pace', p.pace_rank != null ? `${ordinal(p.pace_rank)} fastest` : null],
       ['Usage, last 10', p.usage_l10 != null ? `${p.usage_l10.toFixed(1)}%` : null],
       ['Minutes, last 5', p.minutes_l5 != null ? p.minutes_l5.toFixed(1) : null]];
  return out.filter((r): r is [string, string] => r[1] != null);
}

function Stat({ k, v }: { k: string; v: string }) {
  return (
    <div className="bg-[#0b0e14] rounded-lg p-3 border border-[#1e293b]">
      <div className="text-[10px] text-[#64748b] uppercase font-bold tracking-widest">{k}</div>
      <div className="text-base font-black font-mono text-[#f1f5f9]">{v}</div>
    </div>
  );
}

export function NotFound() {
  return (
    <div className="text-center py-12 font-mono">
      <h1 className="text-xl font-black text-[#f1f5f9]">We couldn't find that play</h1>
      <p className="mt-2 text-sm text-[#94a3b8]">The link may be old or mistyped.</p>
      <Link to="/blind" className="inline-block mt-4 text-sm text-[#a4e6ff] hover:underline">See every blind play</Link>
    </div>
  );
}

function Reveal({ play, sport }: { play: BlindPlayData; sport: Sport }) {
  const { data: evRows } = useEdgeEv({ event_id: play.event_id }, sport);
  const bdl = evRows?.find((r) => r.player === play.player && r.bdl_player_id)?.bdl_player_id;
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
            {play.headshot_url
              ? <img src={play.headshot_url} alt={play.player} className="w-full h-full object-cover" />
              : <Silhouette />}
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
            {stats.map(([k, v]) => <Stat key={k} k={k} v={v} />)}
          </div>
        </section>
      )}
      <div className="mt-6 flex flex-wrap gap-3 text-sm font-mono">
        <Link to={`${edgeBase(sport)}/odds/${play.event_id}?market=${play.market}`}
              className="text-[#a4e6ff] hover:underline">Compare every book on the odds screen</Link>
        {bdl != null && (
          <Link to={`${edgeBase(sport)}/player/${bdl}?market=${play.market}`}
                className="text-[#a4e6ff] hover:underline">Game log and trends for {play.player}</Link>
        )}
      </div>
      <EmailSignup source="blind_reveal" />
      <EdgeFooter />
    </div>
  );
}

export function BlindPlay() {
  const { sport = '', n = '' } = useParams();
  const valid = (sport === 'nba' || sport === 'nfl') && /^\d+$/.test(n);
  const { data, isLoading, isError } = useBlindPlay(valid ? (sport as Sport) : 'nba', valid ? Number(n) : 0);
  if (!valid || isError) return <NotFound />;
  if (isLoading || !data) return <div className="text-center py-8 text-[#64748b] font-mono text-sm">Loading…</div>;
  return <Reveal play={data} sport={sport as Sport} />;
}
