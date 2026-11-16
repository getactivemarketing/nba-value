// frontend/src/pages/BlindList.tsx
import { useState } from 'react';
import { Link } from 'react-router-dom';
import { useBlindList } from '@/hooks/useEdge';
import { EdgeFooter } from '@/components/edge/EdgeFooter';
import { EmailSignup } from '@/components/edge/EmailSignup';
import type { Sport } from '@/lib/edgeFormat';
import { ResultBadge, etDateTime } from './BlindPlay';

export function BlindList() {
  const [sport, setSport] = useState<Sport>('nba');
  const { data, isLoading, isError } = useBlindList(sport);
  const r = data?.record;
  return (
    <div>
      <h1 className="text-3xl font-bold text-[#f1f5f9] font-display tracking-tight">Blind plays</h1>
      <p className="text-sm text-[#64748b] font-mono max-w-2xl mt-2 leading-relaxed">
        Every blind play we post on X, revealed. Wins, losses, pushes and voids all stay on this page.
      </p>
      <div className="mt-3 flex gap-1">
        {(['nba', 'nfl'] as const).map((s) => (
          <button key={s} onClick={() => setSport(s)}
                  className={`text-[10px] font-mono uppercase rounded px-2 py-1 border ${s === sport
                    ? 'border-[#a4e6ff] text-[#a4e6ff]' : 'border-[#1e293b] text-[#64748b]'}`}>{s}</button>
        ))}
      </div>
      {isLoading && <div className="text-center py-8 text-[#64748b] font-mono text-sm">Loading…</div>}
      {isError && <div role="alert" className="py-8 text-center text-[#f87171] font-mono text-sm">Couldn't load plays. Try again shortly.</div>}
      {r && (
        <div className="mt-4 bg-[#0b0e14] rounded-lg p-4 border border-[#1e293b]">
          <div className="text-[10px] text-[#64748b] uppercase font-bold tracking-widest">Record (W-L-P)</div>
          <div className="text-lg font-black font-mono text-[#f1f5f9]">
            {r.wins}-{r.losses}-{r.pushes}
            <span className="ml-3 text-xs font-bold text-[#94a3b8]">
              {r.voids} void · {r.units >= 0 ? '+' : ''}{r.units.toFixed(2)} units · {r.n} graded
            </span>
          </div>
        </div>
      )}
      {data && data.plays.length === 0 && (
        <p className="mt-6 text-sm font-mono text-[#94a3b8]">No blind plays yet.</p>
      )}
      <ul className="mt-4 space-y-2">
        {data?.plays.map((p) => (
          <li key={p.n}>
            <Link to={`/blind/${p.sport}/${p.n}`}
                  className="flex items-center justify-between gap-3 bg-[#0b0e14] rounded-lg p-4 border border-[#1e293b] hover:border-[#a4e6ff]">
              <div className="min-w-0">
                <div className="text-sm font-black font-mono text-[#f1f5f9] truncate">{p.player}</div>
                <div className="text-xs font-mono text-[#94a3b8]">{p.bet}</div>
                <div className="text-[10px] font-mono text-[#64748b]">{etDateTime(p.commence_time)}</div>
              </div>
              <ResultBadge result={p.result} units={p.profit_units} />
            </Link>
          </li>
        ))}
      </ul>
      <EmailSignup source="blind_reveal" />
      <EdgeFooter />
    </div>
  );
}
