// frontend/src/components/edge/EvTable.tsx
import { Link } from 'react-router-dom';
import type { EvRow } from '@/types/edge';
import { bookLabel, edgeBase, fmtAge, fmtAmerican, fmtEv, type Sport } from '@/lib/edgeFormat';

export function EvTable({ rows, playerIds = {}, sport = 'nba' }:
  { rows: EvRow[]; playerIds?: Record<string, number>; sport?: Sport }) {
  if (!rows.length) {
    return <div className="text-center py-8 text-[#64748b] font-mono text-sm">No +EV prices right now.</div>;
  }
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="text-left text-[10px] text-[#64748b] uppercase font-bold tracking-widest border-b border-[#1e293b]">
            <th className="pb-2">EV</th><th className="pb-2">Bet</th><th className="pb-2">Book</th>
            <th className="pb-2 text-center">Price</th><th className="pb-2 text-center">Fair</th>
            <th className="pb-2 text-center">Books</th><th className="pb-2 text-center">Age</th><th className="pb-2" />
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={`${r.event_id}-${r.market}-${r.player}-${r.side}-${r.line}-${r.book}`}
                className="border-b border-[#1e293b]/50">
              <td className={`py-2 font-mono font-bold ${r.verify ? 'text-[#fbbf24]' : 'text-[#a4e6ff]'}`}>
                {fmtEv(r.ev)}{r.verify && <span className="ml-1 text-[9px] uppercase">verify</span>}
              </td>
              <td className="py-2">
                <div className="text-[#f1f5f9]">
                  {sport === 'nba' && r.player && playerIds[r.player]
                    ? <Link to={`/edge/player/${playerIds[r.player]}?market=${r.market}`} className="hover:underline">{r.bet}</Link>
                    : r.bet}
                </div>
                <Link to={`${edgeBase(sport)}/odds/${r.event_id}?market=${r.market}`} className="text-[10px] text-[#64748b] font-mono hover:underline">
                  {r.matchup}
                </Link>
              </td>
              <td className="py-2 text-[#94a3b8] font-mono text-xs">{bookLabel(r.book)}</td>
              <td className="py-2 text-center font-mono text-[#f1f5f9]">{fmtAmerican(r.price)}</td>
              <td className="py-2 text-center font-mono text-[#94a3b8] text-xs">{fmtAmerican(r.fair_price)}</td>
              <td className="py-2 text-center font-mono text-[#94a3b8] text-xs">{r.n_books}</td>
              <td className="py-2 text-center font-mono text-[#64748b] text-xs">{fmtAge(r.observed_at)}</td>
              <td className="py-2 text-right">
                {r.link && (
                  <a href={r.link} target="_blank" rel="noopener noreferrer sponsored"
                     className="rounded bg-[#a4e6ff] px-2 py-1 text-[10px] font-bold uppercase text-[#0b0e14]">Bet</a>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
