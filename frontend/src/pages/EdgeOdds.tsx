// frontend/src/pages/EdgeOdds.tsx
import { useParams, useSearchParams } from 'react-router-dom';
import { useEdgeOdds } from '@/hooks/useEdge';
import { EdgeStatusBanner } from '@/components/edge/EdgeStatusBanner';
import { EdgeFooter } from '@/components/edge/EdgeFooter';
import { BOOK_KEYS, bookLabel, fmtAmerican, marketLabel, MARKET_KEYS, outcomeLabel } from '@/lib/edgeFormat';

export function EdgeOdds() {
  const { eventId = '' } = useParams();
  const [params, setParams] = useSearchParams();
  const market = params.get('market') || 'h2h';
  const { data, isLoading } = useEdgeOdds(eventId, market);
  const books = BOOK_KEYS.filter((b) => data?.rows.some((r) => r.prices[b]));

  return (
    <div>
      <EdgeStatusBanner />
      <h1 className="text-2xl font-black font-mono text-[#f1f5f9]">
        {data?.event ? `${data.event.away_team} @ ${data.event.home_team}` : 'Odds'}
      </h1>
      <div className="mt-3 flex flex-wrap gap-1">
        {MARKET_KEYS.map((m) => (
          <button key={m} onClick={() => setParams({ market: m })}
                  className={`text-[10px] font-mono rounded px-2 py-1 border ${m === market
                    ? 'border-[#a4e6ff] text-[#a4e6ff]' : 'border-[#1e293b] text-[#64748b]'}`}>
            {marketLabel(m)}
          </button>
        ))}
      </div>
      <div className="mt-4 rounded-xl bg-[#191c22] border border-[#1e293b] p-4 overflow-x-auto">
        {isLoading ? <div className="text-center py-8 text-[#64748b] font-mono text-sm">Loading…</div> : (
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-[10px] text-[#64748b] uppercase font-bold tracking-widest border-b border-[#1e293b]">
                <th className="pb-2">Outcome</th><th className="pb-2 text-center">Fair</th>
                {books.map((b) => <th key={b} className="pb-2 text-center whitespace-nowrap">{bookLabel(b)}</th>)}
              </tr>
            </thead>
            <tbody>
              {data?.rows.map((r) => (
                <tr key={`${r.player}-${r.line}-${r.side}`} className="border-b border-[#1e293b]/50">
                  <td className="py-2 text-[#f1f5f9] whitespace-nowrap">
                    {r.player && <span className="mr-1">{r.player}</span>}
                    <span className={r.player ? 'text-[#94a3b8] font-mono text-xs' : ''}>
                      {data.event ? outcomeLabel(market, r.side, r.line, data.event) : r.side}
                    </span>
                  </td>
                  <td className="py-2 text-center font-mono text-xs text-[#94a3b8]">{fmtAmerican(r.fair_price)}</td>
                  {books.map((b) => {
                    const c = r.prices[b];
                    return (
                      <td key={b} className={`py-2 text-center font-mono text-xs whitespace-nowrap ${
                        !c ? 'text-[#475569]' : c.stale ? 'text-[#475569] line-through'
                          : b === r.best_book ? 'text-[#a4e6ff] font-bold' : 'text-[#f1f5f9]'}`}>
                        {c ? (c.link && !c.stale
                          ? <a href={c.link} target="_blank" rel="noopener noreferrer sponsored">{fmtAmerican(c.price)}</a>
                          : fmtAmerican(c.price)) : '—'}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
      <p className="text-[10px] text-[#64748b] font-mono mt-1.5 italic">
        PrizePicks and Underdog show the line only; their payouts are fixed, so they are never priced for EV.
      </p>
      <EdgeFooter />
    </div>
  );
}
