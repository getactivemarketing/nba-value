// frontend/src/pages/EdgeFinder.tsx
import { useMemo, useState } from 'react';
import { useEdgeEv, useEdgeGames } from '@/hooks/useEdge';
import { EvTable } from '@/components/edge/EvTable';
import { EdgeStatusBanner } from '@/components/edge/EdgeStatusBanner';
import { EdgeFooter } from '@/components/edge/EdgeFooter';
import { BOOK_KEYS, DFS_BOOKS, MARKET_KEYS, bookLabel, marketLabel } from '@/lib/edgeFormat';

const MY_BOOKS_KEY = 'edge.myBooks';
function loadMyBooks(): string[] {
  try { return JSON.parse(localStorage.getItem(MY_BOOKS_KEY) || '[]'); } catch { return []; }
}
function saveMyBooks(v: string[]) {
  try { localStorage.setItem(MY_BOOKS_KEY, JSON.stringify(v)); } catch { /* private mode */ }
}

export function EdgeFinder({ limit }: { limit?: number }) {
  const [market, setMarket] = useState('');
  const [minEv, setMinEv] = useState(0.02);
  const [eventId, setEventId] = useState('');
  const [myBooks, setMyBooks] = useState<string[]>(loadMyBooks);
  const { data: games = [] } = useEdgeGames();
  const { data: rows = [], isLoading } = useEdgeEv({
    market: market || undefined, min_ev: minEv, event_id: eventId || undefined });

  const shown = useMemo(() => {
    const r = myBooks.length ? rows.filter((x) => myBooks.includes(x.book)) : rows;
    return limit ? r.slice(0, limit) : r;
  }, [rows, myBooks, limit]);
  const playerIds = useMemo(() => Object.fromEntries(
    rows.filter((r) => r.bdl_player_id).map((r) => [r.player, r.bdl_player_id as number])), [rows]);

  const toggleBook = (b: string) => {
    const next = myBooks.includes(b) ? myBooks.filter((x) => x !== b) : [...myBooks, b];
    setMyBooks(next); saveMyBooks(next);
  };

  return (
    <div>
      <EdgeStatusBanner />
      {!limit && (
        <>
          <h1 className="text-3xl font-bold text-[#f1f5f9] font-display tracking-tight">+EV Finder</h1>
          <p className="text-sm text-[#64748b] mt-1 font-mono">
            Prices that beat the devigged sharp consensus. Updated every 30 seconds.
          </p>
          <div className="mt-4 flex flex-wrap gap-2 text-xs font-mono">
            <select value={market} onChange={(e) => setMarket(e.target.value)}
                    className="bg-[#191c22] border border-[#1e293b] rounded px-2 py-1 text-[#f1f5f9]">
              <option value="">All markets</option>
              {MARKET_KEYS.map((m) => <option key={m} value={m}>{marketLabel(m)}</option>)}
            </select>
            <select value={eventId} onChange={(e) => setEventId(e.target.value)}
                    className="bg-[#191c22] border border-[#1e293b] rounded px-2 py-1 text-[#f1f5f9]">
              <option value="">All games</option>
              {games.map((g) => <option key={g.event_id} value={g.event_id}>{g.away_team} @ {g.home_team}</option>)}
            </select>
            <select value={minEv} onChange={(e) => setMinEv(Number(e.target.value))}
                    className="bg-[#191c22] border border-[#1e293b] rounded px-2 py-1 text-[#f1f5f9]">
              {[0.02, 0.03, 0.05, 0.08].map((v) => <option key={v} value={v}>EV ≥ {v * 100}%</option>)}
            </select>
          </div>
          <div className="mt-2 flex flex-wrap gap-1">
            {BOOK_KEYS.filter((b) => !DFS_BOOKS.has(b)).map((b) => (
              <button key={b} onClick={() => toggleBook(b)}
                      className={`text-[10px] font-mono rounded px-1.5 py-0.5 border ${myBooks.includes(b)
                        ? 'border-[#a4e6ff] text-[#a4e6ff]' : 'border-[#1e293b] text-[#64748b]'}`}>
                {bookLabel(b)}
              </button>
            ))}
          </div>
        </>
      )}
      <div className="mt-4 rounded-xl bg-[#191c22] border border-[#1e293b] p-4">
        {isLoading ? <div className="text-center py-8 text-[#64748b] font-mono text-sm">Loading…</div>
          : <EvTable rows={shown} playerIds={playerIds} />}
      </div>
      {!limit && <EdgeFooter />}
    </div>
  );
}
