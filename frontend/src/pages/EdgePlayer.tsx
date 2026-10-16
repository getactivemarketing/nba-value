// frontend/src/pages/EdgePlayer.tsx
import { useState } from 'react';
import { useParams, useSearchParams } from 'react-router-dom';
import { Bar, BarChart, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { useEdgePlayer } from '@/hooks/useEdge';
import { EdgeFooter } from '@/components/edge/EdgeFooter';
import { bookLabel, fmtAmerican, fmtPct, marketLabel } from '@/lib/edgeFormat';

const WINDOWS = [['l5', 'Last 5'], ['l10', 'Last 10'], ['l20', 'Last 20'], ['season', 'Season'],
  ['vs_opponent', 'vs Opp'], ['home_away', 'Home/Away']] as const;
const PROP_MARKETS = ['player_points', 'player_rebounds', 'player_assists', 'player_threes',
  'player_points_rebounds_assists', 'player_points_rebounds', 'player_points_assists', 'player_rebounds_assists'];

export function EdgePlayer() {
  const { id = '0' } = useParams();
  const [params, setParams] = useSearchParams();
  const market = params.get('market') || 'player_points';
  const [minMinutes, setMinMinutes] = useState(0);
  const { data, isLoading } = useEdgePlayer(Number(id), market, undefined, minMinutes);
  if (isLoading || !data) return <div className="text-center py-8 text-[#64748b] font-mono text-sm">Loading…</div>;
  const chart = [...data.games].reverse().map((g) => ({ ...g, label: g.game_date.slice(5) }));

  return (
    <div>
      <h1 className="text-3xl font-bold text-[#f1f5f9] font-display tracking-tight">{data.name ?? 'Player'}</h1>
      <p className="text-sm text-[#64748b] mt-1 font-mono">
        {data.team}{data.opponent ? ` · next: ${data.is_home ? 'vs' : '@'} ${data.opponent}` : ''}
        {data.line != null ? ` · line ${data.line}` : ' · no line posted'}
      </p>
      <div className="mt-3 flex flex-wrap gap-1">
        {PROP_MARKETS.map((m) => (
          <button key={m} onClick={() => setParams({ market: m })}
                  className={`text-[10px] font-mono rounded px-2 py-1 border ${m === market
                    ? 'border-[#a4e6ff] text-[#a4e6ff]' : 'border-[#1e293b] text-[#64748b]'}`}>{marketLabel(m)}</button>
        ))}
        <select value={minMinutes} onChange={(e) => setMinMinutes(Number(e.target.value))}
                className="ml-2 bg-[#191c22] border border-[#1e293b] rounded px-2 py-1 text-[10px] font-mono text-[#f1f5f9]">
          {[0, 20, 25, 30].map((v) => <option key={v} value={v}>{v ? `≥ ${v} min` : 'All minutes'}</option>)}
        </select>
      </div>
      {data.hit_rates && (
        <div className="mt-4 grid grid-cols-3 sm:grid-cols-6 gap-2">
          {WINDOWS.map(([k, label]) => (
            <div key={k} className="bg-[#0b0e14] rounded-lg p-3 border border-[#1e293b]">
              <div className="text-[10px] text-[#64748b] uppercase font-bold tracking-widest">{label}</div>
              <div className="text-2xl font-black font-mono text-[#f1f5f9]">{fmtPct(data.hit_rates![k].rate)}</div>
              <div className="text-xs font-mono text-[#64748b] mt-1">
                {data.hit_rates![k].hits}-{data.hit_rates![k].misses}{data.hit_rates![k].pushes ? `-${data.hit_rates![k].pushes}` : ''}
              </div>
            </div>
          ))}
        </div>
      )}
      <div className="mt-4 h-56 rounded-xl bg-[#191c22] border border-[#1e293b] p-4">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chart}>
            <XAxis dataKey="label" tick={{ fill: '#64748b', fontSize: 10 }} />
            <YAxis tick={{ fill: '#64748b', fontSize: 10 }} width={28} />
            <Tooltip contentStyle={{ background: '#0b0e14', border: '1px solid #1e293b' }} />
            {data.line != null && <ReferenceLine y={data.line} stroke="#fbbf24" strokeDasharray="4 4" />}
            <Bar dataKey="value">
              {chart.map((g) => (
                <Cell key={g.game_date} fill={data.line == null ? '#475569'
                  : g.value > data.line ? '#22c55e' : g.value < data.line ? '#ef4444' : '#94a3b8'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      {data.prices.length > 0 && (
        <table className="mt-4 w-full text-sm">
          <tbody>
            {data.prices.map((p) => (
              <tr key={`${p.book}-${p.side}-${p.line}`} className="border-b border-[#1e293b]/50">
                <td className="py-2 text-[#94a3b8] font-mono text-xs">{bookLabel(p.book)}</td>
                <td className="py-2 text-[#f1f5f9] font-mono text-xs">{p.side} {p.line}</td>
                <td className="py-2 text-right font-mono text-xs">
                  {p.link ? <a href={p.link} target="_blank" rel="noopener noreferrer sponsored" className="text-[#a4e6ff]">
                    {fmtAmerican(p.price)}</a> : fmtAmerican(p.price)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <EdgeFooter />
    </div>
  );
}
