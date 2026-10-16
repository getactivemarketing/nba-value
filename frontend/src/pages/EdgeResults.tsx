// frontend/src/pages/EdgeResults.tsx
import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { useEdgeResults } from '@/hooks/useEdge';
import { EdgeFooter } from '@/components/edge/EdgeFooter';
import { bookLabel, fmtPct, marketLabel } from '@/lib/edgeFormat';
import type { ResultsBlock } from '@/types/edge';

const pts = (x: number | null) => (x == null ? '—' : `${x >= 0 ? '+' : ''}${(x * 100).toFixed(2)} pts`);

function Breakdown({ title, data, label }: { title: string; data: Record<string, ResultsBlock>;
  label: (k: string) => string }) {
  return (
    <div className="bg-[#0b0e14] rounded-lg p-4 border border-[#1e293b]">
      <div className="text-[10px] text-[#64748b] uppercase font-bold tracking-widest mb-2">{title}</div>
      <table className="w-full text-sm">
        <tbody>
          {Object.entries(data).map(([k, b]) => (
            <tr key={k} className="border-b border-[#1e293b]/50">
              <td className="py-2 text-[#94a3b8] font-mono text-xs">{label(k)}</td>
              <td className="py-2 text-center font-mono text-xs text-[#f1f5f9]">{b.n}</td>
              <td className="py-2 text-center font-mono text-xs text-[#f1f5f9]">{pts(b.clv.mean_clv)}</td>
              <td className="py-2 text-center font-mono text-xs text-[#94a3b8]">{fmtPct(b.record.win_rate)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function EdgeResults() {
  const { data, isLoading } = useEdgeResults();
  if (isLoading || !data) return <div className="text-center py-8 text-[#64748b] font-mono text-sm">Loading…</div>;
  const c = data.clv;
  return (
    <div>
      <h1 className="text-3xl font-bold text-[#f1f5f9] font-display tracking-tight">Track record</h1>
      <p className="text-sm text-[#64748b] font-mono max-w-2xl mt-2 leading-relaxed">
        Every +EV price is logged the moment we first see it and graded against the closing line. Closing-line
        value (CLV) is the gap between the market's final fair price and the price you got. It settles a betting
        edge in hundreds of bets, where win rate needs thousands. We publish it whether it is good or bad.
      </p>
      <div className="mt-4 grid grid-cols-2 sm:grid-cols-4 gap-2">
        {[['Logged', String(data.n)], ['Mean CLV', pts(c.mean_clv)],
          ['95% CI', c.ci95 ? `${pts(c.ci95[0])} … ${pts(c.ci95[1])}` : '—'],
          ['Beat the close', fmtPct(c.beat_close_rate)],
          ['Record', `${data.record.wins}-${data.record.losses}-${data.record.pushes}`],
          ['Win rate', fmtPct(data.record.win_rate)], ['Units', data.record.units.toFixed(2)],
          ['ROI', data.record.roi == null ? '—' : `${(data.record.roi * 100).toFixed(1)}%`]].map(([k, v]) => (
          <div key={k} className="bg-[#0b0e14] rounded-lg p-4 border border-[#1e293b]">
            <div className="text-[10px] text-[#64748b] uppercase font-bold tracking-widest">{k}</div>
            <div className="text-lg font-black font-mono text-[#f1f5f9]">{v}</div>
          </div>
        ))}
      </div>
      <p className="text-[10px] text-[#64748b] font-mono mt-1.5 italic">
        Sample: {c.measured} prices across {c.bets} distinct bets ({c.verdict}). The interval counts each bet once,
        however many books offered it.
      </p>
      <div className="mt-4 h-56 rounded-xl bg-[#191c22] border border-[#1e293b] p-4">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data.daily_clv}>
            <CartesianGrid stroke="#1e293b" />
            <XAxis dataKey="date" tick={{ fill: '#64748b', fontSize: 10 }} />
            <YAxis tick={{ fill: '#64748b', fontSize: 10 }} width={44} tickFormatter={(v) => `${(v * 100).toFixed(1)}`} />
            <Tooltip contentStyle={{ background: '#0b0e14', border: '1px solid #1e293b' }} />
            <ReferenceLine y={0} stroke="#475569" />
            <Line type="monotone" dataKey="mean_clv" stroke="#a4e6ff" dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-3">
        <Breakdown title="By market" data={data.by_market} label={marketLabel} />
        <Breakdown title="By book" data={data.by_book} label={bookLabel} />
      </div>
      <EdgeFooter />
    </div>
  );
}
