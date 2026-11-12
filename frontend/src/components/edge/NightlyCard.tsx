// frontend/src/components/edge/NightlyCard.tsx
import { Link } from 'react-router-dom';
import type { NightlyCard as Card, NightlyLeg, NightlyResponse } from '@/types/edge';
import { bookLabel, fmtAmerican, fmtEv } from '@/lib/edgeFormat';
import { FormCell } from './FormCell';

const fmtTime = (iso: string) => new Date(iso).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
const fmtUnits = (u: number) => `${u >= 0 ? '+' : ''}${u.toFixed(1)}u`;

function ResultBadge({ result }: { result: string | null }) {
  if (!result) return <span className="text-[#475569]">–</span>;
  const [txt, cls] = result === 'win' ? ['W', 'text-[#4ade80]'] : result === 'loss' ? ['L', 'text-[#f87171]']
    : ['P', 'text-[#94a3b8]'];
  return <span className={`font-bold ${cls}`}>{txt}</span>;
}

function LegRow({ l, rank }: { l: NightlyLeg; rank?: number }) {
  return (
    <tr className="border-b border-[#1e293b]/50">
      {rank != null && <td className="py-2 pr-2 font-mono text-[#64748b]">{rank}</td>}
      <td className="py-2">
        <div className="text-[#f1f5f9]">{l.bet}</div>
        <Link to={`/edge/odds/${l.event_id}?market=${l.market}`} className="text-[10px] text-[#64748b] font-mono hover:underline">
          {l.matchup} · {fmtTime(l.commence_time)}
        </Link>
      </td>
      <td className="py-2 text-[#94a3b8] font-mono text-xs">{bookLabel(l.book)}</td>
      <td className="py-2 text-center font-mono text-[#f1f5f9]">{fmtAmerican(l.price)}</td>
      <td className="py-2 text-center font-mono text-xs text-[#a4e6ff]">{fmtEv(l.ev)}</td>
      <td className="py-2 text-center font-mono text-xs"><FormCell form={l.form} flag={false} fairProb={l.fair_prob} /></td>
      <td className="py-2 text-center font-mono text-xs"><ResultBadge result={l.result} /></td>
    </tr>
  );
}

const Head = ({ rank }: { rank?: boolean }) => (
  <thead>
    <tr className="text-left text-[10px] text-[#64748b] uppercase font-bold tracking-widest border-b border-[#1e293b]">
      {rank && <th className="pb-2" />}<th className="pb-2">Bet</th><th className="pb-2">Book</th>
      <th className="pb-2 text-center">Price</th><th className="pb-2 text-center">EV</th>
      <th className="pb-2 text-center">L10</th><th className="pb-2 text-center">Result</th>
    </tr>
  </thead>
);

function CardBody({ card }: { card: Card }) {
  const p = card.parlay;
  return (
    <>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <Head rank />
          <tbody>{card.top.map((l, i) => <LegRow key={`${l.event_id}-${l.bet}`} l={l} rank={i + 1} />)}</tbody>
        </table>
      </div>
      {p && (
        <div className="mt-4 rounded-lg border border-[#a4e6ff]/30 bg-[#a4e6ff]/5 p-3">
          <div className="flex flex-wrap items-baseline justify-between gap-2">
            <div className="text-sm font-bold text-[#f1f5f9]">{p.legs.length}-leg parlay at {bookLabel(p.book)}</div>
            <div className="font-mono">
              <span className="text-lg font-bold text-[#a4e6ff]">{fmtAmerican(p.american)}</span>
              {p.result && <span className="ml-2"><ResultBadge result={p.result} /></span>}
            </div>
          </div>
          <div className="mt-1 text-xs text-[#94a3b8]">
            The market gives it a {Math.round(p.hit_prob * 100)}% chance it hits. Most nights it loses, so bet small.
          </div>
          <div className="mt-2 overflow-x-auto">
            <table className="w-full text-sm"><Head />
              <tbody>{p.legs.map((l) => <LegRow key={`p-${l.event_id}-${l.bet}`} l={l} />)}</tbody>
            </table>
          </div>
        </div>
      )}
    </>
  );
}

export function NightlyCard({ data }: { data: NightlyResponse }) {
  const { tonight, history, record, locks_at } = data;
  const last = history[0];
  return (
    <section className="mt-8 rounded-lg border border-[#1e293b] bg-[#0b0e14] p-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h2 className="text-base font-black font-mono text-[#f1f5f9]">Tonight's Top 5</h2>
        {tonight && <span className="text-[10px] font-mono text-[#64748b]">Locked {fmtTime(tonight.locked_at)}</span>}
      </div>
      <p className="mt-1 text-xs text-[#94a3b8]">
        The night's best props: 5+ books behind the price, one per game, no cold picks.
        {!tonight && <> Tonight's picks lock an hour before the first game{locks_at ? ` (${fmtTime(locks_at)})` : ''}.</>}
      </p>
      {tonight && <div className="mt-3"><CardBody card={tonight} /></div>}
      {!tonight && last && (
        <div className="mt-3">
          <div className="text-[10px] uppercase font-bold tracking-widest text-[#64748b] mb-1">Last slate · {last.slate_date}</div>
          <CardBody card={last} />
        </div>
      )}
      {(record.top.n > 0 || record.parlay.n > 0) && (
        <div className="mt-3 text-xs font-mono text-[#94a3b8]">
          Record: top picks {record.top.wins}-{record.top.losses} ({fmtUnits(record.top.units)})
          {' · '}parlays {record.parlay.wins} of {record.parlay.n} hit ({fmtUnits(record.parlay.units)})
        </div>
      )}
    </section>
  );
}
