// frontend/src/components/edge/blind.tsx
import { Link } from 'react-router-dom';
import type { BlindResult } from '@/types/edge';

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

export function NotFound() {
  return (
    <div className="text-center py-12 font-mono">
      <h1 className="text-xl font-black text-[#f1f5f9]">We couldn't find that play</h1>
      <p className="mt-2 text-sm text-[#94a3b8]">The link may be old or mistyped.</p>
      <Link to="/blind" className="inline-block mt-4 text-sm text-[#a4e6ff] hover:underline">See every blind play</Link>
    </div>
  );
}

