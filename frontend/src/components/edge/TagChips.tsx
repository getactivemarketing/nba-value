// frontend/src/components/edge/TagChips.tsx
import { useState } from 'react';
import type { EvTag } from '@/types/edge';

// Mirrors backend TAG_LABEL.
export const TAG_LABEL: Record<string, string> = {
  def_soft: 'Soft matchup', def_tough: 'Tough matchup', def_rz: 'Red-zone-friendly D',
  def_td: 'TD-friendly D', usage_up: 'Usage rising', rz_role: 'Red-zone role',
  teammate_out: 'Teammate out', questionable: 'Questionable',
};

/** "def_soft:tailwind" -> "Soft matchup ↑" */
export function tagGroupLabel(k: string): string {
  const [key, direction] = k.split(':');
  const arrow = direction === 'tailwind' ? ' ↑' : direction === 'headwind' ? ' ↓' : '';
  return `${TAG_LABEL[key] ?? key}${arrow}`;
}

const CLASS: Record<EvTag['verdict'], string> = {
  proven: 'bg-[#a4e6ff]/15 text-[#a4e6ff] border-[#a4e6ff]/40',
  neutral: 'bg-[#1e293b] text-[#94a3b8] border-[#334155]',
  negative: 'bg-[#fbbf24]/10 text-[#fbbf24] border-[#fbbf24]/40',
};

const signedPct = (x: number) => `${x >= 0 ? '+' : '−'}${Math.abs(x * 100).toFixed(1)}%`;

function verdictCopy(t: EvTag): string {
  if (t.verdict === 'proven') return `Proven in 2023–25 backtest: ${signedPct(t.lift)} CLV`;
  if (t.verdict === 'negative') return `Underperformed in 2023–25 backtest: ${signedPct(t.lift)} CLV`;
  return 'No measurable edge in 2023–25 backtest';
}

const CUE: Record<EvTag['verdict'], string> = { proven: ' ✓', negative: ' !', neutral: '' };
const titleOf = (t: EvTag) => `${t.detail} · ${verdictCopy(t)}`;
const chipText = (t: EvTag) => `${t.label} ${t.direction === 'tailwind' ? '↑' : '↓'}${CUE[t.verdict]}`;

export function TagChips({ tags }: { tags: EvTag[] }) {
  const [open, setOpen] = useState<string | null>(null);
  if (!tags.length) return null;
  const id = (t: EvTag) => `${t.key}:${t.direction}`;
  const toggle = (t: EvTag) => setOpen((o) => (o === id(t) ? null : id(t)));
  const openTag = tags.find((t) => id(t) === open);
  return (
    <div className="mt-1">
      <div className="flex flex-wrap gap-1">
        {tags.map((t) => (
          <span key={id(t)} title={titleOf(t)} aria-label={`${chipText(t)} — ${titleOf(t)}`} tabIndex={0} role="button"
                aria-expanded={open === id(t)} onClick={() => toggle(t)}
                onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(t); } }}
                className={`cursor-pointer rounded border px-1.5 py-0.5 text-[10px] font-mono ${CLASS[t.verdict]}`}>
            {chipText(t)}
          </span>
        ))}
      </div>
      {openTag && <div className="mt-1 text-[10px] font-mono text-[#94a3b8]">{titleOf(openTag)}</div>}
    </div>
  );
}
