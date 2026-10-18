// frontend/src/components/edge/TagChips.tsx
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

export function TagChips({ tags }: { tags: EvTag[] }) {
  if (!tags.length) return null;
  return (
    <div className="mt-1 flex flex-wrap gap-1">
      {tags.map((t) => (
        <span key={`${t.key}:${t.direction}`} title={`${t.detail} · ${verdictCopy(t)}`}
              className={`rounded border px-1.5 py-0.5 text-[10px] font-mono ${CLASS[t.verdict]}`}>
          {`${t.label} ${t.direction === 'tailwind' ? '↑' : '↓'}`}
        </span>
      ))}
    </div>
  );
}
