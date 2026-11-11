// frontend/src/components/edge/FormCell.tsx
import type { EvForm } from '@/types/edge';

/** The bet side's record in the player's last 10 games, next to the market's chance.
 *  "cold" marks picks where recent form runs 15+ points below the market. */
export function FormCell({ form, flag, fairProb }: { form: EvForm | null; flag: boolean; fairProb: number }) {
  if (!form) return <span className="text-[#475569]">–</span>;
  const market = Math.round(fairProb * 100);
  return (
    <span className="inline-flex items-center gap-1">
      <span className={flag ? 'text-[#fbbf24]' : 'text-[#94a3b8]'}
            title={`Won ${form.hits} of his last ${form.games} games at this line · market says ${market}%`}>
        {form.hits}/{form.games}
      </span>
      {flag && (
        <span className="rounded bg-[#fbbf24]/15 px-1 text-[9px] font-bold uppercase text-[#fbbf24]"
              title="Recent form is 15+ points below the market's chance. These picks have done worse so far.">
          cold
        </span>
      )}
    </span>
  );
}
