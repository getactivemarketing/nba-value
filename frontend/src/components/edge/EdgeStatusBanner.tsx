// frontend/src/components/edge/EdgeStatusBanner.tsx
import { useEdgeHealth } from '@/hooks/useEdge';

export function EdgeStatusBanner() {
  const { data, isError } = useEdgeHealth();
  if (!isError && !data?.delayed) return null;
  return (
    <div className="mb-4 rounded-lg border border-[#78350f] bg-[#1c1206] px-4 py-2 text-xs font-mono text-[#fbbf24]">
      Odds delayed — prices may be out of date. +EV is hidden for anything stale.
    </div>
  );
}
