// frontend/src/pages/EdgeHome.tsx
import { Link } from 'react-router-dom';
import { useEdgeGames } from '@/hooks/useEdge';
import { EdgeFinder } from '@/pages/EdgeFinder';
import { EdgeFooter } from '@/components/edge/EdgeFooter';

export function EdgeHome() {
  const { data: games = [] } = useEdgeGames();
  return (
    <div>
      <h1 className="text-3xl font-bold text-[#f1f5f9] font-display tracking-tight">Tonight's NBA slate</h1>
      <div className="mt-4 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
        {games.map((g) => (
          <Link key={g.event_id} to={`/edge/odds/${g.event_id}`}
                className="bg-[#0b0e14] rounded-lg p-4 border border-[#1e293b] hover:border-[#a4e6ff]">
            <div className="text-[#f1f5f9] font-mono text-sm">{g.away_team} @ {g.home_team}</div>
            <div className="text-xs font-mono text-[#64748b] mt-1">
              {new Date(g.commence_time).toLocaleString([], { weekday: 'short', hour: 'numeric', minute: '2-digit' })}
              {' · '}{g.prop_markets} props
            </div>
          </Link>
        ))}
      </div>
      <div className="mt-8 flex items-baseline justify-between">
        <h2 className="text-base font-black font-mono text-[#94a3b8]">Top +EV right now</h2>
        <Link to="/edge" className="text-xs font-mono text-[#a4e6ff]">See all →</Link>
      </div>
      <EdgeFinder limit={10} />
      <EdgeFooter />
    </div>
  );
}
