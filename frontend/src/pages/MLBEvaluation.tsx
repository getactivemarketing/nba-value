import { useState } from 'react';
import { useMLBEvaluationSummary, useMLBDailyResults, useNRFIEvaluation, useUnderdogEvaluation } from '@/hooks/useMLBEvaluation';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { getTeamInfo } from '@/lib/mlbApi';

function StatCard({ label, value, color, small }: {
  label: string;
  value: string;
  color?: 'green' | 'red' | 'cyan';
  small?: boolean;
}) {
  const colorClass = color === 'green' ? 'text-[#66f796]'
    : color === 'red' ? 'text-[#ef4444]'
    : color === 'cyan' ? 'text-[#a4e6ff]'
    : 'text-[#f1f5f9]';

  return (
    <div className="bg-[#0b0e14] rounded-lg p-4 border border-[#1e293b]">
      <div className="text-[10px] text-[#64748b] uppercase font-bold tracking-widest mb-2">{label}</div>
      <div className={`${small ? 'text-xl' : 'text-2xl'} font-black font-mono ${colorClass}`}>{value}</div>
    </div>
  );
}

export function MLBEvaluation() {
  const [dailyDays, setDailyDays] = useState(14);

  const { data: summary, isLoading: summaryLoading } = useMLBEvaluationSummary();
  const { data: daily, isLoading: dailyLoading } = useMLBDailyResults(dailyDays);
  const { data: nrfi, isLoading: nrfiLoading } = useNRFIEvaluation(90);
  const { data: underdogs, isLoading: underDogsLoading } = useUnderdogEvaluation(90);

  const selectClass =
    'text-sm bg-[#0b0e14] border border-[#1e293b] text-[#f1f5f9] rounded px-2 py-1 font-mono focus:outline-none focus:border-[#a4e6ff]';

  const decided = (summary?.wins ?? 0) + (summary?.losses ?? 0);
  const roi = decided > 0 ? ((summary?.total_profit ?? 0) / (decided * 100) * 100).toFixed(1) : null;

  return (
    <div className="max-w-6xl mx-auto px-4 py-6 space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-[#f1f5f9] font-display tracking-tight">
          MLB <span className="text-[#a4e6ff]">PERFORMANCE</span>
        </h1>
        <p className="text-sm text-[#64748b] mt-1 font-mono">
          Season record, NRFI accuracy, and underdog ML track record
        </p>
      </div>

      {/* Season Summary Cards */}
      {summaryLoading && (
        <div className="flex justify-center py-8"><LoadingSpinner /></div>
      )}
      {!summaryLoading && summary && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <StatCard label="Record" value={`${summary.wins}-${summary.losses}${summary.pushes > 0 ? `-${summary.pushes}` : ''}`} />
          <StatCard label="Total Picks" value={String(summary.graded_predictions)} color="cyan" />
          <StatCard
            label="P/L (units)"
            value={`${summary.total_profit >= 0 ? '+' : ''}${summary.total_profit.toFixed(0)}u`}
            color={summary.total_profit >= 0 ? 'green' : 'red'}
          />
          <StatCard
            label="ROI"
            value={roi ? `${Number(roi) >= 0 ? '+' : ''}${roi}%` : '-'}
            color={Number(roi) >= 0 ? 'green' : 'red'}
          />
        </div>
      )}

      {/* By Bet Type */}
      {!summaryLoading && summary?.by_type && Object.keys(summary.by_type).length > 0 && (
        <div className="bg-[#191c22] rounded-xl border border-[#1e293b] p-5">
          <h2 className="text-[10px] text-[#64748b] uppercase font-bold tracking-widest mb-4">By Bet Type</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {['moneyline', 'runline', 'total'].map((bt) => {
              const stats = (summary.by_type as Record<string, any>)[bt];
              if (!stats) return null;
              const d = stats.wins + stats.losses;
              return (
                <div key={bt} className="bg-[#0b0e14] rounded-lg p-4 border border-[#1e293b]">
                  <div className="text-[10px] text-[#64748b] uppercase font-bold tracking-widest mb-2">
                    {bt === 'moneyline' ? 'Moneyline' : bt === 'runline' ? 'Runline' : 'Total'}
                  </div>
                  <div className="text-xl font-black font-mono text-[#f1f5f9]">
                    {stats.wins}-{stats.losses}
                  </div>
                  <div className={`text-sm font-mono mt-1 ${stats.profit >= 0 ? 'text-[#66f796]' : 'text-[#ef4444]'}`}>
                    {stats.profit >= 0 ? '+' : ''}{stats.profit.toFixed(0)}u
                    {d > 0 && ` (${(stats.wins / d * 100).toFixed(0)}%)`}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* NRFI Accuracy */}
      {nrfiLoading && (
        <div className="flex justify-center py-8"><LoadingSpinner /></div>
      )}
      {!nrfiLoading && nrfi && nrfi.total_picks > 0 && (
        <div className="bg-[#191c22] rounded-xl border border-[#1e293b] p-5">
          <h2 className="text-[10px] text-[#64748b] uppercase font-bold tracking-widest mb-4">NRFI Pick Accuracy</h2>
          <div className="grid grid-cols-3 gap-4 mb-4">
            <StatCard label="Accuracy" value={nrfi.accuracy ? `${nrfi.accuracy}%` : '-'} color="cyan" small />
            <StatCard label="Hits" value={String(nrfi.nrfi_hits)} color="green" small />
            <StatCard label="Misses" value={String(nrfi.total_picks - nrfi.nrfi_hits)} color="red" small />
          </div>
          {nrfi.recent.length > 0 && (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-[10px] text-[#64748b] uppercase font-bold tracking-widest border-b border-[#1e293b]">
                    <th className="pb-2">Date</th>
                    <th className="pb-2">Matchup</th>
                    <th className="pb-2 text-center">1st Inn Runs</th>
                    <th className="pb-2 text-right">Result</th>
                  </tr>
                </thead>
                <tbody>
                  {nrfi.recent.slice(0, 10).map((pick, i) => (
                    <tr key={i} className="border-b border-[#1e293b]">
                      <td className="py-2 text-[#94a3b8] font-mono">{pick.date}</td>
                      <td className="py-2 text-[#f1f5f9]">
                        {getTeamInfo(pick.away_team).name} @ {getTeamInfo(pick.home_team).name}
                      </td>
                      <td className="py-2 text-center font-mono text-[#94a3b8]">{pick.first_inning_runs}</td>
                      <td className={`py-2 text-right font-mono font-bold ${pick.result === 'hit' ? 'text-[#66f796]' : 'text-[#ef4444]'}`}>
                        {pick.result === 'hit' ? 'NRFI' : 'YRFI'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {/* Underdog ML Track Record */}
      {underDogsLoading && (
        <div className="flex justify-center py-8"><LoadingSpinner /></div>
      )}
      {!underDogsLoading && underdogs && underdogs.total_picks > 0 && (
        <div className="bg-[#191c22] rounded-xl border border-[#1e293b] p-5">
          <h2 className="text-[10px] text-[#64748b] uppercase font-bold tracking-widest mb-4">Underdog ML Picks</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
            <StatCard label="Record" value={`${underdogs.wins}-${underdogs.losses}`} small />
            <StatCard
              label="P/L"
              value={`${underdogs.profit >= 0 ? '+' : ''}${underdogs.profit.toFixed(0)}u`}
              color={underdogs.profit >= 0 ? 'green' : 'red'}
              small
            />
            <StatCard label="Avg Odds" value={`+${underdogs.avg_odds_american}`} color="cyan" small />
            <StatCard label="Total Picks" value={String(underdogs.total_picks)} small />
          </div>
          {underdogs.biggest_wins.length > 0 && (
            <>
              <h3 className="text-[10px] text-[#64748b] uppercase font-bold tracking-widest mb-2 mt-4">Biggest Wins</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left text-[10px] text-[#64748b] uppercase font-bold tracking-widest border-b border-[#1e293b]">
                      <th className="pb-2">Date</th>
                      <th className="pb-2">Team</th>
                      <th className="pb-2 text-center">Odds</th>
                      <th className="pb-2 text-center">Score</th>
                      <th className="pb-2 text-right">Profit</th>
                    </tr>
                  </thead>
                  <tbody>
                    {underdogs.biggest_wins.map((w, i) => (
                      <tr key={i} className="border-b border-[#1e293b]">
                        <td className="py-2 text-[#94a3b8] font-mono">{w.date}</td>
                        <td className="py-2 text-[#f1f5f9] font-medium">{getTeamInfo(w.team).name}</td>
                        <td className="py-2 text-center font-mono text-[#a4e6ff]">+{w.odds_american}</td>
                        <td className="py-2 text-center font-mono text-[#94a3b8]">{w.score || '-'}</td>
                        <td className="py-2 text-right font-mono font-bold text-[#66f796]">+{w.profit.toFixed(0)}u</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </div>
      )}

      {/* Daily Results */}
      <div className="bg-[#191c22] rounded-xl border border-[#1e293b] p-5">
        <div className="flex justify-between items-center mb-4 flex-wrap gap-3">
          <h2 className="text-[10px] text-[#64748b] uppercase font-bold tracking-widest">Daily Results</h2>
          <select value={dailyDays} onChange={(e) => setDailyDays(Number(e.target.value))} className={selectClass}>
            <option value={7}>7 days</option>
            <option value={14}>14 days</option>
            <option value={30}>30 days</option>
          </select>
        </div>

        {dailyLoading && (
          <div className="flex justify-center py-8"><LoadingSpinner /></div>
        )}

        {!dailyLoading && daily && daily.length > 0 && (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-[10px] text-[#64748b] uppercase font-bold tracking-widest border-b border-[#1e293b]">
                  <th className="pb-2">Date</th>
                  <th className="pb-2 text-center">Picks</th>
                  <th className="pb-2 text-center">Record</th>
                  <th className="pb-2 text-center">Win %</th>
                  <th className="pb-2 text-right">P/L</th>
                </tr>
              </thead>
              <tbody>
                {daily.map((day) => {
                  const record = `${day.wins}-${day.losses}${day.pushes > 0 ? `-${day.pushes}` : ''}`;
                  const wr = day.win_rate ? `${(day.win_rate * 100).toFixed(0)}%` : '-';
                  return (
                    <tr key={day.date} className="border-b border-[#1e293b]">
                      <td className="py-2 text-[#94a3b8] font-mono">{day.date}</td>
                      <td className="py-2 text-center font-mono text-[#f1f5f9]">{day.predictions}</td>
                      <td className="py-2 text-center font-mono text-[#f1f5f9]">{record}</td>
                      <td className="py-2 text-center font-mono text-[#94a3b8]">{wr}</td>
                      <td className={`py-2 text-right font-mono font-bold ${day.profit >= 0 ? 'text-[#66f796]' : 'text-[#ef4444]'}`}>
                        {day.profit >= 0 ? '+' : ''}{day.profit.toFixed(0)}u
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}

        {!dailyLoading && (!daily || daily.length === 0) && (
          <div className="text-center py-8 text-[#64748b] font-mono text-sm">
            No daily results available yet
          </div>
        )}
      </div>
    </div>
  );
}
