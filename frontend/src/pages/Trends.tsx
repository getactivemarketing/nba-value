import { useState } from 'react';
import { useATSLeaderboard, useOULeaderboard, useSituationalTrends, useModelPerformance } from '@/hooks/useTrends';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';

type TrendTab = 'ats' | 'ou' | 'situational' | 'model';

const TEAM_COLORS: Record<string, string> = {
  ATL: '#E03A3E', BOS: '#007A33', BKN: '#000000', CHA: '#1D1160',
  CHI: '#CE1141', CLE: '#860038', DAL: '#00538C', DEN: '#0E2240',
  DET: '#C8102E', GSW: '#1D428A', HOU: '#CE1141', IND: '#002D62',
  LAC: '#C8102E', LAL: '#552583', MEM: '#5D76A9', MIA: '#98002E',
  MIL: '#00471B', MIN: '#0C2340', NOP: '#0C2340', NYK: '#006BB6',
  OKC: '#007AC1', ORL: '#0077C0', PHI: '#006BB6', PHX: '#1D1160',
  POR: '#E03A3E', SAC: '#5A2D81', SAS: '#C4CED4', TOR: '#CE1141',
  UTA: '#002B5C', WAS: '#002B5C',
};

export function Trends() {
  const [activeTab, setActiveTab] = useState<TrendTab>('ats');

  const tabs: { value: TrendTab; label: string }[] = [
    { value: 'ats', label: 'ATS Records' },
    { value: 'ou', label: 'Over/Under' },
    { value: 'situational', label: 'Situational' },
    { value: 'model', label: 'Model Performance' },
  ];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Betting Trends</h1>
        <p className="text-sm text-gray-500 mt-1">
          Team ATS records, O/U tendencies, and situational analysis
        </p>
      </div>

      {/* Tab Navigation */}
      <div className="flex space-x-1 bg-gray-100 p-1 rounded-lg w-fit">
        {tabs.map((tab) => (
          <button
            key={tab.value}
            onClick={() => setActiveTab(tab.value)}
            className={`px-4 py-2 text-sm font-medium rounded-md transition-colors ${
              activeTab === tab.value
                ? 'bg-white text-gray-900 shadow-sm'
                : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <div className="card">
        {activeTab === 'ats' && <ATSLeaderboard />}
        {activeTab === 'ou' && <OULeaderboard />}
        {activeTab === 'situational' && <SituationalTrends />}
        {activeTab === 'model' && <ModelPerformanceSection />}
      </div>
    </div>
  );
}

function ATSLeaderboard() {
  const { data: teams, isLoading, error } = useATSLeaderboard();

  if (isLoading) {
    return (
      <div className="flex justify-center py-12">
        <LoadingSpinner size="lg" />
      </div>
    );
  }

  if (error || !teams) {
    return (
      <div className="text-center py-8 text-gray-500">
        Failed to load ATS data
      </div>
    );
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-gray-900">ATS Leaderboard (Last 10 Games)</h2>
        <span className="text-sm text-gray-500">Against the Spread Performance</span>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="text-left text-sm text-gray-500 border-b">
              <th className="pb-3 pr-4">#</th>
              <th className="pb-3 pr-4">Team</th>
              <th className="pb-3 pr-4 text-center">ATS Record</th>
              <th className="pb-3 pr-4 text-center">ATS %</th>
              <th className="pb-3 pr-4 text-center">L10 Record</th>
              <th className="pb-3 pr-4 text-center">Home</th>
              <th className="pb-3 pr-4 text-center">Away</th>
              <th className="pb-3 text-right">Net Rtg</th>
            </tr>
          </thead>
          <tbody>
            {teams.map((team) => (
              <tr key={team.team} className="border-b last:border-0 hover:bg-gray-50">
                <td className="py-3 pr-4 text-gray-500">{team.rank}</td>
                <td className="py-3 pr-4">
                  <div className="flex items-center gap-2">
                    <div
                      className="w-6 h-6 rounded-full flex items-center justify-center text-white text-xs font-bold"
                      style={{ backgroundColor: TEAM_COLORS[team.team] || '#666' }}
                    >
                      {team.team.charAt(0)}
                    </div>
                    <span className="font-medium">{team.team}</span>
                  </div>
                </td>
                <td className="py-3 pr-4 text-center">
                  <span className={`font-semibold ${
                    team.ats_pct >= 60 ? 'text-emerald-600' :
                    team.ats_pct <= 40 ? 'text-red-600' : 'text-gray-700'
                  }`}>
                    {team.ats_record}
                  </span>
                </td>
                <td className="py-3 pr-4 text-center">
                  <ATSBar percentage={team.ats_pct} />
                </td>
                <td className="py-3 pr-4 text-center text-gray-600">{team.l10_record}</td>
                <td className="py-3 pr-4 text-center text-gray-600">{team.home_record}</td>
                <td className="py-3 pr-4 text-center text-gray-600">{team.away_record}</td>
                <td className={`py-3 text-right font-medium ${
                  team.net_rtg && team.net_rtg > 0 ? 'text-emerald-600' :
                  team.net_rtg && team.net_rtg < 0 ? 'text-red-600' : 'text-gray-600'
                }`}>
                  {team.net_rtg ? (team.net_rtg > 0 ? '+' : '') + team.net_rtg : '-'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function ATSBar({ percentage }: { percentage: number }) {
  const width = Math.max(10, Math.min(100, percentage));
  const color = percentage >= 60 ? 'bg-emerald-500' : percentage <= 40 ? 'bg-red-500' : 'bg-amber-500';

  return (
    <div className="flex items-center gap-2">
      <div className="w-16 h-2 bg-gray-200 rounded-full overflow-hidden">
        <div
          className={`h-full ${color} rounded-full transition-all`}
          style={{ width: `${width}%` }}
        />
      </div>
      <span className="text-sm font-medium text-gray-700 w-12">{percentage}%</span>
    </div>
  );
}

function OULeaderboard() {
  const { data: teams, isLoading, error } = useOULeaderboard();

  if (isLoading) {
    return (
      <div className="flex justify-center py-12">
        <LoadingSpinner size="lg" />
      </div>
    );
  }

  if (error || !teams) {
    return (
      <div className="text-center py-8 text-gray-500">
        Failed to load O/U data
      </div>
    );
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-gray-900">Over/Under Tendencies (Last 10 Games)</h2>
        <span className="text-sm text-gray-500">Teams sorted by Over %</span>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="text-left text-sm text-gray-500 border-b">
              <th className="pb-3 pr-4">#</th>
              <th className="pb-3 pr-4">Team</th>
              <th className="pb-3 pr-4 text-center">O/U Record</th>
              <th className="pb-3 pr-4 text-center">Over %</th>
              <th className="pb-3 pr-4 text-center">PPG</th>
              <th className="pb-3 pr-4 text-center">Opp PPG</th>
              <th className="pb-3 pr-4 text-center">Avg Total</th>
              <th className="pb-3 text-right">Pace</th>
            </tr>
          </thead>
          <tbody>
            {teams.map((team) => (
              <tr key={team.team} className="border-b last:border-0 hover:bg-gray-50">
                <td className="py-3 pr-4 text-gray-500">{team.rank}</td>
                <td className="py-3 pr-4">
                  <div className="flex items-center gap-2">
                    <div
                      className="w-6 h-6 rounded-full flex items-center justify-center text-white text-xs font-bold"
                      style={{ backgroundColor: TEAM_COLORS[team.team] || '#666' }}
                    >
                      {team.team.charAt(0)}
                    </div>
                    <span className="font-medium">{team.team}</span>
                  </div>
                </td>
                <td className="py-3 pr-4 text-center font-semibold">
                  {team.ou_record}
                </td>
                <td className="py-3 pr-4 text-center">
                  <OUBar overPct={team.over_pct} />
                </td>
                <td className="py-3 pr-4 text-center text-gray-600">{team.ppg}</td>
                <td className="py-3 pr-4 text-center text-gray-600">{team.opp_ppg}</td>
                <td className="py-3 pr-4 text-center font-medium">{team.avg_total}</td>
                <td className="py-3 text-right">
                  <span className={`px-2 py-1 rounded text-xs font-medium ${
                    team.pace.includes('Fast') ? 'bg-orange-100 text-orange-700' :
                    team.pace.includes('Slow') ? 'bg-blue-100 text-blue-700' :
                    'bg-gray-100 text-gray-700'
                  }`}>
                    {team.pace}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function OUBar({ overPct }: { overPct: number }) {
  return (
    <div className="flex items-center gap-2">
      <div className="w-20 h-3 bg-gray-200 rounded-full overflow-hidden flex">
        <div
          className="h-full bg-orange-500 transition-all"
          style={{ width: `${overPct}%` }}
        />
        <div
          className="h-full bg-blue-500 transition-all"
          style={{ width: `${100 - overPct}%` }}
        />
      </div>
      <span className="text-xs text-gray-500 w-8">{overPct}%</span>
    </div>
  );
}

function SituationalTrends() {
  const { data, isLoading, error } = useSituationalTrends();

  if (isLoading) {
    return (
      <div className="flex justify-center py-12">
        <LoadingSpinner size="lg" />
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="text-center py-8 text-gray-500">
        Failed to load situational data
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <h2 className="text-lg font-semibold text-gray-900">Situational Betting Trends</h2>

      <div className="grid gap-6 md:grid-cols-2">
        {/* Rest Days Analysis */}
        <div className="border rounded-lg p-4">
          <h3 className="font-medium text-gray-900 mb-4">ATS by Rest Days</h3>
          <div className="space-y-3">
            {data.by_rest.map((trend) => (
              <div key={trend.situation} className="flex items-center justify-between">
                <span className="text-sm text-gray-600">{trend.situation}</span>
                <div className="flex items-center gap-3">
                  <span className="text-sm text-gray-500">{trend.games} games</span>
                  <span className={`font-semibold ${
                    trend.cover_pct > 52 ? 'text-emerald-600' :
                    trend.cover_pct < 48 ? 'text-red-600' : 'text-gray-700'
                  }`}>
                    {trend.cover_pct}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Home vs Away */}
        <div className="border rounded-lg p-4">
          <h3 className="font-medium text-gray-900 mb-4">ATS by Location</h3>
          <div className="space-y-3">
            {data.by_location.map((trend) => (
              <div key={trend.situation} className="flex items-center justify-between">
                <span className="text-sm text-gray-600">{trend.situation} Teams</span>
                <div className="flex items-center gap-3">
                  <span className="text-sm text-gray-500">{trend.covers}/{trend.games}</span>
                  <span className={`font-semibold ${
                    trend.cover_pct > 52 ? 'text-emerald-600' :
                    trend.cover_pct < 48 ? 'text-red-600' : 'text-gray-700'
                  }`}>
                    {trend.cover_pct}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* B2B Summary */}
        <div className="border rounded-lg p-4 md:col-span-2">
          <h3 className="font-medium text-gray-900 mb-4">Back-to-Back Performance</h3>
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-500">Teams on B2B covering the spread</p>
              <p className="text-2xl font-bold text-gray-900 mt-1">
                {data.b2b_summary.covers}/{data.b2b_summary.games}
              </p>
            </div>
            <div className={`text-3xl font-bold ${
              data.b2b_summary.cover_pct > 50 ? 'text-emerald-600' : 'text-red-600'
            }`}>
              {data.b2b_summary.cover_pct}%
            </div>
          </div>
          <p className="text-xs text-gray-500 mt-2">
            {data.b2b_summary.cover_pct > 50
              ? 'Teams on back-to-backs are covering at a profitable rate'
              : 'Fading teams on back-to-backs has been profitable'}
          </p>
        </div>
      </div>
    </div>
  );
}

function ModelPerformanceSection() {
  const [days, setDays] = useState(30);
  const { data, isLoading, error } = useModelPerformance(days);

  if (isLoading) {
    return (
      <div className="flex justify-center py-12">
        <LoadingSpinner size="lg" />
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="text-center py-8 text-gray-500">
        Failed to load model performance data
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-gray-900">Model Performance</h2>
        <select
          value={days}
          onChange={(e) => setDays(Number(e.target.value))}
          className="border rounded-md px-3 py-1.5 text-sm"
        >
          <option value={7}>Last 7 days</option>
          <option value={14}>Last 14 days</option>
          <option value={30}>Last 30 days</option>
          <option value={90}>Last 90 days</option>
        </select>
      </div>

      {data.total_predictions === 0 ? (
        <div className="text-center py-8 text-gray-500">
          <p>No graded predictions yet.</p>
          <p className="text-sm mt-1">Predictions will appear here after games complete.</p>
        </div>
      ) : (
        <>
          {/* Summary Cards */}
          <div className="grid gap-4 grid-cols-2 md:grid-cols-4">
            <div className="border rounded-lg p-4">
              <p className="text-sm text-gray-500">Total Predictions</p>
              <p className="text-2xl font-bold text-gray-900">{data.total_predictions}</p>
            </div>
            <div className="border rounded-lg p-4">
              <p className="text-sm text-gray-500">Winner Accuracy</p>
              <p className="text-2xl font-bold text-gray-900">
                {data.winner_accuracy.pct}%
              </p>
              <p className="text-xs text-gray-500">
                {data.winner_accuracy.correct}/{data.winner_accuracy.total}
              </p>
            </div>
            <div className="border rounded-lg p-4">
              <p className="text-sm text-gray-500">Algo A Record</p>
              <p className="text-2xl font-bold text-gray-900">
                {data.algo_a.wins}-{data.algo_a.losses}
              </p>
              <p className={`text-sm font-medium ${data.algo_a.profit >= 0 ? 'text-emerald-600' : 'text-red-600'}`}>
                {data.algo_a.profit >= 0 ? '+' : ''}${data.algo_a.profit} ({data.algo_a.roi}% ROI)
              </p>
            </div>
            <div className="border rounded-lg p-4">
              <p className="text-sm text-gray-500">Algo B Record</p>
              <p className="text-2xl font-bold text-gray-900">
                {data.algo_b.wins}-{data.algo_b.losses}
              </p>
              <p className={`text-sm font-medium ${data.algo_b.profit >= 0 ? 'text-emerald-600' : 'text-red-600'}`}>
                {data.algo_b.profit >= 0 ? '+' : ''}${data.algo_b.profit} ({data.algo_b.roi}% ROI)
              </p>
            </div>
          </div>

          {/* Performance by Bucket */}
          {data.by_bucket.length > 0 && (
            <div className="border rounded-lg p-4">
              <h3 className="font-medium text-gray-900 mb-4">Performance by Value Score</h3>
              <div className="space-y-3">
                {data.by_bucket.map((bucket) => (
                  <div key={bucket.bucket} className="flex items-center justify-between py-2 border-b last:border-0">
                    <div className="flex items-center gap-3">
                      <span className="font-medium text-gray-900 w-16">{bucket.bucket}</span>
                      <span className="text-sm text-gray-500">{bucket.bets} bets</span>
                    </div>
                    <div className="flex items-center gap-6">
                      <span className="text-sm">{bucket.wins}-{bucket.losses}</span>
                      <span className={`font-semibold w-16 text-right ${
                        bucket.win_rate > 52 ? 'text-emerald-600' :
                        bucket.win_rate < 48 ? 'text-red-600' : 'text-gray-700'
                      }`}>
                        {bucket.win_rate}%
                      </span>
                      <span className={`font-medium w-20 text-right ${
                        bucket.profit >= 0 ? 'text-emerald-600' : 'text-red-600'
                      }`}>
                        {bucket.profit >= 0 ? '+' : ''}${bucket.profit}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Recent Results */}
          {data.recent.length > 0 && (
            <div className="border rounded-lg p-4">
              <h3 className="font-medium text-gray-900 mb-4">Recent Predictions</h3>
              <div className="flex gap-2 flex-wrap">
                {data.recent.map((r, i) => (
                  <div
                    key={i}
                    className={`w-8 h-8 rounded flex items-center justify-center text-xs font-bold ${
                      r.bet_result === 'win' ? 'bg-emerald-100 text-emerald-700' :
                      r.bet_result === 'loss' ? 'bg-red-100 text-red-700' :
                      'bg-gray-100 text-gray-700'
                    }`}
                    title={`${r.date}: ${r.bet_result?.toUpperCase() || 'PUSH'} (${r.value_score}%)`}
                  >
                    {r.bet_result === 'win' ? 'W' : r.bet_result === 'loss' ? 'L' : 'P'}
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
