import axios from 'axios';
import type { Market, BetDetail, BetHistory, MarketFilters, Algorithm } from '@/types/market';
import type { AlgorithmComparison, CalibrationPoint, PerformanceBucket } from '@/types/evaluation';

const API_BASE = import.meta.env.VITE_API_URL || '/api/v1';

const client = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add auth token to requests
client.interceptors.request.use((config) => {
  const token = localStorage.getItem('auth_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

export const api = {
  // Markets
  async getMarkets(filters: Partial<MarketFilters> = {}): Promise<Market[]> {
    const params = new URLSearchParams();
    if (filters.algorithm) params.append('algorithm', filters.algorithm);
    if (filters.marketType) params.append('market_type', filters.marketType);
    if (filters.minValueScore) params.append('min_value_score', String(filters.minValueScore));
    if (filters.minConfidence) params.append('min_confidence', String(filters.minConfidence));

    const response = await client.get<{ markets: Market[] }>(`/markets?${params}`);
    return response.data.markets;
  },

  async getLiveMarkets(algorithm: Algorithm = 'a'): Promise<Market[]> {
    const response = await client.get<Market[]>(`/markets/live?algorithm=${algorithm}`);
    return response.data;
  },

  // Bet Details
  async getBetDetail(marketId: string): Promise<BetDetail> {
    const response = await client.get<BetDetail>(`/bet/${marketId}`);
    return response.data;
  },

  async getBetHistory(marketId: string, hours: number = 24): Promise<BetHistory[]> {
    const response = await client.get<BetHistory[]>(`/bet/${marketId}/history?hours=${hours}`);
    return response.data;
  },

  // Evaluation
  async getAlgorithmComparison(
    startDate?: string,
    endDate?: string,
    marketType?: string
  ): Promise<AlgorithmComparison> {
    const params = new URLSearchParams();
    if (startDate) params.append('start_date', startDate);
    if (endDate) params.append('end_date', endDate);
    if (marketType) params.append('market_type', marketType);

    const response = await client.get<AlgorithmComparison>(`/evaluation/compare?${params}`);
    return response.data;
  },

  async getCalibrationCurve(
    marketType?: string,
    algorithm: Algorithm = 'a'
  ): Promise<CalibrationPoint[]> {
    const params = new URLSearchParams({ algorithm });
    if (marketType) params.append('market_type', marketType);

    const response = await client.get<CalibrationPoint[]>(`/evaluation/calibration?${params}`);
    return response.data;
  },

  async getPerformanceByBucket(
    algorithm: Algorithm = 'a',
    bucketType: 'score' | 'edge' | 'confidence' = 'score'
  ): Promise<PerformanceBucket[]> {
    const response = await client.get<PerformanceBucket[]>(
      `/evaluation/performance?algorithm=${algorithm}&bucket_type=${bucketType}`
    );
    return response.data;
  },

  async getDailyResults(
    days: number = 7,
    algorithm: Algorithm = 'b',
    minValue: number = 50
  ): Promise<DailyResult[]> {
    const response = await client.get<DailyResult[]>(
      `/evaluation/daily?days=${days}&algorithm=${algorithm}&min_value=${minValue}`
    );
    return response.data;
  },

  // Games
  async getUpcomingGames(hours: number = 24): Promise<GameWithTrends[]> {
    const response = await client.get<GameWithTrends[]>(`/games/upcoming?hours=${hours}`);
    return response.data;
  },

  async getGameHistory(days: number = 7, team?: string): Promise<HistoricalGame[]> {
    const params = new URLSearchParams({ days: String(days) });
    if (team) params.append('team', team);
    const response = await client.get<HistoricalGame[]>(`/games/history?${params}`);
    return response.data;
  },

  async getTopPicks(minValueScore: number = 55, algorithm: 'a' | 'b' = 'a'): Promise<TopPicksResponse> {
    const response = await client.get<TopPicksResponse>(
      `/picks/top?min_value_score=${minValueScore}&algorithm=${algorithm}`
    );
    return response.data;
  },

  // Health
  async healthCheck(): Promise<{ status: string }> {
    const response = await client.get<{ status: string }>('/health');
    return response.data;
  },
};

export interface TeamTrends {
  record: string;
  home_record: string;
  away_record: string;
  l10_record: string;
  net_rtg_l10: number | null;
  rest_days: number | null;
  is_b2b: boolean;
}

export interface TeamInjuries {
  players_out: string[];
  players_questionable: string[];
  impact_score: number;  // 0-100
  total_impact_points: number;
  severity: 'none' | 'minor' | 'moderate' | 'severe';
}

export interface BestBet {
  type: string;
  team: string;
  line: number | null;
  value_score: number;
  edge: number;
  p_true: number;
  p_market: number;
}

export interface SpreadPick {
  team: string;
  line: number | null;
  value_score: number;
  edge: number;
  p_true: number;
}

export interface GamePrediction {
  winner: string;
  winner_prob: number;
  confidence: 'high' | 'medium' | 'low';
  spread_pick: SpreadPick | null;
  best_bet: BestBet | null;
  factors: string[];
}

export interface TornadoFactor {
  factor: string;
  label: string;
  home_value: number | string;
  away_value: number | string;
  diff: number;
  home_better: boolean | null;
  expected_pace?: number;
}

export interface GameWithTrends {
  game_id: string;
  home_team: string;
  away_team: string;
  home_team_full: string;
  away_team_full: string;
  tip_time: string;
  time_to_tip_minutes: number;
  markets_count: number;
  status: string;
  home_trends: TeamTrends;
  away_trends: TeamTrends;
  home_injuries: TeamInjuries;
  away_injuries: TeamInjuries;
  injury_edge: number;  // Positive = home has advantage (away more injured)
  prediction: GamePrediction | null;
  tornado_chart: TornadoFactor[];
}

export interface HistoricalGame {
  game_id: string;
  game_date: string;
  home_team: string;
  away_team: string;
  home_team_full: string;
  away_team_full: string;
  home_score: number;
  away_score: number;
  total_score: number;
  margin: number;
  closing_spread: number | null;
  closing_total: number | null;
  actual_winner: string;
  spread_result: string;
  spread_margin: number | null;
  total_result: string;
  total_margin: number | null;
}

export interface TopPick {
  game: string;
  pick: string;
  line: number | null;
  value_score: number;
  edge: number;
  model_prob: number;
  market_prob: number;
  market_type: string;
  tip_time: string;
}

export interface TopPicksResponse {
  spreads: TopPick[];
  moneylines: TopPick[];
  totals: TopPick[];
  best_edges: TopPick[];
  generated_at: string;
}

export interface DailyBet {
  matchup: string;
  bet: string;
  value_score: number;
  result: 'win' | 'loss' | 'push';
  profit: number;
  final_score: string;
}

export interface DailyResult {
  date: string;
  bets: DailyBet[];
  wins: number;
  losses: number;
  pushes: number;
  profit: number;
  record: string;
  roi: number;
}
