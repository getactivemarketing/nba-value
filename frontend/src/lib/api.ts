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

  // Games
  async getUpcomingGames(hours: number = 24): Promise<GameWithTrends[]> {
    const response = await client.get<GameWithTrends[]>(`/games/upcoming?hours=${hours}`);
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
  win_pct_l10: number | null;
  net_rtg_l10: number | null;
  rest_days: number | null;
  is_b2b: boolean;
  home_away_pct: number | null;
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
}
