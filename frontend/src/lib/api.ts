import axios from 'axios';
import type { Market, BetDetail, BetHistory, MarketFilters, Algorithm } from '@/types/market';
import type { PerformanceBucket } from '@/types/evaluation';

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
  async getEvaluationSummary(
    days: number = 14,
    minValue: number = 65
  ): Promise<EvaluationSummary> {
    const response = await client.get<EvaluationSummary>(
      `/evaluation/summary?days=${days}&min_value=${minValue}`
    );
    return response.data;
  },

  async getPerformanceByBucket(
    days: number = 14
  ): Promise<PerformanceBucket[]> {
    const response = await client.get<PerformanceBucket[]>(
      `/evaluation/performance?days=${days}`
    );
    return response.data;
  },

  async getDailyResults(
    days: number = 7,
    minValue: number = 65
  ): Promise<DailyResult[]> {
    const response = await client.get<DailyResult[]>(
      `/evaluation/daily?days=${days}&min_value=${minValue}`
    );
    return response.data;
  },

  async getPredictionPerformance(
    days: number = 14,
    minValue: number = 0
  ): Promise<PredictionPerformance> {
    const response = await client.get<PredictionPerformance>(
      `/evaluation/predictions?days=${days}&min_value=${minValue}`
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

  // Trends
  async getATSLeaderboard(): Promise<ATSTeam[]> {
    const response = await client.get<ATSTeam[]>('/trends/ats');
    return response.data;
  },

  async getOULeaderboard(): Promise<OUTeam[]> {
    const response = await client.get<OUTeam[]>('/trends/ou');
    return response.data;
  },

  async getSituationalTrends(): Promise<SituationalTrends> {
    const response = await client.get<SituationalTrends>('/trends/situational');
    return response.data;
  },

  async getModelPerformance(days: number = 30): Promise<ModelPerformance> {
    const response = await client.get<ModelPerformance>(`/trends/model?days=${days}`);
    return response.data;
  },

  // Line Movement
  async getLineMovement(gameId: string): Promise<LineMovementResponse> {
    const response = await client.get<LineMovementResponse>(`/games/${gameId}/line-movement`);
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
  ats_record: string;
  ou_record: string;
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

export interface PredictedScore {
  home: number;
  away: number;
}

export interface GamePrediction {
  winner: string;
  winner_prob: number;
  confidence: 'high' | 'medium' | 'low';
  spread_pick: SpreadPick | null;
  best_bet: BestBet | null;
  factors: string[];
  predicted_score: PredictedScore | null;
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

export interface HeadToHead {
  home_wins: number;
  away_wins: number;
  total_games: number;
  record: string;
  recent: {
    date: string;
    score: string;
    home_won: boolean;
  }[];
}

export interface SharpMoney {
  signal: 'sharp_home' | 'sharp_away' | 'neutral';
  spread_movement: number;
  total_movement: number;
  opening_spread: number | null;
  current_spread: number | null;
  opening_total: number | null;
  current_total: number | null;
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
  head_to_head: HeadToHead | null;
  sharp_money: SharpMoney | null;
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

export interface BetTypeStats {
  wins: number;
  losses: number;
  pushes: number;
  profit: number;
  record: string;
  roi: number | null;
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
  by_type: {
    spread: BetTypeStats;
    total: BetTypeStats;
    moneyline: BetTypeStats;
  };
}

export interface EvaluationSummary {
  period_days: number;
  min_value_threshold: number;
  total_bets: number;
  wins: number;
  losses: number;
  pushes: number;
  win_rate: number | null;
  profit: number;
  roi: number | null;
}

export interface PredictionPerformance {
  summary: {
    total_predictions: number;
    winner_accuracy: {
      wins: number;
      losses: number;
      rate: number | null;
    };
    best_bet_performance: {
      wins: number;
      losses: number;
      pushes: number;
      win_rate: number | null;
      profit: number;
      roi: number | null;
    };
    pending_grading: number;
  };
  by_value_bucket: {
    bucket: string;
    total: number;
    wins: number;
    losses: number;
    win_rate: number | null;
    profit: number;
    roi: number | null;
  }[];
  recent_predictions: {
    matchup: string;
    tip_time: string;
    predicted_winner: string;
    winner_prob: number;
    confidence: string;
    best_bet: {
      type: string;
      team: string;
      line: number | null;
      value_score: number | null;
    } | null;
    actual_winner: string;
    final_score: string;
    winner_correct: boolean;
    bet_result: 'win' | 'loss' | 'push' | null;
    bet_profit: number | null;
    injury_edge: number | null;
  }[];
}

// Trends Types
export interface ATSTeam {
  rank: number;
  team: string;
  ats_wins: number;
  ats_losses: number;
  ats_pushes: number;
  ats_record: string;
  ats_pct: number;
  wins_l10: number;
  losses_l10: number;
  l10_record: string;
  home_record: string;
  away_record: string;
  net_rtg: number | null;
}

export interface OUTeam {
  rank: number;
  team: string;
  overs: number;
  unders: number;
  pushes: number;
  ou_record: string;
  over_pct: number;
  ppg: number;
  opp_ppg: number;
  avg_total: number;
  pace: string;
}

export interface SituationalTrend {
  situation: string;
  games: number;
  covers: number;
  cover_pct: number;
}

export interface SituationalTrends {
  by_rest: SituationalTrend[];
  by_location: SituationalTrend[];
  b2b_summary: {
    games: number;
    covers: number;
    cover_pct: number;
  };
  summary: {
    total_home_wins: number;
    total_home_losses: number;
    total_away_wins: number;
    total_away_losses: number;
  };
}

export interface ModelPerformance {
  days_analyzed: number;
  total_predictions: number;
  winner_accuracy: {
    correct: number;
    total: number;
    pct: number;
  };
  algo_a: {
    wins: number;
    losses: number;
    pushes: number;
    profit: number;
    win_rate: number;
    roi: number;
  };
  algo_b: {
    wins: number;
    losses: number;
    pushes: number;
    profit: number;
    win_rate: number;
    roi: number;
  };
  by_bucket: {
    bucket: string;
    bets: number;
    wins: number;
    losses: number;
    win_rate: number;
    profit: number;
    roi: number;
  }[];
  recent: {
    date: string;
    winner_correct: boolean;
    bet_result: string;
    value_score: number;
  }[];
}

// Line Movement Types
export interface LineMovementPoint {
  snapshot_time: string;
  minutes_to_tip: number;
  home_spread: number | null;
  away_spread: number | null;
  total_line: number | null;
  home_spread_odds: number | null;
  over_odds: number | null;
}

export interface SharpMoneySignal {
  signal: 'sharp_home' | 'sharp_away' | 'neutral';
  spread_movement: number;
  total_movement: number;
  opening_spread: number | null;
  current_spread: number | null;
  opening_total: number | null;
  current_total: number | null;
  interpretation: string;
}

export interface LineMovementResponse {
  game_id: string;
  snapshots: LineMovementPoint[];
  sharp_money: SharpMoneySignal;
}
