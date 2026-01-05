export type MarketType = 'spread' | 'moneyline' | 'total' | 'prop';
export type GameStatus = 'scheduled' | 'in_progress' | 'final';
export type Algorithm = 'a' | 'b';

export interface Game {
  game_id: string;
  home_team: string;
  away_team: string;
  tip_time: string;
  status: GameStatus;
}

export interface Market {
  market_id: string;
  game_id: string;
  game?: Game;
  market_type: MarketType;
  outcome_label: string;
  line: number | null;
  odds_decimal: number;
  odds_american?: number;

  // Model outputs
  p_true: number;
  p_market: number;
  raw_edge: number;

  // Algorithm A outputs
  algo_a_value_score: number;
  algo_a_confidence: number;
  algo_a_market_quality: number;

  // Algorithm B outputs
  algo_b_value_score: number;
  algo_b_confidence: number;
  algo_b_market_quality: number;

  // Meta
  time_to_tip_minutes: number;
  calc_time: string;
  book?: string;
}

export interface MarketFilters {
  algorithm: Algorithm;
  marketType: MarketType | null;
  minValueScore: number;
  minConfidence: number;
}

export interface ConfidenceBreakdown {
  ensemble_agreement: number;
  calibration_reliability: number;
  injury_certainty: number;
  segment_reliability?: number;
  final_multiplier: number;
}

export interface MarketQualityBreakdown {
  liquidity_score: number;
  book_consensus: number;
  line_stability: number;
  final_multiplier: number;
}

export interface AlgorithmScore {
  algorithm: Algorithm;
  value_score: number;
  edge_score?: number;
  combined_edge?: number;
  confidence: ConfidenceBreakdown;
  market_quality: MarketQualityBreakdown;
}

export interface BetDetail extends Market {
  home_team: string;
  away_team: string;
  tip_time: string;
  edge_percentage: number;
  p_ensemble_mean: number;
  p_ensemble_std: number;
  algo_a: AlgorithmScore;
  algo_b: AlgorithmScore;
  active_algorithm: Algorithm;
  recommended_score: number;
}

export interface BetHistory {
  calc_time: string;
  p_true: number;
  p_market: number;
  raw_edge: number;
  algo_a_value_score: number;
  algo_b_value_score: number;
  odds_decimal: number;
  line?: number;
}
