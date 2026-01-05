export interface AlgorithmMetrics {
  brier_score: number;
  log_loss: number;
  clv_avg: number;
  roi: number;
  win_rate: number;
  bet_count: number;
}

export interface AlgorithmComparison {
  period_start: string;
  period_end: string;
  algo_a_metrics: AlgorithmMetrics;
  algo_b_metrics: AlgorithmMetrics;
  recommendation: 'algo_a' | 'algo_b' | 'insufficient_data' | 'no_difference';
  confidence_level?: number;
}

export interface CalibrationPoint {
  predicted_prob_bin: number;
  actual_win_rate: number;
  sample_count: number;
  confidence_interval_low?: number;
  confidence_interval_high?: number;
}

export interface PerformanceBucket {
  bucket_start: number;
  bucket_end: number;
  bet_count: number;
  win_rate: number;
  roi: number;
  clv_avg: number;
  avg_odds: number;
}
