export interface PerformanceBucket {
  bucket: string;
  bet_count: number;
  wins: number;
  losses: number;
  win_rate: number | null;
  roi: number | null;
  profit: number;
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
