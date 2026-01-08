import { useQuery } from '@tanstack/react-query';
import { api } from '@/lib/api';
import type { Algorithm } from '@/types/market';

export function useAlgorithmComparison(
  startDate?: string,
  endDate?: string,
  marketType?: string
) {
  return useQuery({
    queryKey: ['evaluation', 'compare', startDate, endDate, marketType],
    queryFn: () => api.getAlgorithmComparison(startDate, endDate, marketType),
    staleTime: 300000, // 5 minutes - evaluation data doesn't change often
  });
}

export function useCalibrationCurve(marketType?: string, algorithm: Algorithm = 'a') {
  return useQuery({
    queryKey: ['evaluation', 'calibration', marketType, algorithm],
    queryFn: () => api.getCalibrationCurve(marketType, algorithm),
    staleTime: 300000,
  });
}

export function usePerformanceByBucket(
  algorithm: Algorithm = 'a',
  bucketType: 'score' | 'edge' | 'confidence' = 'score'
) {
  return useQuery({
    queryKey: ['evaluation', 'performance', algorithm, bucketType],
    queryFn: () => api.getPerformanceByBucket(algorithm, bucketType),
    staleTime: 300000,
  });
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

export function useDailyResults(
  days: number = 7,
  algorithm: Algorithm = 'b',
  minValue: number = 50
) {
  return useQuery({
    queryKey: ['evaluation', 'daily', days, algorithm, minValue],
    queryFn: () => api.getDailyResults(days, algorithm, minValue),
    staleTime: 300000,
  });
}

export interface PredictionResult {
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
  recent_predictions: PredictionResult[];
}

export function usePredictionPerformance(days: number = 14, minValue: number = 0) {
  return useQuery({
    queryKey: ['evaluation', 'predictions', days, minValue],
    queryFn: () => api.getPredictionPerformance(days, minValue),
    staleTime: 60000, // 1 minute - more frequent updates for live tracking
  });
}
