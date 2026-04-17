import { useQuery } from '@tanstack/react-query';
import { mlbApi } from '@/lib/mlbApi';

export function useMLBEvaluationSummary() {
  return useQuery({
    queryKey: ['mlb', 'evaluation', 'summary'],
    queryFn: () => mlbApi.getEvaluationSummary(),
    staleTime: 300000,
  });
}

export function useMLBDailyResults(days: number = 14) {
  return useQuery({
    queryKey: ['mlb', 'evaluation', 'daily', days],
    queryFn: () => mlbApi.getDailyEvaluation(days),
    staleTime: 300000,
  });
}

export function useNRFIEvaluation(days: number = 90) {
  return useQuery({
    queryKey: ['mlb', 'evaluation', 'nrfi', days],
    queryFn: () => mlbApi.getNRFIEvaluation(days),
    staleTime: 300000,
  });
}

export function useUnderdogEvaluation(days: number = 90) {
  return useQuery({
    queryKey: ['mlb', 'evaluation', 'underdogs', days],
    queryFn: () => mlbApi.getUnderdogEvaluation(days),
    staleTime: 300000,
  });
}
