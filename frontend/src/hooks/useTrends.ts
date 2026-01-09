import { useQuery } from '@tanstack/react-query';
import {
  api,
  type ATSTeam,
  type OUTeam,
  type SituationalTrends,
  type ModelPerformance,
} from '@/lib/api';

export function useATSLeaderboard() {
  return useQuery<ATSTeam[]>({
    queryKey: ['trends', 'ats'],
    queryFn: () => api.getATSLeaderboard(),
    staleTime: 300000, // 5 minutes - trends don't change often
  });
}

export function useOULeaderboard() {
  return useQuery<OUTeam[]>({
    queryKey: ['trends', 'ou'],
    queryFn: () => api.getOULeaderboard(),
    staleTime: 300000,
  });
}

export function useSituationalTrends() {
  return useQuery<SituationalTrends>({
    queryKey: ['trends', 'situational'],
    queryFn: () => api.getSituationalTrends(),
    staleTime: 300000,
  });
}

export function useModelPerformance(days: number = 30) {
  return useQuery<ModelPerformance>({
    queryKey: ['trends', 'model', days],
    queryFn: () => api.getModelPerformance(days),
    staleTime: 300000,
  });
}
