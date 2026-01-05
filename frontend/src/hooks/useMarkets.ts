import { useQuery } from '@tanstack/react-query';
import { api } from '@/lib/api';
import type { MarketFilters } from '@/types/market';

export function useMarkets(filters: Partial<MarketFilters> = {}) {
  return useQuery({
    queryKey: ['markets', filters],
    queryFn: () => api.getMarkets(filters),
    refetchInterval: 60000, // Refresh every minute
    staleTime: 30000,
  });
}

export function useLiveMarkets(algorithm: 'a' | 'b' = 'a') {
  return useQuery({
    queryKey: ['markets', 'live', algorithm],
    queryFn: () => api.getLiveMarkets(algorithm),
    refetchInterval: 30000, // Refresh every 30 seconds for live
    staleTime: 15000,
  });
}

export function useBetDetail(marketId: string) {
  return useQuery({
    queryKey: ['bet', marketId],
    queryFn: () => api.getBetDetail(marketId),
    enabled: !!marketId,
  });
}

export function useBetHistory(marketId: string, hours: number = 24) {
  return useQuery({
    queryKey: ['bet', marketId, 'history', hours],
    queryFn: () => api.getBetHistory(marketId, hours),
    enabled: !!marketId,
  });
}
