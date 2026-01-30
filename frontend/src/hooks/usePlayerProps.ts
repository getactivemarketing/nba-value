import { useQuery } from '@tanstack/react-query';
import { api, type PlayerPropsResponse } from '@/lib/api';

export function usePlayerProps(gameId: string | null) {
  return useQuery<PlayerPropsResponse>({
    queryKey: ['playerProps', gameId],
    queryFn: () => api.getPlayerProps(gameId!),
    enabled: !!gameId,
    staleTime: 120000, // 2 minutes
    refetchInterval: 300000, // Refresh every 5 minutes (props don't change as often)
  });
}

export type { PlayerPropsResponse };
