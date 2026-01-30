import { useQuery } from '@tanstack/react-query';
import { api, type LineMovementResponse } from '@/lib/api';

export function useLineMovement(gameId: string | null) {
  return useQuery<LineMovementResponse>({
    queryKey: ['lineMovement', gameId],
    queryFn: () => api.getLineMovement(gameId!),
    enabled: !!gameId,
    staleTime: 60000, // 1 minute
    refetchInterval: 120000, // Refresh every 2 minutes
  });
}

export type { LineMovementResponse };
