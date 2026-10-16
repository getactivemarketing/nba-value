// frontend/src/hooks/useEdge.ts
import { useQuery } from '@tanstack/react-query';
import { edgeApi, type EvParams } from '@/lib/edgeApi';

export const useEdgeGames = () =>
  useQuery({ queryKey: ['edge', 'games'], queryFn: edgeApi.games, refetchInterval: 60000 });
export const useEdgeEv = (params: EvParams) =>
  useQuery({ queryKey: ['edge', 'ev', params], queryFn: () => edgeApi.ev(params), refetchInterval: 30000 });
export const useEdgeOdds = (eventId: string, market: string) =>
  useQuery({ queryKey: ['edge', 'odds', eventId, market], queryFn: () => edgeApi.odds(eventId, market),
    refetchInterval: 30000, enabled: !!eventId });
export const useEdgePlayer = (id: number, market: string, line?: number, minMinutes?: number) =>
  useQuery({ queryKey: ['edge', 'player', id, market, line, minMinutes],
    queryFn: () => edgeApi.player(id, market, line, minMinutes), enabled: id > 0 });
export const useEdgeResults = (days = 365) =>
  useQuery({ queryKey: ['edge', 'results', days], queryFn: () => edgeApi.results(days), staleTime: 300000 });
export const useEdgeHealth = () =>
  useQuery({ queryKey: ['edge', 'health'], queryFn: edgeApi.health, refetchInterval: 60000 });
