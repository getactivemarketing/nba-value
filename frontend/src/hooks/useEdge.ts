// frontend/src/hooks/useEdge.ts
import { useQuery } from '@tanstack/react-query';
import { edgeApi, type EvParams } from '@/lib/edgeApi';
import type { Sport } from '@/lib/edgeFormat';

export const useEdgeGames = (sport: Sport = 'nba') =>
  useQuery({ queryKey: ['edge', sport, 'games'], queryFn: () => edgeApi.games(sport), refetchInterval: 60000 });
export const useEdgeEv = (params: EvParams, sport: Sport = 'nba') =>
  useQuery({ queryKey: ['edge', sport, 'ev', params], queryFn: () => edgeApi.ev(params, sport), refetchInterval: 30000 });
export const useEdgeOdds = (eventId: string, market: string, sport: Sport = 'nba') =>
  useQuery({ queryKey: ['edge', sport, 'odds', eventId, market], queryFn: () => edgeApi.odds(eventId, market, sport),
    refetchInterval: 30000, enabled: !!eventId });
export const useEdgePlayer = (id: number, market: string, line?: number, minMinutes?: number, sport: Sport = 'nba') =>
  useQuery({ queryKey: ['edge', sport, 'player', id, market, line, minMinutes],
    queryFn: () => edgeApi.player(id, market, line, minMinutes, sport), enabled: id > 0 });
export const useEdgeResults = (days = 365, sport: Sport = 'nba') =>
  useQuery({ queryKey: ['edge', sport, 'results', days], queryFn: () => edgeApi.results(days, sport), staleTime: 300000 });
export const useEdgeHealth = (sport: Sport = 'nba') =>
  useQuery({ queryKey: ['edge', sport, 'health'], queryFn: () => edgeApi.health(sport), refetchInterval: 60000 });
