// frontend/src/lib/edgeApi.ts
import axios from 'axios';
import type { EdgeGame, EvRow, OddsResponse, PlayerResponse, ResultsResponse } from '@/types/edge';

const client = axios.create({ baseURL: import.meta.env.VITE_API_URL || '/api/v1' });

export interface EvParams { market?: string; book?: string; min_ev?: number; event_id?: string; }

export const edgeApi = {
  async games(): Promise<EdgeGame[]> {
    return (await client.get<{ games: EdgeGame[] }>('/edge/games')).data.games;
  },
  async ev(params: EvParams = {}): Promise<EvRow[]> {
    return (await client.get<{ rows: EvRow[] }>('/edge/ev', { params })).data.rows;
  },
  async odds(eventId: string, market: string): Promise<OddsResponse> {
    return (await client.get<OddsResponse>(`/edge/odds/${eventId}`, { params: { market } })).data;
  },
  async player(id: number, market: string, line?: number, minMinutes?: number): Promise<PlayerResponse> {
    return (await client.get<PlayerResponse>(`/edge/players/${id}`,
      { params: { market, line, min_minutes: minMinutes } })).data;
  },
  async results(days = 365): Promise<ResultsResponse> {
    return (await client.get<ResultsResponse>('/edge/results', { params: { days } })).data;
  },
  async health(): Promise<{ delayed: boolean }> {
    return (await client.get<{ delayed: boolean }>('/edge/health')).data;
  },
};
