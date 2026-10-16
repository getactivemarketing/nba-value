// @vitest-environment jsdom
// /edge and /nfl/edge render the same page components; React must not carry an
// NBA filter (market, game) into the NFL page or back.
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter, useNavigate, type NavigateFunction } from 'react-router-dom';
import { describe, expect, it, vi } from 'vitest';
import App from './App';
import { edgeApi } from '@/lib/edgeApi';

vi.mock('@/lib/edgeApi', () => ({
  edgeApi: {
    games: vi.fn(async () => []), ev: vi.fn(async () => []), odds: vi.fn(async () => ({ event: null, rows: [] })),
    player: vi.fn(), results: vi.fn(), health: vi.fn(async () => ({ delayed: false })),
  },
}));

// recharts' ResponsiveContainer needs ResizeObserver, which jsdom lacks.
globalThis.ResizeObserver ??= class { observe() {} unobserve() {} disconnect() {} } as any;

let nav: NavigateFunction;
function Nav() { nav = useNavigate(); return null; }

describe('sport routes', () => {
  it('does not carry the NBA market filter into the NFL +EV page', async () => {
    render(
      <QueryClientProvider client={new QueryClient()}>
        <MemoryRouter initialEntries={['/edge']}><Nav /><App /></MemoryRouter>
      </QueryClientProvider>,
    );
    const market = (await screen.findAllByRole('combobox'))[0] as HTMLSelectElement;
    fireEvent.change(market, { target: { value: 'player_points' } });
    await waitFor(() => expect(edgeApi.ev).toHaveBeenCalledWith(
      expect.objectContaining({ market: 'player_points' }), 'nba'));
    act(() => nav('/nfl/edge'));
    await screen.findByText('NFL +EV Finder');
    await waitFor(() => expect(edgeApi.ev).toHaveBeenLastCalledWith(
      expect.objectContaining({ market: undefined }), 'nfl'));
    expect(((await screen.findAllByRole('combobox'))[0] as HTMLSelectElement).value).toBe('');
  });
  it('serves the NFL player page and track record from NFL data', async () => {
    (edgeApi.player as any).mockResolvedValue({ bdl_player_id: 20, name: 'DJ Moore', team: 'CHI', position: 'WR',
      market: 'player_receptions', line: 4.5, opponent: 'ATL', is_home: false, hit_rates: null, games: [], prices: [] });
    (edgeApi.results as any).mockResolvedValue({ n: 0, clv: { measured: 0, bets: 0, mean_clv: null, median_clv: null,
      beat_close_rate: null, std_error: null, verdict: '', ci95: null }, record: { wins: 0, losses: 0, pushes: 0,
      voids: 0, win_rate: null, units: 0, roi: null }, by_market: {}, by_book: {}, daily_clv: [] });
    render(
      <QueryClientProvider client={new QueryClient()}>
        <MemoryRouter initialEntries={['/nfl/edge/player/20?market=player_receptions']}><Nav /><App /></MemoryRouter>
      </QueryClientProvider>,
    );
    await screen.findByText('DJ Moore');
    expect(edgeApi.player).toHaveBeenLastCalledWith(20, 'player_receptions', undefined, undefined, 'nfl');
    expect(screen.queryByText('All minutes')).toBeNull();          // no minutes filter on NFL
    act(() => nav('/nfl/edge/results'));
    await waitFor(() => expect(edgeApi.results).toHaveBeenLastCalledWith(365, 'nfl'));
  });
});
