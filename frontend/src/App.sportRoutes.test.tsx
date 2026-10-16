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
});
