// @vitest-environment jsdom
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { cleanup, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { edgeApi } from '@/lib/edgeApi';
import type { BlindPlayData, BlindRecord } from '@/types/edge';
import { BlindList } from './BlindList';
import { BlindPlay } from './BlindPlay';

afterEach(() => { cleanup(); vi.restoreAllMocks(); });

const play = (over: Partial<BlindPlayData> = {}): BlindPlayData => ({
  sport: 'nba', n: 7, slate_date: '2026-11-16', event_id: 'ev1', player: 'Jalen Brunson', team: 'New York Knicks',
  home_team: 'New York Knicks', away_team: 'Boston Celtics', bet: 'Jalen Brunson Over 27.5 Points',
  market: 'player_points', market_label: 'Points', side: 'over', line: 27.5, book: 'draftkings', price: 110,
  fair_prob: 0.55, ev: 0.06, n_books: 6, form: { hits: 7, games: 10, avg: 29.1, avg_games: 10 }, position: 'PG',
  panel: { favorable: true, opp_vs_pos_rank: 27, drtg_rank: 22, pace_rank: 6, usage_l10: 31.2, usage_change: 1.5,
    minutes_l5: 35.4 },
  commence_time: '2026-11-17T00:30:00+00:00',
  explanation: '6 books price this at 55%; DraftKings pays +110, which needs only 48%.',
  result: null, profit_units: null, headshot_url: 'https://a.espncdn.com/x.png', home_logo_url: null,
  away_logo_url: null, ...over,
});
const record: BlindRecord = { n: 3, wins: 1, losses: 1, pushes: 0, voids: 1, units: 0.1 };

const wrap = (path: string) => {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <MemoryRouter initialEntries={[path]}>
        <Routes>
          <Route path="/blind" element={<BlindList />} />
          <Route path="/blind/:sport/:n" element={<BlindPlay />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>);
};

describe('BlindPlay', () => {
  it('reveals the name, exact ranks and the explanation', async () => {
    vi.spyOn(edgeApi, 'blindPlay').mockResolvedValue(play());
    vi.spyOn(edgeApi, 'ev').mockResolvedValue([]);
    wrap('/blind/nba/7');
    await waitFor(() => expect(screen.getByRole('heading', { name: /jalen brunson/i })).toBeTruthy());
    expect(screen.getByText(/6 books price this at 55%/)).toBeTruthy();
    expect(screen.getByText('27th most allowed')).toBeTruthy();
    expect(screen.getByText('6th fastest')).toBeTruthy();
    expect(screen.getByRole('link', { name: /odds screen/i }).getAttribute('href'))
      .toBe('/edge/odds/ev1?market=player_points');
  });

  it('shows void for a voided play', async () => {
    vi.spyOn(edgeApi, 'blindPlay').mockResolvedValue(play({ result: 'void', profit_units: 0 }));
    vi.spyOn(edgeApi, 'ev').mockResolvedValue([]);
    wrap('/blind/nba/7');
    await waitFor(() => expect(screen.getByText(/void/i)).toBeTruthy());
  });

  it('shows a friendly not-found state on 404', async () => {
    vi.spyOn(edgeApi, 'blindPlay').mockRejectedValue(new Error('404'));
    wrap('/blind/nba/99');
    await waitFor(() => expect(screen.getByText(/couldn't find that play/i)).toBeTruthy());
  });

  it('signs visitors up with source blind_reveal', async () => {
    vi.spyOn(edgeApi, 'blindPlay').mockResolvedValue(play());
    vi.spyOn(edgeApi, 'ev').mockResolvedValue([]);
    const spy = vi.spyOn(edgeApi, 'signup').mockResolvedValue(undefined);
    wrap('/blind/nba/7');
    await waitFor(() => screen.getByLabelText(/email address/i));
    const { fireEvent } = await import('@testing-library/react');
    fireEvent.change(screen.getByLabelText(/email address/i), { target: { value: 'fan@example.com' } });
    fireEvent.click(screen.getByRole('button', { name: /notify me/i }));
    await waitFor(() => expect(spy).toHaveBeenCalledWith('fan@example.com', 'blind_reveal', ''));
  });
});

describe('BlindList', () => {
  it('lists losses and voids alongside wins, with the record always visible', async () => {
    vi.spyOn(edgeApi, 'blind').mockResolvedValue({
      record, plays: [
        play({ n: 9, player: 'Win Guy', result: 'win', profit_units: 1.1 }),
        play({ n: 8, player: 'Loss Guy', result: 'loss', profit_units: -1 }),
        play({ n: 7, player: 'Void Guy', result: 'void', profit_units: 0 }),
      ],
    });
    wrap('/blind');
    await waitFor(() => expect(screen.getByText('Loss Guy')).toBeTruthy());
    expect(screen.getByText('Win Guy')).toBeTruthy();
    expect(screen.getByText('Void Guy')).toBeTruthy();
    expect(screen.getByText(/1-1-0/)).toBeTruthy();
    expect(screen.getByText(/1 void/i)).toBeTruthy();
  });
});
