// @vitest-environment jsdom
import { cleanup, render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, describe, expect, it } from 'vitest';
import { NightlyCard } from './NightlyCard';
import type { NightlyLeg, NightlyResponse } from '@/types/edge';

afterEach(cleanup);

const leg = (o: Partial<NightlyLeg>): NightlyLeg => ({
  event_id: 'g1', matchup: 'Boston Celtics @ Toronto Raptors', commence_time: '2026-11-12T00:10:00+00:00',
  market: 'player_rebounds', market_label: 'Rebounds', player: 'Jayson Tatum', side: 'under', line: 7.5,
  bet: 'Jayson Tatum Under 7.5 Rebounds', book: 'betmgm', price: -115, fair_prob: 0.56, ev: 0.04, n_books: 10,
  form: { hits: 8, games: 10, rate: 0.8 }, result: null, profit_units: null, ...o,
});

const base: NightlyResponse = {
  slate_date: '2026-11-11', locks_at: '2026-11-11T23:10:00+00:00', tonight: null, history: [],
  record: { top: { n: 0, wins: 0, losses: 0, units: 0 }, parlay: { n: 0, wins: 0, units: 0 } },
};

const show = (data: NightlyResponse) => render(<MemoryRouter><NightlyCard data={data} /></MemoryRouter>);

describe('NightlyCard', () => {
  it('says when the picks lock before they are locked', () => {
    show(base);
    expect(screen.getByText(/lock an hour before the first game/i)).toBeTruthy();
  });

  it('lists the top picks with their L10 and the parlay with its hit chance', () => {
    const legs = [leg({}), leg({ event_id: 'g2', player: 'Al Horford', bet: 'Al Horford Under 0.5 Threes', price: 200 })];
    show({ ...base, tonight: { slate_date: '2026-11-11', locked_at: '2026-11-11T23:10:00+00:00', top: legs,
      parlay: { book: 'betmgm', legs, decimal: 5.6, american: 460, hit_prob: 0.168, ev: 0.04, result: null, profit_units: null } } });
    expect(screen.getAllByText('Jayson Tatum Under 7.5 Rebounds').length).toBe(2);
    expect(screen.getAllByText('8/10').length).toBeGreaterThan(0);
    expect(screen.getByText('+460')).toBeTruthy();
    expect(screen.getByText(/17% chance it hits/)).toBeTruthy();
    expect(screen.getByText(/2-leg parlay at BetMGM/)).toBeTruthy();
  });

  it('shows results and the running record', () => {
    show({ ...base, tonight: { slate_date: '2026-11-11', locked_at: '2026-11-11T23:10:00+00:00',
      top: [leg({ result: 'win', profit_units: 0.87 })], parlay: null },
      record: { top: { n: 20, wins: 12, losses: 8, units: 3.4 }, parlay: { n: 4, wins: 1, units: 18.2 } } });
    expect(screen.getByText('W')).toBeTruthy();
    expect(screen.getByText(/12-8 \(\+3\.4u\)/)).toBeTruthy();
    expect(screen.getByText(/1 of 4 hit \(\+18\.2u\)/)).toBeTruthy();
  });
});
