// frontend/src/lib/edgeFormat.test.ts
import { describe, expect, it } from 'vitest';
import { bookLabel, fmtAge, fmtAmerican, fmtEv, fmtPct, marketLabel, outcomeLabel } from './edgeFormat';

describe('edgeFormat', () => {
  it('formats American odds with a sign', () => {
    expect(fmtAmerican(150)).toBe('+150');
    expect(fmtAmerican(-110)).toBe('-110');
    expect(fmtAmerican(null)).toBe('—');
  });
  it('formats EV and percentages', () => {
    expect(fmtEv(0.0432)).toBe('+4.3%');
    expect(fmtPct(0.5)).toBe('50%');
    expect(fmtPct(null)).toBe('—');
  });
  it('formats ages', () => {
    const now = new Date('2026-10-20T18:00:00Z');
    expect(fmtAge('2026-10-20T17:59:30Z', now)).toBe('30s');
    expect(fmtAge('2026-10-20T17:55:00Z', now)).toBe('5m');
  });
  it('labels books and markets', () => {
    expect(bookLabel('williamhill_us')).toBe('Caesars');
    expect(bookLabel('unknown_book')).toBe('unknown_book');
    expect(marketLabel('player_points_rebounds_assists')).toBe('Pts+Reb+Ast');
  });
});

describe('outcomeLabel', () => {
  const ev = { home_team: 'Detroit Pistons', away_team: 'Boston Celtics' };
  it('shows the away spread from the away side', () => {
    expect(outcomeLabel('spreads', 'away', -5.5, ev)).toBe('Boston Celtics +5.5');
    expect(outcomeLabel('spreads', 'home', -5.5, ev)).toBe('Detroit Pistons -5.5');
  });
  it('names moneyline teams and totals/props sides', () => {
    expect(outcomeLabel('h2h', 'home', 0, ev)).toBe('Detroit Pistons');
    expect(outcomeLabel('totals', 'over', 221.5, ev)).toBe('Over 221.5');
    expect(outcomeLabel('player_points', 'under', 27.5, ev)).toBe('Under 27.5');
  });
});
