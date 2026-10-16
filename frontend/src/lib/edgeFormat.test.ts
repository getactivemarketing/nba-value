// frontend/src/lib/edgeFormat.test.ts
import { describe, expect, it } from 'vitest';
import { bookLabel, fmtAge, fmtAmerican, fmtEv, fmtPct, marketLabel } from './edgeFormat';

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
