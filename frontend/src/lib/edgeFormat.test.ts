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

import { edgeBase, marketKeysFor } from './edgeFormat';

describe('sports', () => {
  it('labels NFL markets', () => {
    expect(marketLabel('player_rush_reception_yds')).toBe('Rush+Rec Yds');
    expect(marketLabel('player_pass_interceptions')).toBe('INTs');
    expect(marketLabel('player_anytime_td')).toBe('Anytime TD');
  });
  it('lists each sport\'s own markets, lines first', () => {
    expect(marketKeysFor('nfl')).toEqual(['h2h', 'spreads', 'totals', 'player_pass_yds', 'player_pass_tds',
      'player_pass_completions', 'player_pass_attempts', 'player_pass_interceptions', 'player_rush_yds',
      'player_rush_attempts', 'player_receptions', 'player_reception_yds', 'player_rush_reception_yds',
      'player_anytime_td']);
    expect(marketKeysFor('nba')).toContain('player_points');
    expect(marketKeysFor('nba')).not.toContain('player_pass_yds');
  });
  it('routes each sport under its own base', () => {
    expect(edgeBase('nba')).toBe('/edge');
    expect(edgeBase('nfl')).toBe('/nfl/edge');
  });
});

import { propMarketsFor } from './edgeFormat';
describe('prop markets', () => {
  it('lists only player markets per sport', () => {
    expect(propMarketsFor('nfl')).toHaveLength(11); // incl. player_anytime_td
    expect(propMarketsFor('nfl').every((m) => m.startsWith('player_'))).toBe(true);
    expect(propMarketsFor('nba')).toContain('player_points');
  });
});
