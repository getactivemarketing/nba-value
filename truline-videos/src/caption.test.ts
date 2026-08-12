import { describe, expect, it } from 'vitest';
import { buildPreviewCaption } from './caption';

describe('buildPreviewCaption — with a turn beat', () => {
  const caption = buildPreviewCaption({
    teamName: 'Chicago White Sox',
    oddsAmerican: 155,
    turnNarration: 'But Castillo has held opponents scoreless in the first in 10 of 17 starts.',
  });

  it('keeps the blank line between every block — not one run-on paragraph', () => {
    expect(caption).toBe(
      'Chicago White Sox ML +155.\n'
      + '\n'
      + 'But Castillo has held opponents scoreless in the first in 10 of 17 starts.\n'
      + '\n'
      + 'Not betting advice. 21+.\n'
      + '#MLB #SportsAnalytics',
    );
  });

  it('separates the price, the turn and the disclaimer block with real paragraph breaks', () => {
    // Three blocks => exactly two blank-line separators.
    expect(caption.split('\n\n')).toHaveLength(3);
  });

  it('never collapses to a single line', () => {
    expect(caption).toContain('\n\n');
  });

  it('signs a negative price without a plus', () => {
    expect(buildPreviewCaption({ teamName: 'Dodgers', oddsAmerican: -140 }))
      .toMatch(/^Dodgers ML -140\.$/m);
  });
});

describe('buildPreviewCaption — without a turn beat', () => {
  // The backend drops the turn beat when the first-inning split is too thin
  // (MIN_TURN_BEAT_STARTS), so this is a normal slate, not an error.
  const caption = buildPreviewCaption({ teamName: 'Marlins', oddsAmerican: 210 });

  it('leaves no double blank line and no dangling gap where the turn would be', () => {
    expect(caption).toBe(
      'Marlins ML +210.\n'
      + '\n'
      + 'Not betting advice. 21+.\n'
      + '#MLB #SportsAnalytics',
    );
  });

  it('has no run of three or more newlines anywhere', () => {
    expect(caption).not.toMatch(/\n{3}/);
  });

  it('drops the block for an explicitly null or blank turn narration too', () => {
    const fromNull = buildPreviewCaption({ teamName: 'Marlins', oddsAmerican: 210, turnNarration: null });
    const fromBlank = buildPreviewCaption({ teamName: 'Marlins', oddsAmerican: 210, turnNarration: '   ' });
    expect(fromNull).toBe(caption);
    expect(fromBlank).toBe(caption);
  });
});

describe('buildPreviewCaption — contract copy', () => {
  it('always carries the exact disclaimer and the hashtags on their own line', () => {
    const caption = buildPreviewCaption({ teamName: 'Cubs', oddsAmerican: 120, turnNarration: 'x' });
    expect(caption.endsWith('Not betting advice. 21+.\n#MLB #SportsAnalytics')).toBe(true);
  });

  it('does not leak the banned word from a well-formed input', () => {
    const caption = buildPreviewCaption({ teamName: 'Cubs', oddsAmerican: 120, turnNarration: 'x' });
    expect(caption.toLowerCase()).not.toContain('edge');
  });
});
