import { describe, expect, it } from 'vitest';
import { calculatePickPreviewMetadata, type BeatClip } from './PickPreview';

const beat = (durationInFrames: number): BeatClip => ({
  key: 'x', overlay: {}, audioSrc: 'a.mp3', durationInFrames,
});

describe('calculatePickPreviewMetadata', () => {
  it('total duration is the sum of beat durations', () => {
    const props = { beats: [beat(60), beat(90), beat(30)], teamColor: '#000', logoUrl: '' };
    expect(calculatePickPreviewMetadata({ props }).durationInFrames).toBe(180);
  });

  it('a longer narration lengthens only its own beat', () => {
    const base = { beats: [beat(60), beat(90)], teamColor: '#000', logoUrl: '' };
    const longer = { beats: [beat(60), beat(150)], teamColor: '#000', logoUrl: '' };
    expect(calculatePickPreviewMetadata({ props: longer }).durationInFrames
      - calculatePickPreviewMetadata({ props: base }).durationInFrames).toBe(60);
  });

  it('never returns zero frames, which would fail the render', () => {
    const props = { beats: [], teamColor: '#000', logoUrl: '' };
    expect(calculatePickPreviewMetadata({ props }).durationInFrames).toBeGreaterThan(0);
  });

  it('a beat with zero duration does not crash the render', () => {
    const props = { beats: [beat(0), beat(90)], teamColor: '#000', logoUrl: '' };
    const result = calculatePickPreviewMetadata({ props });
    expect(result.durationInFrames).toBeGreaterThan(0);
  });

  it('default props produce a real video duration, not a still', () => {
    const defaultBeats: BeatClip[] = [
      { key: 'hook', overlay: {}, audioSrc: '', durationInFrames: 60 },
      { key: 'pick', overlay: { team: 'CWS', price: '-110', priceLabel: 'Moneyline' }, audioSrc: '', durationInFrames: 90 },
      { key: 'turn', overlay: { stat: '4.2% Edge', statLabel: 'Model Edge' }, audioSrc: '', durationInFrames: 75 },
      { key: 'numbers', overlay: { number: '71.3%', numberLabel: 'Win Probability' }, audioSrc: '', durationInFrames: 75 },
      { key: 'close', overlay: { cta: 'Follow for Updates', disclaimer: 'Not investment advice' }, audioSrc: '', durationInFrames: 60 },
    ];
    const props = { beats: defaultBeats, teamColor: '#27251F', logoUrl: 'https://a.espncdn.com/i/teamlogos/mlb/500/chw.png' };
    const result = calculatePickPreviewMetadata({ props });
    expect(result.durationInFrames).toBe(360);
    expect(result.durationInFrames).toBeGreaterThan(1);
  });
});
