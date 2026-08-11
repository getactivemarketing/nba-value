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
});
