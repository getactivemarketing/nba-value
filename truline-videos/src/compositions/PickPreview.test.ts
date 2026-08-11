import { describe, expect, it } from 'vitest';
import { calculatePickPreviewMetadata, type BeatClip } from './PickPreview';
import { PICK_PREVIEW_DEFAULT_PROPS } from '../pick-preview-defaults';

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
    const result = calculatePickPreviewMetadata({ props: PICK_PREVIEW_DEFAULT_PROPS });
    expect(result.durationInFrames).toBeGreaterThan(1);
  });
});

describe('Narration contract guard — copy must match backend constraints', () => {
  it('no default beat overlay contains "edge" (case-insensitive)', () => {
    for (const beat of PICK_PREVIEW_DEFAULT_PROPS.beats) {
      for (const [key, value] of Object.entries(beat.overlay)) {
        expect(value.toLowerCase(),
          `Beat "${beat.key}" overlay[${key}] must not contain "edge": ${value}`
        ).not.toContain('edge');
      }
    }
  });

  it('close beat disclaimer is exactly "Not betting advice. 21+."', () => {
    const closeBeat = PICK_PREVIEW_DEFAULT_PROPS.beats.find(b => b.key === 'close');
    expect(closeBeat).toBeDefined();
    expect(closeBeat?.overlay.disclaimer).toBe('Not betting advice. 21+.');
  });
});
