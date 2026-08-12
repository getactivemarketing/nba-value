import { describe, expect, it, vi } from 'vitest';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';

vi.mock('remotion', async () => await import('../test-support/remotion-stub'));

import {
  brollLoopFrames, calculatePickPreviewMetadata, PickPreview, type BeatClip,
} from './PickPreview';
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

describe('brollLoopFrames — one iteration, not the whole video', () => {
  it('is the clip\'s own length', () => {
    expect(brollLoopFrames(180)).toBe(180);
  });

  it('rounds a fractional measurement to a whole frame', () => {
    expect(brollLoopFrames(179.4)).toBe(179);
  });

  it('rounds a sub-frame clip up to one frame rather than to zero', () => {
    expect(brollLoopFrames(0.6)).toBe(1);
  });

  it('refuses to loop when the duration is unknown', () => {
    expect(brollLoopFrames(undefined)).toBeNull();
  });

  it('refuses to loop on zero, a negative, NaN or Infinity — <Loop> throws on all of them', () => {
    for (const bad of [0, 0.4, -1, -300, NaN, Infinity, -Infinity]) {
      expect(brollLoopFrames(bad), `brollLoopFrames(${bad})`).toBeNull();
    }
  });
});

describe('b-roll looping in the rendered tree', () => {
  const render = (over: Record<string, unknown>) => renderToStaticMarkup(
    React.createElement(PickPreview, {
      beats: [{ key: 'hook', overlay: { line: 'x' }, audioSrc: '', durationInFrames: 60 }],
      teamColor: '#000', logoUrl: '', ...over,
    } as never),
  );

  it('loops on the CLIP length, not the composition length', () => {
    // The composition is 435 frames (see the stubbed useVideoConfig) and the
    // clip is 180. Passing 435 would be exactly one iteration — i.e. the clip
    // freezing on its last frame for the rest of the video, which is the bug
    // <Loop> was added to fix.
    const html = render({ brollSrc: '/public/b.mp4', brollDurationInFrames: 180 });
    expect(html).toContain('data-loop-frames="180"');
    expect(html).not.toContain('data-loop-frames="435"');
  });

  it('plays the clip once, un-looped, when the duration could not be measured', () => {
    const html = render({ brollSrc: '/public/b.mp4' });
    expect(html).toContain('data-remotion="OffthreadVideo"');
    expect(html).not.toContain('data-remotion="Loop"');
  });

  it('does not crash the render on a zero or negative measurement', () => {
    for (const bad of [0, -30, NaN]) {
      expect(() => render({ brollSrc: '/public/b.mp4', brollDurationInFrames: bad })).not.toThrow();
    }
  });

  it('renders no b-roll at all when there is no clip', () => {
    const html = render({ brollDurationInFrames: 180 });
    expect(html).not.toContain('data-remotion="OffthreadVideo"');
    expect(html).not.toContain('data-remotion="Loop"');
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
