/**
 * The backend↔composition contract test.
 *
 * The two halves of this pipeline are a Python service and a React
 * composition with no shared type. Nothing forces them to agree, and the
 * composition renders whatever overlay object it is handed — so drift shows
 * up as a blank screen in a published video, not as an error. This test is
 * the only thing that catches it.
 */
import { describe, expect, it, vi } from 'vitest';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';

vi.mock('remotion', async () => await import('./test-support/remotion-stub'));

import {
  ALWAYS_EMITTED_BEATS, BEAT_OVERLAY_CONTRACT, CONTRACT_BEAT_SAMPLES,
  isContractBeatKey, type ContractBeatKey,
} from './pick-preview-contract';
import { PICK_PREVIEW_DEFAULT_PROPS } from './pick-preview-defaults';
import { PickPreview, beatTextVariant, type BeatClip } from './compositions/PickPreview';

const contractKeys = Object.keys(BEAT_OVERLAY_CONTRACT) as ContractBeatKey[];

const renderBeats = (beats: BeatClip[]): string =>
  renderToStaticMarkup(React.createElement(PickPreview, {
    beats, teamColor: '#27251F', logoUrl: 'https://a.espncdn.com/x.png',
  }));

const clip = (key: string, overlay: Record<string, string>): BeatClip =>
  ({ key, overlay, audioSrc: `/public/${key}.mp3`, durationInFrames: 60 });

describe('PICK_PREVIEW_DEFAULT_PROPS conforms to what build_beats() emits', () => {
  it('every default beat key is one the backend can actually produce', () => {
    for (const beat of PICK_PREVIEW_DEFAULT_PROPS.beats) {
      expect(isContractBeatKey(beat.key), `unknown beat key "${beat.key}"`).toBe(true);
    }
  });

  it.each(contractKeys)('the %s beat\'s overlay keys are exactly the contracted set', (key) => {
    const beat = PICK_PREVIEW_DEFAULT_PROPS.beats.find((b) => b.key === key);
    expect(beat, `no default beat for "${key}"`).toBeDefined();
    expect(Object.keys(beat!.overlay).sort())
      .toEqual([...BEAT_OVERLAY_CONTRACT[key]].sort());
  });

  it('no default overlay is empty — an empty overlay renders as a blank screen', () => {
    for (const beat of PICK_PREVIEW_DEFAULT_PROPS.beats) {
      expect(Object.keys(beat.overlay).length,
        `beat "${beat.key}" has an empty overlay; the backend never emits one`,
      ).toBeGreaterThan(0);
    }
  });

  it('the hook beat carries the hook sentence, the way the backend writes it', () => {
    const hook = PICK_PREVIEW_DEFAULT_PROPS.beats.find((b) => b.key === 'hook');
    expect(Object.keys(hook!.overlay)).toEqual(['line']);
    expect(hook!.overlay.line.length).toBeGreaterThan(10);
  });

  it('covers the always-emitted beats, so the Studio preview is a real video', () => {
    const present = PICK_PREVIEW_DEFAULT_PROPS.beats.map((b) => b.key);
    for (const key of ALWAYS_EMITTED_BEATS) expect(present).toContain(key);
  });
});

describe('PickPreview renders every declared beat shape', () => {
  it.each(contractKeys)('renders the %s beat without throwing, and shows its copy', (key) => {
    const html = renderBeats([clip(key, CONTRACT_BEAT_SAMPLES[key])]);
    for (const [k, value] of Object.entries(CONTRACT_BEAT_SAMPLES[key])) {
      if (beatTextVariant(k) === 'logo') {
        expect(html, `beat "${key}" should render a logo for overlay.${k}`).toContain('<img');
      } else {
        // A conforming beat must put something on screen. The failure this
        // guards is silent: an overlay the composition draws nothing for
        // becomes seconds of blank video with narration over it.
        expect(html, `beat "${key}" rendered nothing for overlay.${k}`)
          .toContain(value.replace(/&/g, '&amp;').replace(/'/g, '&#x27;'));
      }
    }
  });

  it('renders a whole slate — every contract beat at once — without throwing', () => {
    const html = renderBeats(contractKeys.map((k) => clip(k, CONTRACT_BEAT_SAMPLES[k])));
    expect((html.match(/data-remotion="Sequence"/g) || []).length).toBe(contractKeys.length);
  });

  it('renders the default props (what Studio shows) without throwing', () => {
    expect(() => renderBeats(PICK_PREVIEW_DEFAULT_PROPS.beats)).not.toThrow();
  });
});

describe('overlay keys the composition has no styling branch for', () => {
  /**
   * `BeatText` styles by key NAME, and anything it does not recognise falls
   * through to generic 64px white display text. That fallback is not wrong,
   * but it is invisible — so pin the exact set that relies on it. A new
   * backend key landing here fails this test, which is the prompt to decide
   * whether it deserves its own treatment rather than to discover it on
   * TikTok.
   */
  const byVariant = (variant: string) => contractKeys
    .flatMap((k) => [...BEAT_OVERLAY_CONTRACT[k]] as string[])
    .filter((key, i, all) => all.indexOf(key) === i)
    .filter((key) => beatTextVariant(key) === variant)
    .sort();

  it('the unstyled (generic body text) keys are exactly these', () => {
    expect(byVariant('body')).toEqual(['chips', 'cta', 'line', 'market', 'model']);
  });

  it('the big mono figures are exactly these', () => {
    expect(byVariant('figure')).toEqual(['price', 'stat']);
  });

  it('the muted small captions are exactly these', () => {
    expect(byVariant('caption')).toEqual(['disclaimer', 'label', 'market_label', 'model_label']);
  });

  it('only the team key becomes a logo', () => {
    expect(byVariant('logo')).toEqual(['team']);
  });
});
