/**
 * A minimal stand-in for the `remotion` module, so compositions can be
 * rendered with react-dom/server in a plain Node test.
 *
 * Why a full stub and not a partial mock: remotion ships as ONE bundled ESM
 * file, so its components call their own `useVideoConfig` internally and a
 * partial mock of just the hooks cannot reach them. Rendering a composition
 * for real therefore means a Remotion Studio or an actual render — neither of
 * which belongs in a unit test, and the render path here is publish-capable.
 *
 * What this buys: the composition's OWN logic — which beats become sequences,
 * which overlay keys become elements, whether <Loop> is used at all and with
 * what iteration length — is exercised exactly as written. What it cannot
 * catch is a change in Remotion's own behaviour; that is what the exact
 * version pin in package.json is for.
 *
 * Each stub marks itself with `data-remotion` so tests can assert on which
 * Remotion component was used.
 */
import React from 'react';

interface Kids { children?: React.ReactNode }

const box = (tag: string) =>
  ({ children, style }: Kids & { style?: React.CSSProperties }) =>
    React.createElement('div', { 'data-remotion': tag, style }, children);

export const AbsoluteFill = box('AbsoluteFill');

export const Sequence = ({ children, from, durationInFrames }: Kids & {
  from?: number; durationInFrames?: number;
}) => React.createElement('div', {
  'data-remotion': 'Sequence',
  'data-from': String(from ?? 0),
  'data-duration': String(durationInFrames ?? 0),
}, children);

/**
 * Mirrors the one behaviour of <Loop> a caller can get wrong: remotion
 * 4.0.448 runs `validateDurationInFrames` on it and THROWS for anything that
 * is not a positive finite number (see
 * node_modules/remotion/dist/cjs/loop/index.js). A composition that hands it
 * a zero, a negative or a NaN takes the whole render down.
 */
export const Loop = ({ children, durationInFrames }: Kids & { durationInFrames: number }) => {
  if (typeof durationInFrames !== 'number' || !Number.isFinite(durationInFrames) || durationInFrames <= 0) {
    throw new Error(
      `durationInFrames must be a positive number, but got ${durationInFrames}`,
    );
  }
  return React.createElement('div', {
    'data-remotion': 'Loop',
    'data-loop-frames': String(durationInFrames),
  }, children);
};

export const Audio = ({ src }: { src: string }) =>
  React.createElement('audio', { 'data-remotion': 'Audio', src });

export const Img = ({ src }: { src: string }) =>
  React.createElement('img', { 'data-remotion': 'Img', src });

export const OffthreadVideo = ({ src }: { src: string }) =>
  React.createElement('video', { 'data-remotion': 'OffthreadVideo', src });

export const staticFile = (path: string): string => `/public/${path}`;

export const interpolate = (
  _value: number, _input: readonly number[], output: readonly number[],
): number => output[output.length - 1];

export const spring = (): number => 1;

export const useCurrentFrame = (): number => 0;

export const useVideoConfig = () => ({
  fps: 30, width: 1080, height: 1920, durationInFrames: 435,
  id: 'pick-preview', defaultProps: {}, props: {},
});
