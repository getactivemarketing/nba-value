import React from 'react';
import {
  AbsoluteFill, Audio, Img, Loop, OffthreadVideo, Sequence,
  interpolate, spring, staticFile, useCurrentFrame, useVideoConfig,
} from 'remotion';
import { COLORS, FONTS, FPS } from '../constants';

export interface BeatClip {
  key: string;
  overlay: Record<string, string>;
  audioSrc: string;
  durationInFrames: number;
}

export interface PickPreviewProps {
  beats: BeatClip[];
  teamColor: string;
  logoUrl: string;
  brollSrc?: string;
  /**
   * Length of ONE b-roll clip, in frames — not the video's length. Remotion's
   * <Loop durationInFrames> is the length of a single iteration, so passing
   * the composition duration here would produce exactly one iteration (i.e.
   * no looping, and the clip freezing on its last frame once it runs out).
   * Optional because b-roll must never block a render: when it is missing or
   * unusable the clip simply plays once, un-looped.
   */
  brollDurationInFrames?: number;
  musicFile?: string;
}

/**
 * Iteration length for the b-roll <Loop>, or null to not loop at all.
 *
 * <Loop> validates its durationInFrames and THROWS on zero, a negative, NaN
 * or Infinity — which would take down the whole render over a background
 * video. An unmeasurable clip must degrade to "plays once", never to a crash.
 */
export const brollLoopFrames = (brollDurationInFrames?: number): number | null => {
  if (typeof brollDurationInFrames !== 'number') return null;
  if (!Number.isFinite(brollDurationInFrames)) return null;
  const rounded = Math.round(brollDurationInFrames);
  return rounded > 0 ? rounded : null;
};

/**
 * Duration is derived from the narration, never the reverse. Hardcoding beat
 * lengths and dropping audio in afterwards desyncs the moment a team name or a
 * stat reads longer than the template assumed.
 */
export const calculatePickPreviewMetadata = ({ props }: { props: PickPreviewProps }) => ({
  durationInFrames: Math.max(
    1,
    props.beats.reduce((sum, b) => sum + b.durationInFrames, 0),
  ),
});

/**
 * How one overlay key is drawn. There are only three branches, keyed off the
 * key NAME rather than any declared schema:
 *
 * - 'logo'    `team` — rendered as the team badge, never as text.
 * - 'figure'  `price`/`stat` — the one big mono number the beat is about.
 * - 'caption' anything ending in "label", plus `disclaimer` — muted and small.
 * - 'body'    everything else — 64px white display text.
 *
 * 'body' is the FALLBACK, not a decision: a key the backend starts emitting
 * that nothing here recognises silently lands in it and renders as generic
 * large white text. That is usually survivable and occasionally very wrong
 * (a disclaimer-ish string rendered at hero size), so the contract test
 * asserts the exact set of contract keys that fall through here — making a
 * new unstyled key a test failure to look at rather than a surprise on TikTok.
 */
export type BeatTextVariant = 'logo' | 'figure' | 'caption' | 'body';

export const beatTextVariant = (key: string): BeatTextVariant => {
  if (key === 'team') return 'logo';
  if (key === 'price' || key === 'stat') return 'figure';
  if (key.endsWith('label') || key === 'disclaimer') return 'caption';
  return 'body';
};

const VARIANT_STYLE: Record<Exclude<BeatTextVariant, 'logo'>, React.CSSProperties> = {
  figure: { fontFamily: FONTS.mono, fontWeight: 800, fontSize: 150, color: COLORS.accent },
  caption: { fontFamily: FONTS.display, fontWeight: 500, fontSize: 36, color: COLORS.muted },
  body: { fontFamily: FONTS.display, fontWeight: 800, fontSize: 64, color: COLORS.text },
};

const BeatText: React.FC<{ overlay: Record<string, string>; teamColor: string; logoUrl: string }> = ({
  overlay, teamColor, logoUrl,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const progress = spring({ frame, fps, config: { damping: 18, stiffness: 220, mass: 0.5 } });
  const opacity = interpolate(progress, [0, 1], [0, 1]);
  const scale = interpolate(progress, [0, 1], [0.6, 1]);

  const entries = Object.entries(overlay);

  return (
    <AbsoluteFill style={{
      justifyContent: 'center', alignItems: 'center', padding: 80,
      opacity, transform: `scale(${scale})`,
    }}>
      {overlay.team && logoUrl && (
        <Img src={logoUrl} width={280} height={280}
             style={{ filter: `drop-shadow(0 0 60px ${teamColor})`, marginBottom: 40 }} />
      )}
      {entries
        .filter(([k]) => beatTextVariant(k) !== 'logo')
        .map(([key, value]) => (
          <div key={key} style={{
            textAlign: 'center',
            ...VARIANT_STYLE[beatTextVariant(key) as Exclude<BeatTextVariant, 'logo'>],
            letterSpacing: '-0.02em', lineHeight: 1.15, marginBottom: 18,
          }}>
            {value}
          </div>
        ))}
    </AbsoluteFill>
  );
};

export const PickPreview: React.FC<PickPreviewProps> = ({
  beats, teamColor, logoUrl, brollSrc, brollDurationInFrames, musicFile,
}) => {
  let cursor = 0;

  const loopFrames = brollLoopFrames(brollDurationInFrames);
  const brollVideo = brollSrc ? (
    <OffthreadVideo src={brollSrc} muted
                    style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
  ) : null;

  return (
    <AbsoluteFill style={{ backgroundColor: COLORS.bg }}>
      {brollVideo && (
        <AbsoluteFill style={{ opacity: 0.15 }}>
          {loopFrames === null
            ? brollVideo
            : <Loop durationInFrames={loopFrames}>{brollVideo}</Loop>}
        </AbsoluteFill>
      )}

      <AbsoluteFill style={{
        background: `radial-gradient(circle at 50% 35%, ${teamColor}55 0%, transparent 65%)`,
      }} />

      {musicFile && <Audio src={staticFile(musicFile)} volume={0.15} />}

      {beats.map((beat) => {
        const clampedDuration = Math.max(1, beat.durationInFrames);
        const from = cursor;
        cursor += clampedDuration;
        return (
          <Sequence key={beat.key} from={from} durationInFrames={clampedDuration}>
            {beat.audioSrc && <Audio src={beat.audioSrc} />}
            <BeatText overlay={beat.overlay} teamColor={teamColor} logoUrl={logoUrl} />
          </Sequence>
        );
      })}
    </AbsoluteFill>
  );
};

export const PICK_PREVIEW_FPS = FPS;
