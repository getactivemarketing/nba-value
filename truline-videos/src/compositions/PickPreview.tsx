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
  musicFile?: string;
}

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
      {entries.filter(([k]) => k !== 'team').map(([key, value]) => (
        <div key={key} style={{
          textAlign: 'center',
          fontFamily: key === 'price' || key === 'stat' ? FONTS.mono : FONTS.display,
          fontWeight: key.endsWith('label') || key === 'disclaimer' ? 500 : 800,
          fontSize: key === 'price' || key === 'stat' ? 150
            : key.endsWith('label') || key === 'disclaimer' ? 36 : 64,
          color: key === 'price' || key === 'stat' ? COLORS.accent
            : key.endsWith('label') || key === 'disclaimer' ? COLORS.muted : COLORS.text,
          letterSpacing: '-0.02em', lineHeight: 1.15, marginBottom: 18,
        }}>
          {value}
        </div>
      ))}
    </AbsoluteFill>
  );
};

export const PickPreview: React.FC<PickPreviewProps> = ({
  beats, teamColor, logoUrl, brollSrc, musicFile,
}) => {
  const { durationInFrames } = useVideoConfig();
  let cursor = 0;

  return (
    <AbsoluteFill style={{ backgroundColor: COLORS.bg }}>
      {brollSrc && (
        <AbsoluteFill style={{ opacity: 0.15 }}>
          <Loop durationInFrames={durationInFrames}>
            <OffthreadVideo src={brollSrc} muted
                            style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
          </Loop>
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
