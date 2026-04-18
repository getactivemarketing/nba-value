import React from 'react';
import {
  AbsoluteFill,
  Sequence,
  Img,
  Audio,
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig,
  staticFile,
} from 'remotion';
import { COLORS, FONTS, seconds, espnLogoUrl } from '../constants';

export interface ModelHitProps {
  winnerTeam: string;
  winnerName: string;
  oddsAmerican: number;
  profitUnits: number;
  scoreText: string;
  sport: 'mlb' | 'nba';
  teamColor: string;
  musicFile?: string;  // e.g. "music.mp3" in public/ dir — omit for no music
}

/**
 * Snappy animated text — higher stiffness = faster pop-in.
 */
const Pop: React.FC<{
  children: string;
  delay?: number;
  size?: number;
  color?: string;
  font?: string;
  weight?: number;
  y?: number;
}> = ({ children, delay = 0, size = 80, color = COLORS.text, font = FONTS.display, weight = 800, y }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const local = frame - delay;
  if (local < 0) return null;

  const progress = spring({
    frame: local,
    fps,
    config: { damping: 18, stiffness: 220, mass: 0.5 },
  });

  const scale = interpolate(progress, [0, 1], [0.4, 1]);
  const opacity = interpolate(progress, [0, 1], [0, 1]);

  return (
    <div
      style={{
        position: y !== undefined ? 'absolute' : 'relative',
        top: y,
        left: 0,
        right: 0,
        textAlign: 'center',
        fontSize: size,
        fontFamily: font,
        fontWeight: weight,
        color,
        opacity,
        transform: `scale(${scale})`,
        letterSpacing: '-0.02em',
      }}
    >
      {children}
    </div>
  );
};

/**
 * Animated profit counter that ticks up from 0 to target.
 */
const ProfitCounter: React.FC<{
  target: number;
  delay?: number;
}> = ({ target, delay = 0 }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const local = frame - delay;
  if (local < 0) return null;

  // Fast count-up over ~15 frames (0.5 sec)
  const progress = Math.min(local / 15, 1);
  const value = target * progress;
  const opacity = Math.min(local / 5, 1);

  return (
    <div
      style={{
        textAlign: 'center',
        fontSize: 72,
        fontFamily: FONTS.mono,
        fontWeight: 700,
        color: COLORS.green,
        opacity,
      }}
    >
      +{value.toFixed(2)}u
    </div>
  );
};

/**
 * MODEL HIT — 8-second celebration video (1080x1920 vertical).
 *
 * Timeline (30fps = 240 frames total):
 *   0.0-1.5s (0-44):   "MODEL HIT" + "Underdog cashed" slam in
 *   1.5-4.0s (45-119):  Team logo scales up with glow + team name
 *   4.0-6.0s (120-179): Odds badge + "UNDERDOG CASHED" + profit counter
 *   6.0-8.0s (180-239): Score + truline.app CTA (holds for screenshot)
 *
 * Audio: if public/music.mp3 exists, plays from frame 0.
 */
export const ModelHit: React.FC<ModelHitProps> = ({
  winnerTeam,
  winnerName,
  oddsAmerican,
  profitUnits,
  scoreText,
  sport,
  teamColor,
  musicFile,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Faster glow pulse (full cycle every 20 frames = 0.67s)
  const glowPulse = interpolate(
    frame % 20,
    [0, 10, 20],
    [0.15, 0.3, 0.15],
  );

  const logoUrl = espnLogoUrl(winnerTeam, sport);
  const oddsStr = `+${oddsAmerican}`;

  return (
    <AbsoluteFill style={{ backgroundColor: COLORS.bg }}>
      {/* Background music — drop an mp3 in public/ and pass musicFile="filename.mp3" */}
      {musicFile && (
        <Audio src={staticFile(musicFile)} volume={0.5} />
      )}

      {/* Team color radial glow — pulsing */}
      <div
        style={{
          position: 'absolute',
          top: '30%',
          left: '50%',
          width: 900,
          height: 900,
          transform: 'translate(-50%, -50%)',
          borderRadius: '50%',
          background: `radial-gradient(circle, ${teamColor} 0%, transparent 70%)`,
          opacity: glowPulse,
          filter: 'blur(80px)',
        }}
      />

      {/* Phase 1: 0-1.5s — "MODEL HIT" slam (persists entire video) */}
      <Sequence from={0} durationInFrames={seconds(8)}>
        <Pop size={150} color={COLORS.green} weight={900} y={250} delay={2}>
          MODEL HIT
        </Pop>
        <Pop size={56} color={COLORS.muted} weight={500} y={420} delay={8}>
          Underdog cashed
        </Pop>
      </Sequence>

      {/* Phase 2: 1.5-4s — Team logo + name */}
      <Sequence from={seconds(1.5)} durationInFrames={seconds(6.5)}>
        {(() => {
          const localFrame = frame - seconds(1.5);
          if (localFrame < 0) return null;

          const logoProgress = spring({
            frame: localFrame,
            fps,
            config: { damping: 16, stiffness: 180, mass: 0.6 },
          });

          const logoScale = interpolate(logoProgress, [0, 1], [0.2, 1]);
          const logoOpacity = interpolate(logoProgress, [0, 1], [0, 1]);

          return (
            <>
              <div
                style={{
                  position: 'absolute',
                  top: 550,
                  left: '50%',
                  transform: `translateX(-50%) scale(${logoScale})`,
                  opacity: logoOpacity,
                  filter: `drop-shadow(0 0 80px ${teamColor})`,
                }}
              >
                <Img src={logoUrl} width={380} height={380} />
              </div>
              <Pop size={96} color={COLORS.text} weight={800} y={980} delay={seconds(1.5) + 8}>
                {winnerName.toUpperCase()}
              </Pop>
            </>
          );
        })()}
      </Sequence>

      {/* Phase 3: 4-6s — Odds + profit */}
      <Sequence from={seconds(4)} durationInFrames={seconds(4)}>
        <Pop size={160} color={COLORS.accent} font={FONTS.mono} weight={900} y={1120} delay={seconds(4)}>
          {oddsStr}
        </Pop>
        <Pop size={48} color={COLORS.muted} weight={600} y={1300} delay={seconds(4) + 5}>
          UNDERDOG CASHED
        </Pop>
        <div style={{ position: 'absolute', top: 1380, left: 0, right: 0 }}>
          <ProfitCounter target={profitUnits} delay={seconds(4) + 10} />
        </div>
      </Sequence>

      {/* Phase 4: 6-8s — Score + CTA (holds for screenshots) */}
      <Sequence from={seconds(6)} durationInFrames={seconds(2)}>
        <Pop size={42} color={COLORS.muted} weight={500} y={1520} delay={seconds(6)}>
          {scoreText}
        </Pop>
        <Pop size={56} color={COLORS.accent} weight={700} y={1640} delay={seconds(6) + 4}>
          truline.app
        </Pop>
        <Pop size={40} color={COLORS.muted} weight={500} y={1730} delay={seconds(6) + 8}>
          Follow for daily picks
        </Pop>
      </Sequence>
    </AbsoluteFill>
  );
};
