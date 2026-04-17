import React from 'react';
import {
  AbsoluteFill,
  Sequence,
  Img,
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig,
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
}

const AnimatedText: React.FC<{
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
    config: { damping: 14, stiffness: 100, mass: 0.8 },
  });

  const scale = interpolate(progress, [0, 1], [0.6, 1]);
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

const ProfitCounter: React.FC<{
  target: number;
  delay?: number;
}> = ({ target, delay = 0 }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const local = frame - delay;
  if (local < 0) return null;

  const progress = spring({
    frame: local,
    fps,
    config: { damping: 20, stiffness: 80, mass: 1 },
  });

  const value = interpolate(progress, [0, 1], [0, target]);
  const opacity = interpolate(progress, [0, 1], [0, 1]);

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

export const ModelHit: React.FC<ModelHitProps> = ({
  winnerTeam,
  winnerName,
  oddsAmerican,
  profitUnits,
  scoreText,
  sport,
  teamColor,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const glowPulse = interpolate(
    frame % 60,
    [0, 30, 60],
    [0.15, 0.25, 0.15],
  );

  const logoUrl = espnLogoUrl(winnerTeam, sport);
  const oddsStr = `+${oddsAmerican}`;

  return (
    <AbsoluteFill style={{ backgroundColor: COLORS.bg }}>
      {/* Team color radial glow */}
      <div
        style={{
          position: 'absolute',
          top: '30%',
          left: '50%',
          width: 800,
          height: 800,
          transform: 'translate(-50%, -50%)',
          borderRadius: '50%',
          background: `radial-gradient(circle, ${teamColor} 0%, transparent 70%)`,
          opacity: glowPulse,
          filter: 'blur(80px)',
        }}
      />

      {/* Phase 1: 0-3s — MODEL HIT title (persists) */}
      <Sequence from={0} durationInFrames={seconds(15)}>
        <AnimatedText size={140} color={COLORS.green} weight={900} y={280} delay={5}>
          MODEL HIT
        </AnimatedText>
        <AnimatedText size={56} color={COLORS.muted} weight={500} y={440} delay={15}>
          Underdog cashed
        </AnimatedText>
      </Sequence>

      {/* Phase 2: 3-7s — Team logo + name */}
      <Sequence from={seconds(3)} durationInFrames={seconds(12)}>
        {(() => {
          const localFrame = frame - seconds(3);
          if (localFrame < 0) return null;

          const logoProgress = spring({
            frame: localFrame,
            fps,
            config: { damping: 12, stiffness: 80, mass: 1 },
          });

          const logoScale = interpolate(logoProgress, [0, 1], [0.3, 1]);
          const logoOpacity = interpolate(logoProgress, [0, 1], [0, 1]);

          return (
            <>
              <div
                style={{
                  position: 'absolute',
                  top: 580,
                  left: '50%',
                  transform: `translateX(-50%) scale(${logoScale})`,
                  opacity: logoOpacity,
                  filter: `drop-shadow(0 0 60px ${teamColor})`,
                }}
              >
                <Img src={logoUrl} width={380} height={380} />
              </div>
              <AnimatedText size={96} color={COLORS.text} weight={800} y={1000} delay={seconds(3) + 15}>
                {winnerName.toUpperCase()}
              </AnimatedText>
            </>
          );
        })()}
      </Sequence>

      {/* Phase 3: 7-11s — Odds + profit */}
      <Sequence from={seconds(7)} durationInFrames={seconds(8)}>
        <AnimatedText size={160} color={COLORS.accent} font={FONTS.mono} weight={900} y={1160} delay={seconds(7)}>
          {oddsStr}
        </AnimatedText>
        <AnimatedText size={48} color={COLORS.muted} weight={600} y={1340} delay={seconds(7) + 10}>
          UNDERDOG CASHED
        </AnimatedText>
        <div style={{ position: 'absolute', top: 1420, left: 0, right: 0 }}>
          <ProfitCounter target={profitUnits} delay={seconds(7) + 20} />
        </div>
      </Sequence>

      {/* Phase 4: 11-15s — CTA */}
      <Sequence from={seconds(11)} durationInFrames={seconds(4)}>
        <AnimatedText size={42} color={COLORS.muted} weight={500} y={1560} delay={seconds(11)}>
          {scoreText}
        </AnimatedText>
        <AnimatedText size={56} color={COLORS.accent} weight={700} y={1680} delay={seconds(11) + 10}>
          truline.app
        </AnimatedText>
        <AnimatedText size={40} color={COLORS.muted} weight={500} y={1760} delay={seconds(11) + 15}>
          Follow for daily picks
        </AnimatedText>
      </Sequence>
    </AbsoluteFill>
  );
};
