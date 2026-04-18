import React from 'react';
import { Composition } from 'remotion';
import { ModelHit, type ModelHitProps } from './compositions/ModelHit';
import { FPS, WIDTH, HEIGHT, seconds } from './constants';

export const Root: React.FC = () => {
  const defaultProps: ModelHitProps = {
    winnerTeam: 'GSW',
    winnerName: 'Warriors',
    oddsAmerican: 180,
    profitUnits: 1.80,
    scoreText: 'GSW 118, LAC 105',
    sport: 'nba',
    teamColor: '#1D428A',
  };

  return (
    <Composition
      id="model-hit"
      component={ModelHit as unknown as React.ComponentType<Record<string, unknown>>}
      durationInFrames={seconds(8)}
      fps={FPS}
      width={WIDTH}
      height={HEIGHT}
      defaultProps={defaultProps as unknown as Record<string, unknown>}
    />
  );
};
