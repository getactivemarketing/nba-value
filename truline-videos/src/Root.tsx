import React from 'react';
import { Composition } from 'remotion';
import { ModelHit, type ModelHitProps } from './compositions/ModelHit';
import { PickPreview, calculatePickPreviewMetadata } from './compositions/PickPreview';
import { FPS, WIDTH, HEIGHT, seconds } from './constants';
import { PICK_PREVIEW_DEFAULT_PROPS } from './pick-preview-defaults';

export const Root: React.FC = () => {
  const modelHitDefaultProps: ModelHitProps = {
    winnerTeam: 'GSW',
    winnerName: 'Warriors',
    oddsAmerican: 180,
    profitUnits: 1.80,
    scoreText: 'GSW 118, LAC 105',
    sport: 'nba',
    teamColor: '#1D428A',
  };


  return (
    <>
      <Composition
        id="model-hit"
        component={ModelHit as unknown as React.ComponentType<Record<string, unknown>>}
        durationInFrames={seconds(8)}
        fps={FPS}
        width={WIDTH}
        height={HEIGHT}
        defaultProps={modelHitDefaultProps as unknown as Record<string, unknown>}
      />
      <Composition
        id="pick-preview"
        component={PickPreview as unknown as React.ComponentType<Record<string, unknown>>}
        durationInFrames={900}
        fps={FPS}
        width={WIDTH}
        height={HEIGHT}
        defaultProps={PICK_PREVIEW_DEFAULT_PROPS as unknown as Record<string, unknown>}
        calculateMetadata={calculatePickPreviewMetadata as never}
      />
    </>
  );
};
