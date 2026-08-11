import React from 'react';
import { Composition } from 'remotion';
import { ModelHit, type ModelHitProps } from './compositions/ModelHit';
import { PickPreview, calculatePickPreviewMetadata, type PickPreviewProps } from './compositions/PickPreview';
import { FPS, WIDTH, HEIGHT, seconds, espnLogoUrl } from './constants';

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

  const pickPreviewDefaultProps: PickPreviewProps = {
    beats: [
      { key: 'hook', overlay: {}, audioSrc: '', durationInFrames: 60 },
      { key: 'pick', overlay: { team: 'CWS', price: '-110', priceLabel: 'Moneyline' }, audioSrc: '', durationInFrames: 90 },
      { key: 'turn', overlay: { stat: '4.2% Edge', statLabel: 'Model Edge' }, audioSrc: '', durationInFrames: 75 },
      { key: 'numbers', overlay: { number: '71.3%', numberLabel: 'Win Probability' }, audioSrc: '', durationInFrames: 75 },
      { key: 'close', overlay: { cta: 'Follow for Updates', disclaimer: 'Not investment advice' }, audioSrc: '', durationInFrames: 60 },
    ],
    teamColor: '#27251F',
    logoUrl: espnLogoUrl('CWS', 'mlb'),
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
        defaultProps={pickPreviewDefaultProps as unknown as Record<string, unknown>}
        calculateMetadata={calculatePickPreviewMetadata as never}
      />
    </>
  );
};
