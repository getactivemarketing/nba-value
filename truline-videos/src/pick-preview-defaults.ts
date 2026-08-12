/**
 * Default beats for PickPreview composition.
 *
 * These defaults represent realistic pick-preview content from the backend
 * and are used in:
 * - Root.tsx for Remotion Studio preview
 * - test files for composition behavior verification
 *
 * Keep these aligned with backend/src/services/mlb/pick_script.py Beat structure.
 * The overlay KEY SETS are pinned by src/pick-preview-contract.ts and asserted
 * in pick-preview-contract.test.ts — a default whose keys the backend cannot
 * emit is not a harmless placeholder, it makes Studio and every composition
 * test exercise a shape that never occurs in production.
 *
 * Hard constraints enforced by NarrationContractError:
 * - No beat overlay contains "edge" (case-insensitive)
 * - Close beat disclaimer is EXACTLY "Not betting advice. 21+."
 */

import { espnLogoUrl } from './constants';
import type { PickPreviewProps } from './compositions/PickPreview';

export const PICK_PREVIEW_DEFAULT_PROPS: PickPreviewProps = {
  beats: [
    {
      key: 'hook',
      // The backend always emits {line: <the hook sentence>} — never an empty
      // overlay. An empty one renders as a blank screen for the whole beat.
      overlay: { line: "Backing a team that's lost 2 straight." },
      audioSrc: '',
      durationInFrames: 60,
    },
    {
      key: 'pick',
      overlay: { team: 'CWS', market: 'MONEYLINE', price: '+155' },
      audioSrc: '',
      durationInFrames: 90,
    },
    {
      key: 'case_against',
      overlay: { chips: '5-5 L10 · Castillo 5.06 ERA' },
      audioSrc: '',
      durationInFrames: 75,
    },
    {
      key: 'turn',
      overlay: { stat: '10 of 17', label: 'SCORELESS 1ST' },
      audioSrc: '',
      durationInFrames: 75,
    },
    {
      key: 'numbers',
      overlay: {
        model: '48%',
        model_label: 'MODEL PROJECTION',
        market: '39%',
        market_label: 'BREAKEVEN',
      },
      audioSrc: '',
      durationInFrames: 75,
    },
    {
      key: 'close',
      overlay: { cta: 'truline.app', disclaimer: 'Not betting advice. 21+.' },
      audioSrc: '',
      durationInFrames: 60,
    },
  ],
  teamColor: '#27251F',
  logoUrl: espnLogoUrl('CWS', 'mlb'),
};
