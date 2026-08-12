/**
 * The backend↔composition overlay contract.
 *
 * This table is a transcription of what `build_beats()` in
 * backend/src/services/mlb/pick_script.py actually emits. It is the ONLY
 * shared description of that shape — the two halves live in different
 * languages, in different deploy units, and nothing else forces them to
 * agree. Without it the drift is silent: the composition happily renders any
 * overlay object it is handed, so a beat whose keys the backend never
 * produces just renders as a blank screen and no test notices.
 *
 * Read it as: "WHEN a beat with key K appears, its overlay keys are EXACTLY
 * this set." Not every beat appears in every video —
 *   - `case_against` is emitted only when at least one clause survives,
 *   - `turn` only when the first-inning split clears MIN_TURN_BEAT_STARTS,
 * — so the contract is about shape, never about presence.
 *
 * If you change `build_beats()`, change this table in the same change.
 */
export const BEAT_OVERLAY_CONTRACT = {
  hook: ['line'],
  pick: ['team', 'market', 'price'],
  case_against: ['chips'],
  turn: ['stat', 'label'],
  numbers: ['model', 'model_label', 'market', 'market_label'],
  close: ['cta', 'disclaimer'],
} as const satisfies Record<string, readonly string[]>;

export type ContractBeatKey = keyof typeof BEAT_OVERLAY_CONTRACT;

/** Beats the backend emits on every single pick. The rest are conditional. */
export const ALWAYS_EMITTED_BEATS: readonly ContractBeatKey[] = [
  'hook', 'pick', 'numbers', 'close',
];

export const isContractBeatKey = (key: string): key is ContractBeatKey =>
  Object.prototype.hasOwnProperty.call(BEAT_OVERLAY_CONTRACT, key);

/** A plausible overlay for each contract beat, used to exercise the
 *  composition against every declared shape. Values mirror the backend's
 *  formatting (upper-cased team, "%"-suffixed probabilities, and so on). */
export const CONTRACT_BEAT_SAMPLES: Record<ContractBeatKey, Record<string, string>> = {
  hook: { line: "Backing a team that's lost 2 straight." },
  pick: { team: 'CHICAGO WHITE SOX', market: 'MONEYLINE', price: '+155' },
  case_against: { chips: '5-5 L10 · Castillo 5.06 ERA' },
  turn: { stat: '10 of 17', label: 'SCORELESS 1ST' },
  numbers: {
    model: '48%', model_label: 'MODEL PROJECTION',
    market: '39%', market_label: 'BREAKEVEN',
  },
  close: { cta: 'truline.app', disclaimer: 'Not betting advice. 21+.' },
};
