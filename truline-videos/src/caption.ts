/**
 * The caption that actually posts publicly.
 *
 * Extracted from the orchestrator so its exact shape is testable without
 * going near the publishing path — this string is the one part of the
 * pipeline that ships free text to a public account.
 */

export interface CaptionInput {
  teamName: string;
  oddsAmerican: number;
  /** The turn beat's narration, if that beat exists at all. The backend drops
   *  the turn beat when the first-inning split is too thin (see
   *  MIN_TURN_BEAT_STARTS), so absence is normal, not an error. */
  turnNarration?: string | null;
}

const DISCLAIMER = 'Not betting advice. 21+.';
const HASHTAGS = '#MLB #SportsAnalytics';

/**
 * Blocks joined by a BLANK LINE — the paragraph breaks are load-bearing, a
 * single run-on block reads like spam. The turn narration is the only
 * optional block: when the backend omitted the turn beat, its block is
 * dropped whole so there is no double blank line and no dangling gap.
 */
export function buildPreviewCaption({
  teamName,
  oddsAmerican,
  turnNarration,
}: CaptionInput): string {
  const price = `${oddsAmerican > 0 ? '+' : ''}${oddsAmerican}`;
  const turn = (turnNarration || '').trim();

  const blocks = [
    `${teamName} ML ${price}.`,
    ...(turn ? [turn] : []),
    `${DISCLAIMER}\n${HASHTAGS}`,
  ];

  return blocks.join('\n\n');
}
