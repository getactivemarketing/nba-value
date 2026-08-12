import type { BlotatoUploadResult } from './blotato';

/**
 * What happened to one pick preview this run, and whether it is safe to
 * dedupe (write into previews-posted.json so a later run skips it).
 *
 * - 'posted'  — at least one platform actually confirmed the post. Only this
 *               case is safe to dedupe; a pick must not be marked posted
 *               forever off the back of a merely-resolved promise.
 * - 'skipped' — a deliberate, expected non-publish: the lead-time/publish
 *               gate refused (say-narrated render, or too close to first
 *               pitch). Not a failure — retried automatically next run
 *               because it was never deduped.
 * - 'failed'  — something went wrong: the caption tripped the banned-word
 *               guard, or an upload attempt resolved without any platform
 *               confirming. Logged loudly, never deduped, so a later run
 *               retries it.
 */
export type PreviewOutcome = 'posted' | 'skipped' | 'failed';

export interface PreviewResult {
  outcome: PreviewOutcome;
  /** Whether the caller should write this pick's game_id into the dedupe file. */
  dedupe: boolean;
}

/**
 * Decides the outcome of one preview from the publish gate and (if the gate
 * allowed it) the upload attempt's result. `upload` is `null` when the gate
 * refused, or when upload was never attempted at all (e.g. the caption
 * banned-word guard refused before calling uploadToBlotato).
 *
 * The core rule this exists to enforce: `uploadToBlotato` resolving is NOT
 * proof anything posted (see blotato.ts) — only `upload.posted.length > 0`
 * is. Everything else, including a `mayPublish` refusal, must never dedupe.
 */
export function decidePreviewResult(
  mayPublish: boolean,
  upload: BlotatoUploadResult | null,
): PreviewResult {
  if (!mayPublish) return { outcome: 'skipped', dedupe: false };
  if (upload && upload.posted.length > 0) return { outcome: 'posted', dedupe: true };
  return { outcome: 'failed', dedupe: false };
}

export interface RunTally {
  posted: number;
  skipped: number;
  failed: number;
}

export const emptyTally = (): RunTally => ({ posted: 0, skipped: 0, failed: 0 });

/** Returns a NEW tally with `outcome`'s counter incremented — the other two
 *  counters are untouched, so "posted" and "skipped" can never bleed into
 *  each other in the reported summary. */
export function addOutcome(tally: RunTally, outcome: PreviewOutcome): RunTally {
  return { ...tally, [outcome]: tally[outcome] + 1 };
}

/** The one line this pipeline gets monitored by (grepped for "Done."). Must
 *  never collapse skipped/failed into "succeeded" — see IMPORTANT 2. */
export function formatTally(tally: RunTally): string {
  return `Done. ${tally.posted} posted, ${tally.skipped} skipped, ${tally.failed} failed.`;
}
