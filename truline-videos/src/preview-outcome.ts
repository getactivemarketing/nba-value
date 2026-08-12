import type { BlotatoUploadResult } from './blotato';
import type { PublishDecision, WithholdReason } from './publish-guard';

/**
 * What happened to one pick preview this run, and whether it is safe to
 * dedupe (write into previews-posted.json so a later run skips it).
 *
 * - 'posted'   — at least one platform actually confirmed the post. Only this
 *                case is safe to dedupe; a pick must not be marked posted
 *                forever off the back of a merely-resolved promise.
 * - 'skipped'  — a deliberate, expected non-publish driven by the pick itself:
 *                the lead-time/publish gate refused (say-narrated render, or
 *                too close to first pitch). Not a failure — retried
 *                automatically next run because it was never deduped.
 * - 'withheld' — the RUN was not allowed to publish at all: fixture data, or
 *                an explicit DRY_RUN. The render happened and is kept on
 *                disk; nothing was uploaded. Neither a success nor a failure.
 * - 'capped'   — the per-run post cap was already used up, so this preview
 *                was not even rendered. Must never dedupe: it has to stay
 *                eligible for the next run.
 * - 'failed'   — something went wrong: the caption tripped the banned-word
 *                guard, or an upload attempt resolved without any platform
 *                confirming. Logged loudly, never deduped, so a later run
 *                retries it.
 */
export type PreviewOutcome = 'posted' | 'skipped' | 'withheld' | 'capped' | 'failed';

export interface PreviewResult {
  outcome: PreviewOutcome;
  /** Whether the caller should write this pick's game_id into the dedupe file. */
  dedupe: boolean;
}

/** Maps a guard refusal onto the run outcome it is reported as. Keeping this
 *  table explicit is what stops a withheld fixture run from reading like a
 *  routine lead-time skip in the log line. */
const OUTCOME_FOR_REASON: Record<WithholdReason, PreviewOutcome> = {
  'post-cap': 'capped',
  fixture: 'withheld',
  'dry-run': 'withheld',
  'not-publishable': 'skipped',
  'lead-time': 'skipped',
};

/**
 * Decides the outcome of one preview from the publish gate's decision and (if
 * the gate allowed it) the upload attempt's result. `upload` is `null` when
 * the gate refused, or when upload was never attempted at all (e.g. the
 * caption banned-word guard refused before calling uploadToBlotato).
 *
 * The core rule this exists to enforce: `uploadToBlotato` resolving is NOT
 * proof anything posted (see blotato.ts) — only `upload.posted.length > 0`
 * is. Everything else, including any `mayPublish` refusal, must never dedupe.
 */
export function decidePreviewResult(
  decision: PublishDecision,
  upload: BlotatoUploadResult | null,
): PreviewResult {
  if (!decision.allowed) {
    return { outcome: OUTCOME_FOR_REASON[decision.reason], dedupe: false };
  }
  if (upload && upload.posted.length > 0) return { outcome: 'posted', dedupe: true };
  return { outcome: 'failed', dedupe: false };
}

/**
 * The slate line, logged before any work starts.
 *
 * The endpoint returns a `skipped` count for eligible snapshots it refused —
 * the narration contract rejected them (its banned-word check is a naive
 * substring match, so an ordinary surname like "Wedge" trips it), or their
 * data was unmeasurable. Without that count in the log, a night where the
 * backend dropped the entire slate is byte-identical to a night with no
 * games, which is exactly the confusion the field was added to remove.
 */
export function describeSlate(eligible: number, skipped: number): string {
  const base = `${eligible} eligible pick(s)`;
  if (skipped <= 0) return base;
  return (
    `${base}, ${skipped} dropped by the backend ` +
    '(narration contract refused them, or their data was unmeasurable)'
  );
}

export interface RunTally {
  posted: number;
  skipped: number;
  withheld: number;
  capped: number;
  failed: number;
}

export const emptyTally = (): RunTally =>
  ({ posted: 0, skipped: 0, withheld: 0, capped: 0, failed: 0 });

/** Returns a NEW tally with `outcome`'s counter incremented — the other
 *  counters are untouched, so no two outcomes can ever bleed into each other
 *  in the reported summary. */
export function addOutcome(tally: RunTally, outcome: PreviewOutcome): RunTally {
  return { ...tally, [outcome]: tally[outcome] + 1 };
}

/** The one line this pipeline gets monitored by (grepped for "Done."). Must
 *  never collapse skipped/withheld/capped/failed into "succeeded" — see
 *  IMPORTANT 2. */
export function formatTally(tally: RunTally): string {
  return (
    `Done. ${tally.posted} posted, ${tally.skipped} skipped, ` +
    `${tally.withheld} withheld, ${tally.capped} capped, ${tally.failed} failed.`
  );
}
