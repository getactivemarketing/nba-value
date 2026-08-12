/** Minutes of clearance required before first pitch. Mirrors the backend's
 *  PREVIEW_MIN_LEAD_MINUTES — the backend gates selection, this gates upload,
 *  and rendering between them can take minutes. */
export const MIN_LEAD_MINUTES = 45;

/**
 * How many previews a single run may post.
 *
 * Nothing else bounds this: a backfill, a wrong clock, or a wide `days=`
 * parameter would otherwise dump the whole slate onto TikTok in one run.
 * Posting is irreversible and a burst is an account-health risk, so the
 * default is deliberately small — raise it per-run via MAX_POSTS_PER_RUN
 * rather than editing this constant.
 */
export const DEFAULT_MAX_POSTS_PER_RUN = 3;

/**
 * Why the guard refused. Callers map these onto run outcomes (see
 * src/preview-outcome.ts) — they are NOT interchangeable:
 *
 * - 'post-cap'        the per-run cap is already used up. The ONLY reason
 *                     that is worth knowing before rendering: a capped
 *                     preview's render would be thrown away, and it must
 *                     stay un-deduped so the next run picks it up. Reported
 *                     only when the run could otherwise have published —
 *                     see the ordering note on mayPublish.
 * - 'fixture'         previews came from PICK_PREVIEWS_FIXTURE. Hand-built
 *                     or stale JSON never passed the backend's
 *                     NarrationContractError guard, so it must never reach
 *                     a public account — whatever else is true.
 * - 'dry-run'         DRY_RUN was asked for explicitly, against live data.
 * - 'not-publishable' the TTS adapter is not fit to publish (`say`).
 * - 'lead-time'       too close to (or past) first pitch.
 */
export type WithholdReason =
  | 'post-cap'
  | 'fixture'
  | 'dry-run'
  | 'not-publishable'
  | 'lead-time';

export type PublishDecision =
  | { allowed: true }
  | { allowed: false; reason: WithholdReason };

export interface PublishContext {
  /** Whether the TTS adapter that narrated this render may be published. */
  adapterPublishable: boolean;
  minutesToFirstPitch: number;
  /** True when previews were loaded from PICK_PREVIEWS_FIXTURE. */
  fixture?: boolean;
  /** True when DRY_RUN was requested. */
  dryRun?: boolean;
  /** Posts already confirmed earlier in this run. */
  postsThisRun?: number;
  maxPostsPerRun?: number;
  minLeadMinutes?: number;
}

/**
 * The single chokepoint that decides whether anything may be uploaded.
 *
 * Every reason to withhold lives here — fixture data, an explicit dry run,
 * the per-run post cap, an unpublishable narrator, and the lead-time gate —
 * so there is exactly one place to read (and one place to break) when asking
 * "could this run have posted?". Do not add publish conditions to the
 * orchestrator.
 *
 * Order is deliberate. The whole-run refusals come first: a run that cannot
 * publish anything makes the post cap meaningless, and the cap is the one
 * refusal that suppresses the RENDER as well as the upload — so checking it
 * ahead of them made `DRY_RUN=1 MAX_POSTS_PER_RUN=0` produce nothing at all
 * when the entire point of that combination is to render everything and
 * upload none of it. Fixture beats dry-run so a fixture run reports the more
 * specific reason. The cap comes next, before the per-preview safety gates,
 * because it is the only refusal the caller acts on BEFORE spending render
 * time. Adapter and lead time come last: they are facts about one preview.
 */
export function mayPublish(ctx: PublishContext): PublishDecision {
  const {
    adapterPublishable,
    minutesToFirstPitch,
    fixture = false,
    dryRun = false,
    postsThisRun = 0,
    maxPostsPerRun = DEFAULT_MAX_POSTS_PER_RUN,
    minLeadMinutes = MIN_LEAD_MINUTES,
  } = ctx;

  if (fixture) return { allowed: false, reason: 'fixture' };
  if (dryRun) return { allowed: false, reason: 'dry-run' };
  if (postsThisRun >= maxPostsPerRun) return { allowed: false, reason: 'post-cap' };
  if (!adapterPublishable) return { allowed: false, reason: 'not-publishable' };
  if (!(minutesToFirstPitch > minLeadMinutes)) return { allowed: false, reason: 'lead-time' };
  return { allowed: true };
}

/** True only for the refusal worth knowing before rendering — see
 *  WithholdReason. Fixture/dry-run renders are still produced and kept. */
export function refusedBeforeRender(decision: PublishDecision): boolean {
  return !decision.allowed && decision.reason === 'post-cap';
}

/** Human-readable one-liner for the run log. */
export function describeDecision(decision: PublishDecision): string {
  if (decision.allowed) return 'clear to publish';
  switch (decision.reason) {
    case 'post-cap':
      return 'per-run post cap reached';
    case 'fixture':
      return 'PICK_PREVIEWS_FIXTURE is set — fixture data must never be published';
    case 'dry-run':
      return 'DRY_RUN is set';
    case 'not-publishable':
      return 'TTS adapter is not publishable';
    case 'lead-time':
      return 'inside the lead-time gate before first pitch';
  }
}

const TRUTHY = new Set(['1', 'true', 'yes', 'on']);

/** DRY_RUN as a boolean. Anything unset, empty, "0" or "false" is off, so a
 *  stray empty env var cannot silently disable publishing forever. */
export function dryRunFromEnv(env: NodeJS.ProcessEnv = process.env): boolean {
  return TRUTHY.has((env.DRY_RUN || '').trim().toLowerCase());
}

/** MAX_POSTS_PER_RUN as a number. 0 is legal (post nothing); anything
 *  unparseable or negative falls back to the default rather than being
 *  read as "unlimited". */
export function maxPostsPerRunFromEnv(env: NodeJS.ProcessEnv = process.env): number {
  const raw = (env.MAX_POSTS_PER_RUN || '').trim();
  if (!raw) return DEFAULT_MAX_POSTS_PER_RUN;
  const parsed = Number(raw);
  if (!Number.isInteger(parsed) || parsed < 0) return DEFAULT_MAX_POSTS_PER_RUN;
  return parsed;
}
