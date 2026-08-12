import { describe, expect, it } from 'vitest';
import {
  DEFAULT_MAX_POSTS_PER_RUN,
  dryRunFromEnv,
  mayPublish,
  maxPostsPerRunFromEnv,
  refusedBeforeRender,
  type PublishContext,
} from './publish-guard';

/** A run that is otherwise entirely clear to publish. Each test turns exactly
 *  one thing off, so a passing assertion can only be about that thing. */
const clear = (over: Partial<PublishContext> = {}): PublishContext => ({
  adapterPublishable: true,
  minutesToFirstPitch: 120,
  ...over,
});

describe('mayPublish — lead time and narrator', () => {
  it('allows a publishable adapter with clearance', () => {
    expect(mayPublish(clear())).toEqual({ allowed: true });
  });

  it('refuses a say-narrated render however much time is left', () => {
    expect(mayPublish(clear({ adapterPublishable: false, minutesToFirstPitch: 600 })))
      .toEqual({ allowed: false, reason: 'not-publishable' });
  });

  it('refuses inside the lead-time gate', () => {
    expect(mayPublish(clear({ minutesToFirstPitch: 30 })))
      .toEqual({ allowed: false, reason: 'lead-time' });
  });

  it('refuses after first pitch', () => {
    expect(mayPublish(clear({ minutesToFirstPitch: -5 })))
      .toEqual({ allowed: false, reason: 'lead-time' });
  });

  it('re-checks at upload time, not render time', () => {
    // A render that took 20 minutes leaves 40 — under the gate, so refuse.
    expect(mayPublish(clear({ minutesToFirstPitch: 40 })))
      .toEqual({ allowed: false, reason: 'lead-time' });
  });
});

describe('mayPublish — fixture data can never be published', () => {
  it('refuses fixture data even when everything else is clear', () => {
    expect(mayPublish(clear({ fixture: true })))
      .toEqual({ allowed: false, reason: 'fixture' });
  });

  it('refuses fixture data with a publishable narrator and hours of clearance', () => {
    // The exact combination that would otherwise post to TikTok. Fixture JSON
    // is hand-written and never went through NarrationContractError.
    expect(mayPublish(clear({ fixture: true, minutesToFirstPitch: 600 })).allowed).toBe(false);
  });

  it('fixture refusal is not something a big post cap can override', () => {
    expect(mayPublish(clear({ fixture: true, maxPostsPerRun: 999, postsThisRun: 0 })).allowed)
      .toBe(false);
  });
});

describe('mayPublish — explicit dry run against live data', () => {
  it('refuses when DRY_RUN was asked for, with live (non-fixture) data', () => {
    expect(mayPublish(clear({ dryRun: true })))
      .toEqual({ allowed: false, reason: 'dry-run' });
  });

  it('reports the more specific reason when data is BOTH fixture and dry-run', () => {
    expect(mayPublish(clear({ fixture: true, dryRun: true })))
      .toEqual({ allowed: false, reason: 'fixture' });
  });

  it('a run without DRY_RUN is unaffected', () => {
    expect(mayPublish(clear({ dryRun: false })).allowed).toBe(true);
  });
});

describe('mayPublish — per-run post cap', () => {
  it('allows posts below the cap', () => {
    expect(mayPublish(clear({ postsThisRun: 2, maxPostsPerRun: 3 })).allowed).toBe(true);
  });

  it('refuses once the cap is reached', () => {
    expect(mayPublish(clear({ postsThisRun: 3, maxPostsPerRun: 3 })))
      .toEqual({ allowed: false, reason: 'post-cap' });
  });

  it('refuses beyond the cap — a mis-count can never re-open publishing', () => {
    expect(mayPublish(clear({ postsThisRun: 9, maxPostsPerRun: 3 })))
      .toEqual({ allowed: false, reason: 'post-cap' });
  });

  it('a cap of 0 refuses everything', () => {
    expect(mayPublish(clear({ postsThisRun: 0, maxPostsPerRun: 0 })))
      .toEqual({ allowed: false, reason: 'post-cap' });
  });

  it('defaults the cap when none is supplied — an unbounded backfill is impossible', () => {
    expect(mayPublish(clear({ postsThisRun: DEFAULT_MAX_POSTS_PER_RUN })))
      .toEqual({ allowed: false, reason: 'post-cap' });
    expect(mayPublish(clear({ postsThisRun: DEFAULT_MAX_POSTS_PER_RUN - 1 })).allowed).toBe(true);
  });

  it('the cap is reported ahead of the per-preview gates, so the caller can skip the render', () => {
    const decision = mayPublish(clear({
      postsThisRun: 5, maxPostsPerRun: 3,
      adapterPublishable: false, minutesToFirstPitch: -100,
    }));
    expect(decision).toEqual({ allowed: false, reason: 'post-cap' });
  });

  it('a whole-run refusal outranks the cap, so a dry run still renders everything', () => {
    // MAX_POSTS_PER_RUN=0 with DRY_RUN=1 previously reported 'post-cap' for
    // every preview, and the caller skips the render on 'post-cap' — so the
    // one combination that means "render it all, upload none of it" produced
    // nothing at all.
    const dry = mayPublish(clear({ dryRun: true, postsThisRun: 0, maxPostsPerRun: 0 }));
    expect(dry).toEqual({ allowed: false, reason: 'dry-run' });
    expect(refusedBeforeRender(dry)).toBe(false);

    const fixture = mayPublish(clear({ fixture: true, postsThisRun: 9, maxPostsPerRun: 0 }));
    expect(fixture).toEqual({ allowed: false, reason: 'fixture' });
    expect(refusedBeforeRender(fixture)).toBe(false);
  });
});

describe('refusedBeforeRender — only the cap is worth knowing pre-render', () => {
  it('is true for the post cap', () => {
    expect(refusedBeforeRender({ allowed: false, reason: 'post-cap' })).toBe(true);
  });

  it('is false for fixture and dry-run — those renders are still wanted on disk', () => {
    expect(refusedBeforeRender({ allowed: false, reason: 'fixture' })).toBe(false);
    expect(refusedBeforeRender({ allowed: false, reason: 'dry-run' })).toBe(false);
  });

  it('is false for the lead-time and narrator gates', () => {
    expect(refusedBeforeRender({ allowed: false, reason: 'lead-time' })).toBe(false);
    expect(refusedBeforeRender({ allowed: false, reason: 'not-publishable' })).toBe(false);
  });

  it('is false when publishing is allowed', () => {
    expect(refusedBeforeRender({ allowed: true })).toBe(false);
  });
});

describe('dryRunFromEnv', () => {
  it('is off when unset', () => {
    expect(dryRunFromEnv({})).toBe(false);
  });

  it('is on for 1/true/yes/on, in any case', () => {
    for (const v of ['1', 'true', 'TRUE', 'yes', 'On', ' 1 ']) {
      expect(dryRunFromEnv({ DRY_RUN: v }), `DRY_RUN=${v}`).toBe(true);
    }
  });

  it('is off for an empty or explicitly-negative value', () => {
    for (const v of ['', '0', 'false', 'no']) {
      expect(dryRunFromEnv({ DRY_RUN: v }), `DRY_RUN=${v}`).toBe(false);
    }
  });
});

describe('maxPostsPerRunFromEnv', () => {
  it('defaults when unset', () => {
    expect(maxPostsPerRunFromEnv({})).toBe(DEFAULT_MAX_POSTS_PER_RUN);
  });

  it('reads an explicit override', () => {
    expect(maxPostsPerRunFromEnv({ MAX_POSTS_PER_RUN: '7' })).toBe(7);
  });

  it('treats 0 as a real value — post nothing', () => {
    expect(maxPostsPerRunFromEnv({ MAX_POSTS_PER_RUN: '0' })).toBe(0);
  });

  it('falls back to the default rather than reading garbage as unlimited', () => {
    for (const v of ['abc', '-1', '2.5', 'Infinity', '']) {
      expect(maxPostsPerRunFromEnv({ MAX_POSTS_PER_RUN: v }), `MAX_POSTS_PER_RUN=${v}`)
        .toBe(DEFAULT_MAX_POSTS_PER_RUN);
    }
  });
});
