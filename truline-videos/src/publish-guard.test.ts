import { describe, expect, it } from 'vitest';
import { mayPublish } from './publish-guard';

describe('mayPublish', () => {
  it('allows a publishable adapter with clearance', () => {
    expect(mayPublish(true, 120)).toBe(true);
  });

  it('refuses a say-narrated render however much time is left', () => {
    expect(mayPublish(false, 600)).toBe(false);
  });

  it('refuses inside the lead-time gate', () => {
    expect(mayPublish(true, 30)).toBe(false);
  });

  it('refuses after first pitch', () => {
    expect(mayPublish(true, -5)).toBe(false);
  });

  it('re-checks at upload time, not render time', () => {
    // A render that took 20 minutes leaves 40 — under the gate, so refuse.
    expect(mayPublish(true, 40)).toBe(false);
  });
});
