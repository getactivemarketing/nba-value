import { describe, expect, it } from 'vitest';
import { addOutcome, decidePreviewResult, emptyTally, formatTally } from './preview-outcome';

describe('decidePreviewResult', () => {
  it('the publish gate refusing is "skipped", never deduped', () => {
    expect(decidePreviewResult(false, null)).toEqual({ outcome: 'skipped', dedupe: false });
  });

  it('the gate refusing wins even if an upload result is somehow present', () => {
    expect(decidePreviewResult(false, { uploaded: true, posted: ['tiktok'] }))
      .toEqual({ outcome: 'skipped', dedupe: false });
  });

  it('a null upload after the gate allowed it (e.g. caption guard refusal) is "failed", never deduped', () => {
    expect(decidePreviewResult(true, null)).toEqual({ outcome: 'failed', dedupe: false });
  });

  it('an upload that resolved but confirmed nothing is "failed", never deduped', () => {
    expect(decidePreviewResult(true, { uploaded: true, posted: [] }))
      .toEqual({ outcome: 'failed', dedupe: false });
  });

  it('uploaded=false with posted=[] (dry run / total failure) is "failed", never deduped', () => {
    expect(decidePreviewResult(true, { uploaded: false, posted: [] }))
      .toEqual({ outcome: 'failed', dedupe: false });
  });

  it('at least one platform confirming is "posted" and IS deduped', () => {
    expect(decidePreviewResult(true, { uploaded: true, posted: ['instagram'] }))
      .toEqual({ outcome: 'posted', dedupe: true });
  });
});

describe('run tally — posted, skipped and failed never bleed into each other', () => {
  it('starts at all zero', () => {
    expect(emptyTally()).toEqual({ posted: 0, skipped: 0, failed: 0 });
  });

  it('increments only the counter for the given outcome', () => {
    let tally = emptyTally();
    tally = addOutcome(tally, 'skipped');
    expect(tally).toEqual({ posted: 0, skipped: 1, failed: 0 });
    tally = addOutcome(tally, 'skipped');
    tally = addOutcome(tally, 'posted');
    tally = addOutcome(tally, 'failed');
    expect(tally).toEqual({ posted: 1, skipped: 2, failed: 1 });
  });

  it('a run of only skips and failures reports zero posted, distinctly — not folded into "succeeded"', () => {
    let tally = emptyTally();
    tally = addOutcome(tally, 'skipped');
    tally = addOutcome(tally, 'skipped');
    expect(formatTally(tally)).toBe('Done. 0 posted, 2 skipped, 0 failed.');
  });

  it('formats all three counts distinctly', () => {
    const tally = { posted: 2, skipped: 1, failed: 3 };
    expect(formatTally(tally)).toBe('Done. 2 posted, 1 skipped, 3 failed.');
  });
});
