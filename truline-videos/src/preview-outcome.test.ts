import { describe, expect, it } from 'vitest';
import { addOutcome, decidePreviewResult, describeSlate, emptyTally, formatTally } from './preview-outcome';
import type { PublishDecision } from './publish-guard';

const ALLOWED: PublishDecision = { allowed: true };
const refused = (reason: Extract<PublishDecision, { allowed: false }>['reason']): PublishDecision =>
  ({ allowed: false, reason });

describe('decidePreviewResult', () => {
  it('the lead-time gate refusing is "skipped", never deduped', () => {
    expect(decidePreviewResult(refused('lead-time'), null))
      .toEqual({ outcome: 'skipped', dedupe: false });
  });

  it('an unpublishable narrator is "skipped", never deduped', () => {
    expect(decidePreviewResult(refused('not-publishable'), null))
      .toEqual({ outcome: 'skipped', dedupe: false });
  });

  it('the gate refusing wins even if an upload result is somehow present', () => {
    expect(decidePreviewResult(refused('lead-time'), { uploaded: true, posted: ['tiktok'] }))
      .toEqual({ outcome: 'skipped', dedupe: false });
  });

  it('a null upload after the gate allowed it (e.g. caption guard refusal) is "failed", never deduped', () => {
    expect(decidePreviewResult(ALLOWED, null)).toEqual({ outcome: 'failed', dedupe: false });
  });

  it('an upload that resolved but confirmed nothing is "failed", never deduped', () => {
    expect(decidePreviewResult(ALLOWED, { uploaded: true, posted: [] }))
      .toEqual({ outcome: 'failed', dedupe: false });
  });

  it('uploaded=false with posted=[] (dry run / total failure) is "failed", never deduped', () => {
    expect(decidePreviewResult(ALLOWED, { uploaded: false, posted: [] }))
      .toEqual({ outcome: 'failed', dedupe: false });
  });

  it('at least one platform confirming is "posted" and IS deduped', () => {
    expect(decidePreviewResult(ALLOWED, { uploaded: true, posted: ['instagram'] }))
      .toEqual({ outcome: 'posted', dedupe: true });
  });
});

describe('decidePreviewResult — withheld and capped are their own outcomes', () => {
  it('a fixture run is "withheld", not a success and not a failure, never deduped', () => {
    expect(decidePreviewResult(refused('fixture'), null))
      .toEqual({ outcome: 'withheld', dedupe: false });
  });

  it('an explicit dry run is "withheld", never deduped', () => {
    expect(decidePreviewResult(refused('dry-run'), null))
      .toEqual({ outcome: 'withheld', dedupe: false });
  });

  it('a fixture run never dedupes even if an upload result is somehow present', () => {
    // Belt and braces: fixture narration must not be able to mark a real
    // game_id as posted-forever.
    expect(decidePreviewResult(refused('fixture'), { uploaded: true, posted: ['tiktok'] }))
      .toEqual({ outcome: 'withheld', dedupe: false });
  });

  it('hitting the per-run cap is "capped" and MUST stay eligible for the next run', () => {
    expect(decidePreviewResult(refused('post-cap'), null))
      .toEqual({ outcome: 'capped', dedupe: false });
  });
});

describe('run tally — outcomes never bleed into each other', () => {
  it('starts at all zero', () => {
    expect(emptyTally()).toEqual({ posted: 0, skipped: 0, withheld: 0, capped: 0, failed: 0 });
  });

  it('increments only the counter for the given outcome', () => {
    let tally = emptyTally();
    tally = addOutcome(tally, 'skipped');
    expect(tally).toEqual({ posted: 0, skipped: 1, withheld: 0, capped: 0, failed: 0 });
    tally = addOutcome(tally, 'skipped');
    tally = addOutcome(tally, 'posted');
    tally = addOutcome(tally, 'withheld');
    tally = addOutcome(tally, 'capped');
    tally = addOutcome(tally, 'failed');
    expect(tally).toEqual({ posted: 1, skipped: 2, withheld: 1, capped: 1, failed: 1 });
  });

  it('a run of only skips and failures reports zero posted, distinctly — not folded into "succeeded"', () => {
    let tally = emptyTally();
    tally = addOutcome(tally, 'skipped');
    tally = addOutcome(tally, 'skipped');
    expect(formatTally(tally))
      .toBe('Done. 0 posted, 2 skipped, 0 withheld, 0 capped, 0 failed.');
  });

  it('a fully withheld fixture run does not read as if anything posted', () => {
    let tally = emptyTally();
    tally = addOutcome(tally, 'withheld');
    tally = addOutcome(tally, 'withheld');
    const line = formatTally(tally);
    expect(line).toBe('Done. 0 posted, 0 skipped, 2 withheld, 0 capped, 0 failed.');
    expect(line).toContain('0 posted');
  });

  it('a capped run does not read as if the capped picks posted', () => {
    let tally = emptyTally();
    tally = addOutcome(tally, 'posted');
    tally = addOutcome(tally, 'capped');
    tally = addOutcome(tally, 'capped');
    expect(formatTally(tally))
      .toBe('Done. 1 posted, 0 skipped, 0 withheld, 2 capped, 0 failed.');
  });

  it('formats every count distinctly', () => {
    const tally = { posted: 2, skipped: 1, withheld: 4, capped: 5, failed: 3 };
    expect(formatTally(tally))
      .toBe('Done. 2 posted, 1 skipped, 4 withheld, 5 capped, 3 failed.');
  });

  it('still starts with "Done." — the string the pipeline is monitored by', () => {
    expect(formatTally(emptyTally()).startsWith('Done.')).toBe(true);
  });
});

describe('describeSlate — an empty slate must not hide a refused one', () => {
  it('says nothing extra when the backend dropped nothing', () => {
    expect(describeSlate(3, 0)).toBe('3 eligible pick(s)');
  });

  it('reports the dropped count so a silenced slate is not read as a quiet night', () => {
    const line = describeSlate(0, 5);
    expect(line).toContain('0 eligible pick(s)');
    expect(line).toContain('5 dropped by the backend');
    // The distinction that matters: this cannot be confused with a genuinely
    // empty schedule.
    expect(line).not.toBe(describeSlate(0, 0));
  });

  it('reports drops alongside a partial slate too', () => {
    expect(describeSlate(2, 1)).toContain('2 eligible pick(s), 1 dropped by the backend');
  });

  it('treats a missing or nonsense count as no drops', () => {
    expect(describeSlate(1, 0)).toBe('1 eligible pick(s)');
    expect(describeSlate(1, -3)).toBe('1 eligible pick(s)');
  });
});
