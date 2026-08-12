import { describe, expect, it, vi } from 'vitest';
import { fetchBroll, pickBrollQuery } from './broll';

const deps = (over: Partial<Parameters<typeof fetchBroll>[2]> = {}) => ({
  get: vi.fn(), exists: () => false, write: vi.fn(), apiKey: 'k', ...over,
});

describe('pickBrollQuery', () => {
  it('returns an unbranded query — never a team or player name', () => {
    const q = pickBrollQuery('mlb');
    expect(q).toMatch(/baseball|stadium|crowd/i);
    expect(q).not.toMatch(/yankees|dodgers|white sox/i);
  });
});

describe('fetchBroll', () => {
  it('returns undefined without an api key rather than throwing', async () => {
    const d = deps({ apiKey: undefined });
    await expect(fetchBroll('baseball', '/tmp', d)).resolves.toBeUndefined();
    expect(d.get).not.toHaveBeenCalled();
  });

  it('returns undefined when the request fails — b-roll never blocks a render', async () => {
    const d = deps({ get: vi.fn().mockRejectedValue(new Error('429')) });
    await expect(fetchBroll('baseball', '/tmp', d)).resolves.toBeUndefined();
  });

  it('skips the network entirely when the clip is already cached', async () => {
    const d = deps({ exists: () => true });
    await expect(fetchBroll('baseball', '/tmp', d)).resolves.toContain('/tmp');
    expect(d.get).not.toHaveBeenCalled();
  });

  it('downloads and writes the first returned video file', async () => {
    const d = deps({
      get: vi.fn()
        .mockResolvedValueOnce({ data: { videos: [{ video_files: [{ link: 'http://v/1.mp4', width: 1080 }] }] } })
        .mockResolvedValueOnce({ data: Buffer.from('bytes') }),
    });
    const out = await fetchBroll('baseball', '/tmp', d);
    expect(out).toContain('/tmp');
    expect(d.write).toHaveBeenCalled();
  });

  it('returns undefined when the search yields no clips', async () => {
    const d = deps({ get: vi.fn().mockResolvedValue({ data: { videos: [] } }) });
    await expect(fetchBroll('baseball', '/tmp', d)).resolves.toBeUndefined();
  });
});
