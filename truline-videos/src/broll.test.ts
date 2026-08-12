import { describe, expect, it, vi } from 'vitest';
import { fetchBroll, isPreferredLength, pickBrollQuery, pickRendition } from './broll';

const deps = (over: Partial<Parameters<typeof fetchBroll>[2]> = {}) => ({
  get: vi.fn(), exists: () => false, write: vi.fn(), apiKey: 'k', ...over,
});

/** A search page of `n` distinct Pexels results. */
const page = (n: number) => ({
  data: { videos: Array.from({ length: n }, (_, i) => ({ video_files: [{ link: `http://v/${i}.mp4` }] })) },
});

/** deps whose search returns `n` results and whose download always succeeds. */
const searchDeps = (n: number, over: Partial<Parameters<typeof fetchBroll>[2]> = {}) => deps({
  get: vi.fn(async (url: string) => (
    url.includes('api.pexels.com') ? page(n) : { data: Buffer.from('bytes') }
  )),
  ...over,
});

const downloadedLink = (d: ReturnType<typeof searchDeps>): string =>
  (d.get as ReturnType<typeof vi.fn>).mock.calls[1][0] as string;

describe('pickBrollQuery', () => {
  it('returns an unbranded query — never a team or player name', () => {
    const q = pickBrollQuery('mlb');
    expect(q).toMatch(/baseball|stadium|crowd/i);
    expect(q).not.toMatch(/yankees|dodgers|white sox/i);
  });

  it('without a seed, keeps returning the first query in the pool', () => {
    expect(pickBrollQuery('mlb')).toBe(pickBrollQuery('mlb'));
  });

  it('is deterministic for a given seed — same game_id, same query', () => {
    const q1 = pickBrollQuery('mlb', 'game-123');
    const q2 = pickBrollQuery('mlb', 'game-123');
    expect(q1).toBe(q2);
  });

  it('varies the query across different seeds — no shared background across a slate', () => {
    const seeds = Array.from({ length: 12 }, (_, i) => `2026-08-11_TEAM_${i}`);
    const results = new Set(seeds.map((s) => pickBrollQuery('mlb', s)));
    expect(results.size).toBeGreaterThan(1);
  });

  it('a seeded query is still unbranded — never a team or player name', () => {
    const q = pickBrollQuery('mlb', 'game-456');
    expect(q).toMatch(/baseball|stadium|crowd/i);
    expect(q).not.toMatch(/yankees|dodgers|white sox/i);
  });
});

describe('fetchBroll — b-roll never blocks a render', () => {
  it('returns undefined without an api key rather than throwing', async () => {
    const d = deps({ apiKey: undefined });
    await expect(fetchBroll('baseball', '/tmp', d)).resolves.toBeUndefined();
    expect(d.get).not.toHaveBeenCalled();
  });

  it('returns undefined when the request fails', async () => {
    const d = deps({ get: vi.fn().mockRejectedValue(new Error('429')) });
    await expect(fetchBroll('baseball', '/tmp', d)).resolves.toBeUndefined();
  });

  it('returns undefined when the search yields no clips', async () => {
    const d = deps({ get: vi.fn().mockResolvedValue({ data: { videos: [] } }) });
    await expect(fetchBroll('baseball', '/tmp', d)).resolves.toBeUndefined();
  });

  it('returns undefined when the search yields no videos key at all', async () => {
    const d = deps({ get: vi.fn().mockResolvedValue({ data: {} }) });
    await expect(fetchBroll('baseball', '/tmp', d)).resolves.toBeUndefined();
  });

  it('returns undefined when no result carries a downloadable file', async () => {
    const d = deps({
      get: vi.fn().mockResolvedValue({ data: { videos: [{ video_files: [] }, {}] } }),
    });
    await expect(fetchBroll('baseball', '/tmp', d, 'seed')).resolves.toBeUndefined();
  });

  it('skips the network entirely when the clip is already cached', async () => {
    const d = deps({ exists: () => true });
    await expect(fetchBroll('baseball', '/tmp', d)).resolves.toContain('/tmp');
    expect(d.get).not.toHaveBeenCalled();
  });

  it('downloads and writes the selected video file', async () => {
    const d = searchDeps(1);
    const out = await fetchBroll('baseball', '/tmp', d);
    expect(out).toContain('/tmp');
    expect(d.write).toHaveBeenCalled();
  });
});

describe('fetchBroll — the pool is a page of results, not one clip', () => {
  it('asks Pexels for many results, not one', async () => {
    const d = searchDeps(20);
    await fetchBroll('baseball', '/tmp', d, 'game-1');
    const params = (d.get as ReturnType<typeof vi.fn>).mock.calls[0][1].params;
    expect(params.per_page).toBeGreaterThan(1);
  });

  it('different game_ids draw different clips out of the same query', async () => {
    const links = new Set<string>();
    for (let i = 0; i < 12; i++) {
      const d = searchDeps(20);
      await fetchBroll('baseball stadium night', '/tmp', d, `2026-08-12_GAME_${i}`);
      links.add(downloadedLink(d));
    }
    // The old code could only ever return videos[0] — this set would be 1.
    expect(links.size).toBeGreaterThan(1);
  });

  it('the same game_id always draws the same clip — reproducible renders', async () => {
    const first = searchDeps(20);
    const second = searchDeps(20);
    await fetchBroll('baseball', '/tmp', first, 'game-abc');
    await fetchBroll('baseball', '/tmp', second, 'game-abc');
    expect(downloadedLink(first)).toBe(downloadedLink(second));
  });

  it('the clip choice is independent of the query choice — same seed, different query', async () => {
    // Both selections are seeded on game_id; they must not be the same hash,
    // or the clip index would be pinned to the query index.
    const links = new Set<string>();
    for (const query of ['baseball stadium night', 'baseball crowd', 'stadium floodlights']) {
      const d = searchDeps(20);
      await fetchBroll(query, '/tmp', d, 'game-abc');
      links.add(downloadedLink(d));
    }
    // Same seed, so the offset is identical across queries — proving the
    // choice is seeded, not query-derived.
    expect(links.size).toBe(1);
  });

  it('falls forward past a result with no downloadable file rather than giving up', async () => {
    const d = deps({
      get: vi.fn(async (url: string) => (
        url.includes('api.pexels.com')
          ? { data: { videos: [{ video_files: [] }, { video_files: [{ link: 'http://v/ok.mp4' }] }] } }
          : { data: Buffer.from('bytes') }
      )),
    });
    await expect(fetchBroll('baseball', '/tmp', d, 'game-abc')).resolves.toContain('/tmp');
    expect(downloadedLink(d)).toBe('http://v/ok.mp4');
  });

  it('stays on videos[0] for a caller with no seed', async () => {
    const d = searchDeps(20);
    await fetchBroll('baseball', '/tmp', d);
    expect(downloadedLink(d)).toBe('http://v/0.mp4');
  });
});

describe('fetchBroll — cache key covers whatever selects the clip', () => {
  const cachePath = async (query: string, seed?: string) => {
    const d = searchDeps(20);
    return await fetchBroll(query, '/tmp', d, seed);
  };

  it('two seeds that pick different clips from one query do not collide on disk', async () => {
    const a = await cachePath('baseball stadium night', 'game-a');
    const b = await cachePath('baseball stadium night', 'game-b');
    expect(a).not.toBe(b);
  });

  it('the same query and seed resolve to the same cached file', async () => {
    expect(await cachePath('baseball crowd', 'game-a'))
      .toBe(await cachePath('baseball crowd', 'game-a'));
  });

  it('different queries never share a cache file either', async () => {
    expect(await cachePath('baseball crowd', 'game-a'))
      .not.toBe(await cachePath('stadium floodlights', 'game-a'));
  });

  it('a cached path is returned before any network call, for a seeded fetch too', async () => {
    const d = deps({ exists: () => true });
    await expect(fetchBroll('baseball', '/tmp', d, 'game-a')).resolves.toContain('/tmp');
    expect(d.get).not.toHaveBeenCalled();
  });
});

describe('pickRendition — the background must not be an upscaled thumbnail', () => {
  /** Pexels returns video_files sorted ASCENDING by resolution. */
  const ladder = [
    { link: 'http://v/360.mp4', width: 360, height: 640 },
    { link: 'http://v/540.mp4', width: 540, height: 960 },
    { link: 'http://v/720.mp4', width: 720, height: 1280 },
    { link: 'http://v/1080.mp4', width: 1080, height: 1920 },
    { link: 'http://v/2160.mp4', width: 2160, height: 3840 },
  ];

  it('never takes video_files[0], which is always the 360p preview', () => {
    expect(pickRendition(ladder)).not.toBe('http://v/360.mp4');
  });

  it('takes the smallest rendition that fills the 1080x1920 frame', () => {
    expect(pickRendition(ladder)).toBe('http://v/1080.mp4');
  });

  it('does not take the largest either — 4K is tens of MB for a 15%-opacity background', () => {
    expect(pickRendition(ladder)).not.toBe('http://v/2160.mp4');
  });

  it('is order-independent, since the API ordering is not a contract', () => {
    expect(pickRendition([...ladder].reverse())).toBe('http://v/1080.mp4');
  });

  it('falls back to the largest available when nothing reaches the frame', () => {
    expect(pickRendition(ladder.slice(0, 3))).toBe('http://v/720.mp4');
  });

  it('ignores renditions with no link', () => {
    expect(pickRendition([{ width: 1080, height: 1920 }, ...ladder.slice(0, 2)]))
      .toBe('http://v/540.mp4');
  });

  it('survives missing dimensions rather than throwing', () => {
    expect(pickRendition([{ link: 'http://v/x.mp4' }])).toBe('http://v/x.mp4');
    expect(pickRendition([])).toBeUndefined();
    expect(pickRendition(undefined)).toBeUndefined();
  });

  it('is what fetchBroll actually downloads', async () => {
    const get = vi.fn()
      .mockResolvedValueOnce({ data: { videos: [{ video_files: ladder }] } })
      .mockResolvedValueOnce({ data: new ArrayBuffer(8) });
    await fetchBroll('q', '/tmp', deps({ get }), 'seed');
    expect(get.mock.calls[1][0]).toBe('http://v/1080.mp4');
  });
});

describe('clip length — the cache keeps one clip per game for a season', () => {
  const files = [{ link: 'http://v/hd.mp4', width: 1080, height: 1920 }];
  const vid = (duration: number | null | undefined, link: string) => ({
    duration, video_files: [{ ...files[0], link }],
  });

  it('accepts the sane band and rejects either extreme', () => {
    expect(isPreferredLength({ duration: 12 })).toBe(true);
    expect(isPreferredLength({ duration: 8 })).toBe(true);
    expect(isPreferredLength({ duration: 30 })).toBe(true);
    expect(isPreferredLength({ duration: 88 })).toBe(false);  // the 77MB case
    expect(isPreferredLength({ duration: 3 })).toBe(false);
  });

  it('treats a missing or nonsense duration as not-preferred, never as a crash', () => {
    expect(isPreferredLength({})).toBe(false);
    expect(isPreferredLength({ duration: null })).toBe(false);
    expect(isPreferredLength({ duration: NaN })).toBe(false);
    expect(isPreferredLength({ duration: Infinity })).toBe(false);
  });

  it('skips an over-long clip in favour of a short one', async () => {
    const get = vi.fn()
      .mockResolvedValueOnce({ data: { videos: [vid(88, 'http://v/long.mp4'), vid(12, 'http://v/short.mp4')] } })
      .mockResolvedValueOnce({ data: new ArrayBuffer(8) });
    await fetchBroll('q', '/tmp', deps({ get }), 'seed-a');
    expect(get.mock.calls[1][0]).toBe('http://v/short.mp4');
  });

  it('still returns b-roll when every result is over-long', async () => {
    const get = vi.fn()
      .mockResolvedValueOnce({ data: { videos: [vid(88, 'http://v/a.mp4'), vid(120, 'http://v/b.mp4')] } })
      .mockResolvedValueOnce({ data: new ArrayBuffer(8) });
    const out = await fetchBroll('q', '/tmp', deps({ get }), 'seed-a');
    expect(out).toBeDefined();
    expect(['http://v/a.mp4', 'http://v/b.mp4']).toContain(get.mock.calls[1][0]);
  });

  it('still varies by seed within the preferred set', async () => {
    const page = Array.from({ length: 12 }, (_, i) => vid(10 + (i % 5), `http://v/${i}.mp4`));
    const chosen = new Set<string>();
    for (const seed of ['g1', 'g2', 'g3', 'g4', 'g5', 'g6']) {
      const get = vi.fn()
        .mockResolvedValueOnce({ data: { videos: page } })
        .mockResolvedValueOnce({ data: new ArrayBuffer(8) });
      await fetchBroll('q', '/tmp', deps({ get }), seed);
      chosen.add(get.mock.calls[1][0]);
    }
    expect(chosen.size).toBeGreaterThan(1);
  });
});
