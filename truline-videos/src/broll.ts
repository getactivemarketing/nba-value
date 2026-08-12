import { createHash } from 'crypto';
import { resolve } from 'path';

/**
 * Unbranded stock b-roll only.
 *
 * League game footage and Getty/AP editorial clips cannot be licensed for
 * betting promotion, and TikTok Content-ID mutes them. Queries here must never
 * name a team or player.
 */
const QUERIES: Record<string, string[]> = {
  mlb: ['baseball stadium night', 'baseball crowd', 'stadium floodlights'],
  nba: ['basketball court', 'arena crowd', 'basketball hoop night'],
};

/**
 * How many search results to choose from.
 *
 * Requesting one result means each query can only ever return one specific
 * clip, so the whole pipeline draws from `QUERIES.mlb.length` files — forever.
 * That is precisely the unoriginal-content signature the seeding below exists
 * to avoid, so the page has to be wide enough for the seed to have something
 * to choose between.
 */
const SEARCH_PAGE_SIZE = 20;

/** Byte 0 of sha1(seed) picks the query; a separate namespace picks the clip
 *  within that query's results, so the two choices don't move together. */
const seedByte = (namespace: string, seed: string): number =>
  createHash('sha1').update(`${namespace}:${seed}`).digest()[0];

/**
 * Deterministically varies the query within a sport's pool, keyed on `seed`
 * (pass the game_id). Every video otherwise shares one identical background
 * clip — TikTok's unoriginal-content classifier penalises exactly that, and
 * consecutive posts must not repeat a background. Omitting `seed` keeps the
 * old pool[0] behaviour for any caller that has no natural key to hash.
 */
export function pickBrollQuery(sport: string, seed?: string): string {
  const pool = QUERIES[sport.toLowerCase()] || QUERIES.mlb;
  if (!seed) return pool[0];
  const hash = createHash('sha1').update(seed).digest();
  return pool[hash[0] % pool.length];
}

export interface BrollDeps {
  get: (url: string, cfg: unknown) => Promise<{ data: unknown }>;
  exists: (path: string) => boolean;
  write: (path: string, body: Buffer) => void;
  apiKey?: string;
}

interface PexelsVideo { video_files?: { link?: string }[] }

/**
 * Downloads one background clip, caching it on disk.
 *
 * `seed` (the game_id) chooses BOTH the position in the result page and the
 * cache filename. Those two must stay tied together: the cache key used to be
 * hashed from the query alone, which was safe only while a query could yield
 * exactly one clip — now that it can yield twenty, a query-only key would let
 * two different clips collide on one filename and the first one downloaded
 * would be served forever.
 *
 * Never throws and never blocks a render: a missing key, an empty result set,
 * a clip with no usable file, or any network failure all return undefined and
 * the video renders on its flat background.
 */
export async function fetchBroll(
  query: string,
  cacheDir: string,
  deps: BrollDeps,
  seed?: string,
): Promise<string | undefined> {
  const cacheKey = `${query}::${seed ?? ''}`;
  const slug = createHash('sha1').update(cacheKey).digest('hex').slice(0, 12);
  const out = resolve(cacheDir, `broll_${slug}.mp4`);

  if (deps.exists(out)) return out;
  if (!deps.apiKey) return undefined;

  try {
    const search = await deps.get('https://api.pexels.com/videos/search', {
      headers: { Authorization: deps.apiKey },
      params: { query, orientation: 'portrait', per_page: SEARCH_PAGE_SIZE },
      timeout: 30000,
    }) as { data: { videos?: PexelsVideo[] } };

    const videos = search.data.videos ?? [];
    if (videos.length === 0) return undefined;

    const link = pickClipLink(videos, seed);
    if (!link) return undefined;

    const clip = await deps.get(link, { responseType: 'arraybuffer', timeout: 60000 });
    deps.write(out, Buffer.from(clip.data as ArrayBuffer));
    return out;
  } catch {
    return undefined;
  }
}

/**
 * The seeded choice within a result page. Starts at the seeded offset and
 * walks forward, so a result that carries no downloadable file degrades to
 * the next clip rather than to no b-roll at all — while staying entirely
 * deterministic for a given (seed, page) pair.
 */
function pickClipLink(videos: PexelsVideo[], seed?: string): string | undefined {
  const start = seed ? seedByte('clip', seed) % videos.length : 0;
  for (let i = 0; i < videos.length; i++) {
    const link = videos[(start + i) % videos.length]?.video_files?.[0]?.link;
    if (link) return link;
  }
  return undefined;
}
