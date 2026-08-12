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

interface PexelsFile { link?: string; width?: number | null; height?: number | null }
interface PexelsVideo { video_files?: PexelsFile[]; duration?: number | null }

/**
 * Longest background clip worth downloading, in seconds.
 *
 * Now that <Loop> actually loops (it was previously handed the composition
 * duration as its iteration length, so it ran once), a short clip covers a
 * whole video. Length is the dominant term in file size at 1080x1920: an 88s
 * result weighs 77MB against roughly 10MB for a 12s one, and the cache keeps
 * a clip per game for a season. Preference, not a filter — a page of nothing
 * but long clips still yields b-roll.
 */
const MAX_PREFERRED_CLIP_SECONDS = 30;

/**
 * Shortest clip worth preferring. Below this the loop turnover gets obvious
 * enough to read as a glitch behind a 35-second read.
 */
const MIN_PREFERRED_CLIP_SECONDS = 8;

/** Whether a result is in the size-sane band the seeded walk tries first. */
export function isPreferredLength(video: PexelsVideo): boolean {
  const d = video.duration;
  if (typeof d !== 'number' || !Number.isFinite(d)) return false;
  return d >= MIN_PREFERRED_CLIP_SECONDS && d <= MAX_PREFERRED_CLIP_SECONDS;
}

/** The frame the composition renders at. A background rendition smaller than
 *  this gets upscaled by the renderer. */
const TARGET_WIDTH = 1080;
const TARGET_HEIGHT = 1920;

/**
 * Picks a rendition of one clip.
 *
 * Pexels returns `video_files` sorted ASCENDING by resolution, so `[0]` is
 * always the 360x640 preview — which the renderer then upscales 3x into a
 * 1080x1920 frame, and soft blocky footage reads as low-effort content even
 * at 15% opacity. Taking the last entry is no better: the top rendition is
 * routinely 2160x3840, tens of megabytes, downloaded on every cache miss for
 * a background nobody looks at directly.
 *
 * So: the SMALLEST rendition that still meets the composition's frame. Falls
 * back to the largest available when nothing does, which at least upscales
 * from as high a source as the clip offers.
 */
export function pickRendition(files: PexelsFile[] | undefined): string | undefined {
  const usable = (files ?? []).filter((f) => f.link);
  if (usable.length === 0) return undefined;

  const dims = (f: PexelsFile) => ({ w: f.width ?? 0, h: f.height ?? 0 });
  const meets = usable.filter((f) => {
    const { w, h } = dims(f);
    return w >= TARGET_WIDTH && h >= TARGET_HEIGHT;
  });

  const byArea = (a: PexelsFile, b: PexelsFile) =>
    dims(a).w * dims(a).h - dims(b).w * dims(b).h;

  if (meets.length > 0) return meets.sort(byArea)[0].link;
  return usable.sort(byArea)[usable.length - 1].link;
}

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
 *
 * Two passes, both walking from the same offset: length-appropriate results
 * first, then everything. The seed still decides, so different game_ids still
 * get different clips; the second pass only exists so a page of exclusively
 * long or untimed results yields b-roll rather than none.
 */
function pickClipLink(videos: PexelsVideo[], seed?: string): string | undefined {
  const start = seed ? seedByte('clip', seed) % videos.length : 0;

  const walk = (accept: (v: PexelsVideo) => boolean): string | undefined => {
    for (let i = 0; i < videos.length; i++) {
      const video = videos[(start + i) % videos.length];
      if (!video || !accept(video)) continue;
      const link = pickRendition(video.video_files);
      if (link) return link;
    }
    return undefined;
  };

  return walk(isPreferredLength) ?? walk(() => true);
}
