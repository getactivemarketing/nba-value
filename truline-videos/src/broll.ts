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

export async function fetchBroll(
  query: string,
  cacheDir: string,
  deps: BrollDeps,
): Promise<string | undefined> {
  const slug = createHash('sha1').update(query).digest('hex').slice(0, 12);
  const out = resolve(cacheDir, `broll_${slug}.mp4`);

  if (deps.exists(out)) return out;
  if (!deps.apiKey) return undefined;

  try {
    const search = await deps.get('https://api.pexels.com/videos/search', {
      headers: { Authorization: deps.apiKey },
      params: { query, orientation: 'portrait', per_page: 1 },
      timeout: 30000,
    }) as { data: { videos?: { video_files: { link: string }[] }[] } };

    const link = search.data.videos?.[0]?.video_files?.[0]?.link;
    if (!link) return undefined;

    const clip = await deps.get(link, { responseType: 'arraybuffer', timeout: 60000 });
    deps.write(out, Buffer.from(clip.data as ArrayBuffer));
    return out;
  } catch {
    return undefined;
  }
}
