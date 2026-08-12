/**
 * Render and publish pre-game pick previews.
 *
 * Run manually: npx tsx scripts/render-pick-previews.ts
 * Dry-run against hand-built data instead of the live API by setting
 * PICK_PREVIEWS_FIXTURE to a local JSON file shaped like the endpoint's
 * response ({ generated_at, previews }) — useful before today's snapshots
 * have landed (they land ~30-45min before first pitch) or while the
 * pick-previews endpoint hasn't shipped to the deployed backend yet.
 *
 * Mirrors render-celebrations.ts, which is left untouched. Reuses its Blotato
 * upload path and rendered.json-style dedupe.
 */

import { config } from 'dotenv';
import axios from 'axios';
import { execFileSync, execSync } from 'child_process';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'fs';
import { relative, resolve, sep } from 'path';

import { selectAdapter, type TtsAdapter } from '../src/tts';
import { fetchBroll, pickBrollQuery } from '../src/broll';
import { teamColor } from '../src/teams';
import { blotatoConfigFromEnv, uploadToBlotato } from '../src/blotato';
import { mayPublish } from '../src/publish-guard';
import { FPS } from '../src/constants';
import type { BeatClip } from '../src/compositions/PickPreview';

config({ path: resolve(__dirname, '..', '.env') });

const API_BASE = 'https://nba-value-production.up.railway.app/api/v1';

const OUT_DIR = resolve(__dirname, '..', 'rendered', 'previews');
// Generated audio and b-roll must live under Remotion's public/ dir (its
// default, since no --public-dir override is passed) — see the toPublicSrc
// comment below for why. Render OUTPUT (mp4 + props.json) stays under
// rendered/previews/, same as render-celebrations.ts, since it never needs
// to be servable.
const PUBLIC_DIR = resolve(__dirname, '..', 'public');
const AUDIO_DIR = resolve(PUBLIC_DIR, 'previews-audio');
const BROLL_DIR = resolve(PUBLIC_DIR, 'previews-broll');
const POSTED_FILE = resolve(__dirname, '..', 'previews-posted.json');

interface ApiBeat { key: string; narration: string; overlay: Record<string, string>; }
interface ApiPreview {
  game_id: string; game_date: string; game_time: string;
  team_abbr: string; team_name: string;
  logo_url: string; odds_american: number; beats: ApiBeat[];
}

/**
 * Path Remotion's <Audio>/<OffthreadVideo> can actually resolve during
 * rendering. Two other things were tried and both failed against @remotion/
 * renderer 4.0.448:
 *   - a bare OS absolute path ("/Users/...") is resolved by the *browser*
 *     side (getAbsoluteSrc in remotion/dist/cjs/absolute-src.js) against the
 *     dev-server origin, producing a mangled URL and a 404.
 *   - a "file://" URI passes the browser side, but the renderer's own
 *     server-side asset-download step explicitly rejects any scheme other
 *     than http(s): "Can only download URLs starting with http:// or
 *     https://".
 *   - `--public-dir` pointed at a per-run temp folder: the bundler snapshots
 *     the public dir once and appeared to reuse a stale bundle across
 *     back-to-back renders with different --public-dir values, 404ing on
 *     freshly-written files.
 * What works: files placed under Remotion's *default* public/ dir (no
 * --public-dir override, so it is always the same folder Remotion already
 * expects) and referenced with a leading "/public/" — that is the actual
 * mount point (see `staticHash` in @remotion/bundler/dist/bundle.js: the
 * bundler copies public/ to `<bundle>/public` and mounts it at "/public",
 * which is what staticFile() prefixes onto its argument via
 * window.remotion_staticBase at runtime). getAbsoluteSrc then resolves that
 * path against the dev server's own origin.
 */
const toPublicSrc = (absolutePath: string): string =>
  `/public/${relative(PUBLIC_DIR, absolutePath).split(sep).join('/')}`;

const loadPosted = (): string[] =>
  existsSync(POSTED_FILE) ? JSON.parse(readFileSync(POSTED_FILE, 'utf-8')) : [];
const savePosted = (ids: string[]) =>
  writeFileSync(POSTED_FILE, JSON.stringify(ids, null, 2));

/**
 * Seconds of audio at `path`, via ffprobe.
 *
 * @remotion/media-utils' getAudioDurationInSeconds only works in a browser
 * (it throws "only available in the browser" the instant `document` is
 * undefined) — that rules it out for a plain Node orchestrator script, so
 * duration is measured with ffprobe instead.
 */
function measureSeconds(path: string): number {
  const out = execFileSync('ffprobe', [
    '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', path,
  ], { encoding: 'utf-8' });
  const seconds = parseFloat(out.trim());
  if (!Number.isFinite(seconds) || seconds <= 0) {
    throw new Error(`ffprobe could not read a duration from ${path}: "${out.trim()}"`);
  }
  return seconds;
}

/**
 * Fetches eligible pick previews. A non-200 response, or the request failing
 * outright, is logged as one clear line and treated as an empty slate —
 * nothing to render this run, not a crash. Set PICK_PREVIEWS_FIXTURE to dry
 * -run against a local payload instead of the live endpoint.
 */
async function loadPreviews(): Promise<ApiPreview[]> {
  const fixture = process.env.PICK_PREVIEWS_FIXTURE;
  if (fixture) {
    console.log(`Reading previews from local fixture: ${fixture}`);
    const data = JSON.parse(readFileSync(fixture, 'utf-8'));
    return data.previews || [];
  }

  try {
    const resp = await axios.get(`${API_BASE}/mlb/video/pick-previews?days=1`, { timeout: 20000 });
    return resp.data.previews || [];
  } catch (err: any) {
    const status = err?.response?.status;
    const detail = err?.response?.data?.detail || err.message;
    console.error(`Failed to fetch pick previews (${status ?? 'network error'}): ${detail}`);
    return [];
  }
}

/** One preview end to end: narrate, fetch b-roll, render, gate, upload. */
async function processPreview(preview: ApiPreview, tts: TtsAdapter): Promise<'posted' | 'skipped'> {
  // 1. narration, one clip per beat. Extension follows the adapter: `say`
  // writes a WAV container (its fixed --data-format only opens under .wav/
  // .caf — .mp3 fails outright), elevenlabs/openai both return real MP3 bytes.
  const audioExt = tts.id === 'say' ? 'wav' : 'mp3';
  const beats: BeatClip[] = [];
  for (const [i, beat] of preview.beats.entries()) {
    const audioPath = resolve(AUDIO_DIR, `${preview.game_id}_${i}_${beat.key}.${audioExt}`);
    if (!existsSync(audioPath)) await tts.synthesize(beat.narration, audioPath);
    const seconds = measureSeconds(audioPath);
    beats.push({
      key: beat.key,
      overlay: beat.overlay,
      audioSrc: toPublicSrc(audioPath),
      // Half-second of air after each beat so it does not clip into the next.
      durationInFrames: Math.round((seconds + 0.5) * FPS),
    });
  }

  // 2. b-roll (optional — absence never blocks a render). Query is seeded on
  // game_id so consecutive posts in a slate don't share a background clip.
  const brollPath = await fetchBroll(pickBrollQuery('mlb', preview.game_id), BROLL_DIR, {
    get: (url, cfg) => axios.get(url, cfg as never),
    exists: existsSync,
    write: (p, b) => writeFileSync(p, b),
    apiKey: process.env.PEXELS_API_KEY,
  });

  // 3. render
  const outPath = resolve(OUT_DIR, `${preview.game_id}.mp4`);
  const propsFile = resolve(OUT_DIR, `${preview.game_id}_props.json`);
  writeFileSync(propsFile, JSON.stringify({
    beats,
    teamColor: teamColor(preview.team_abbr),
    logoUrl: preview.logo_url,
    brollSrc: brollPath ? toPublicSrc(brollPath) : undefined,
  }));
  execSync(
    `npx remotion render src/index.ts pick-preview "${outPath}" --props="${propsFile}"`,
    { cwd: resolve(__dirname, '..'), stdio: 'inherit', timeout: 300000 },
  );

  // 4. re-check the gate at UPLOAD time — rendering just consumed minutes
  const minutesLeft = (new Date(preview.game_time).getTime() - Date.now()) / 60000;
  if (!mayPublish(tts.publishable, minutesLeft)) {
    console.log(
      `SKIP upload ${preview.game_id}: publishable=${tts.publishable}, ` +
      `${minutesLeft.toFixed(0)}min to first pitch. Render kept at ${outPath}`,
    );
    return 'skipped';
  }

  const caption = [
    `${preview.team_name} ML ${preview.odds_american > 0 ? '+' : ''}${preview.odds_american}.`,
    '',
    preview.beats.find((b) => b.key === 'turn')?.narration || '',
    '',
    'Not betting advice. 21+.',
    '#MLB #SportsAnalytics',
  ].filter(Boolean).join('\n');

  await uploadToBlotato(outPath, caption, blotatoConfigFromEnv());
  return 'posted';
}

async function main() {
  mkdirSync(AUDIO_DIR, { recursive: true });
  mkdirSync(BROLL_DIR, { recursive: true });
  mkdirSync(OUT_DIR, { recursive: true });
  const tts = selectAdapter(process.env);
  console.log(`TTS provider: ${tts.id} (publishable: ${tts.publishable})`);

  const posted = loadPosted();
  const previews = await loadPreviews();
  console.log(`${previews.length} eligible pick(s)`);

  let succeeded = 0;
  let failed = 0;

  for (const preview of previews) {
    if (posted.includes(preview.game_id)) continue;

    // One bad pick — banned narration copy that slipped past the backend,
    // a synthesis failure, a render crash — must not take down the rest of
    // the slate. Log clearly and move on to the next pick.
    try {
      const outcome = await processPreview(preview, tts);
      if (outcome === 'posted') {
        posted.push(preview.game_id);
        savePosted(posted);
        console.log(`Posted: ${preview.game_id}`);
      }
      succeeded++;
    } catch (err: any) {
      failed++;
      console.error(`FAILED ${preview.game_id}: ${err?.message || err}`);
    }
  }

  console.log(`Done. ${succeeded} succeeded, ${failed} failed.`);
}

main().catch((err: any) => {
  console.error(`Fatal error: ${err?.message || err}`);
  process.exit(1);
});
