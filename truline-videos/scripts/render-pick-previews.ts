/**
 * Render and publish pre-game pick previews.
 *
 * THIS SCRIPT POSTS PUBLICLY TO TIKTOK AND INSTAGRAM. With a BLOTATO_API_KEY,
 * account IDs and a publishable TTS provider in .env, a plain run is a live
 * publish — not a rehearsal. Two ways to render without publishing:
 *
 *   DRY_RUN=1 npx tsx scripts/render-pick-previews.ts
 *       live API data, renders kept on disk, nothing uploaded.
 *
 *   PICK_PREVIEWS_FIXTURE=./some.json npx tsx scripts/render-pick-previews.ts
 *       reads a local payload shaped like the endpoint's response
 *       ({ generated_at, previews }) instead of calling the API — useful
 *       before today's snapshots have landed (they land ~30-45min before
 *       first pitch). Uploading is REFUSED outright in this mode and cannot
 *       be re-enabled: fixture JSON is hand-written or stale and has never
 *       been through the backend's NarrationContractError guard, so it must
 *       not be able to reach a public account.
 *
 * Both refusals, the per-run post cap (MAX_POSTS_PER_RUN, default 3) and the
 * lead-time gate are decided in ONE place — mayPublish() in
 * src/publish-guard.ts. Do not add a publish condition anywhere else.
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
import { buildPreviewCaption } from '../src/caption';
import { blotatoConfigFromEnv, uploadToBlotato, type BlotatoConfig } from '../src/blotato';
import {
  describeDecision, dryRunFromEnv, mayPublish, maxPostsPerRunFromEnv, refusedBeforeRender,
} from '../src/publish-guard';
import { addOutcome, decidePreviewResult, emptyTally, formatTally, type PreviewResult } from '../src/preview-outcome';
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
 *
 * LOAD-BEARING PIN: every one of those behaviours is Remotion INTERNALS, not
 * public API — `staticHash`/the "/public" mount point in @remotion/bundler and
 * getAbsoluteSrc's resolution rules in remotion. Nothing here is covered by
 * semver, so a patch bump can silently break the audio and b-roll of every
 * render (which still exits 0 — you get a silent video, not an error). That is
 * why remotion, @remotion/bundler, @remotion/cli, @remotion/google-fonts,
 * react and react-dom are pinned to EXACT versions in package.json with no
 * caret. Do not re-loosen them in a dependency sweep; upgrading means bumping
 * all four Remotion packages together and re-rendering one preview to confirm
 * the audio and background actually made it into the file.
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
 * nothing to render this run, not a crash.
 *
 * `fromFixture` is carried out of here rather than re-read from the
 * environment later, so the flag that forbids publishing is the same fact
 * that chose the data source.
 */
async function loadPreviews(): Promise<{ previews: ApiPreview[]; fromFixture: boolean }> {
  const fixture = process.env.PICK_PREVIEWS_FIXTURE;
  if (fixture) {
    console.log(`Reading previews from local fixture: ${fixture}`);
    const data = JSON.parse(readFileSync(fixture, 'utf-8'));
    return { previews: data.previews || [], fromFixture: true };
  }

  try {
    const resp = await axios.get(`${API_BASE}/mlb/video/pick-previews?days=1`, { timeout: 20000 });
    return { previews: resp.data.previews || [], fromFixture: false };
  } catch (err: any) {
    const status = err?.response?.status;
    const detail = err?.response?.data?.detail || err.message;
    console.error(`Failed to fetch pick previews (${status ?? 'network error'}): ${detail}`);
    return { previews: [], fromFixture: false };
  }
}

/** Everything about the RUN that feeds the publish gate. Assembled once in
 *  main() and threaded through, so no per-preview code re-reads the
 *  environment and reaches a different conclusion. */
interface RunGate {
  fromFixture: boolean;
  dryRun: boolean;
  maxPostsPerRun: number;
}

/** One preview end to end: narrate, fetch b-roll, render, gate, upload.
 *  Returns the outcome AND whether it is safe to dedupe — see
 *  decidePreviewResult in src/preview-outcome.ts for the rule. */
async function processPreview(
  preview: ApiPreview,
  tts: TtsAdapter,
  blotatoCfg: BlotatoConfig,
  gate: RunGate,
  postsThisRun: number,
): Promise<PreviewResult> {
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

  // 2. b-roll (optional — absence never blocks a render). Both the query and
  // the clip chosen within that query's results are seeded on game_id, so
  // consecutive posts in a slate don't share a background clip.
  const brollPath = await fetchBroll(
    pickBrollQuery('mlb', preview.game_id),
    BROLL_DIR,
    {
      get: (url, cfg) => axios.get(url, cfg as never),
      exists: existsSync,
      write: (p, b) => writeFileSync(p, b),
      apiKey: process.env.PEXELS_API_KEY,
    },
    preview.game_id,
  );

  // The b-roll clip is shorter than the video, so <Loop> needs ONE
  // iteration's length — the clip's own duration, not the composition's.
  // Measured the same way the narration is. A clip ffprobe cannot read must
  // not take the render down with it: fall back to no loop (plays once).
  let brollDurationInFrames: number | undefined;
  if (brollPath) {
    try {
      brollDurationInFrames = Math.round(measureSeconds(brollPath) * FPS);
    } catch (err: any) {
      console.warn(`Could not measure b-roll ${brollPath} — it will play once, un-looped: ${err?.message || err}`);
    }
  }

  // 3. render
  const outPath = resolve(OUT_DIR, `${preview.game_id}.mp4`);
  const propsFile = resolve(OUT_DIR, `${preview.game_id}_props.json`);
  writeFileSync(propsFile, JSON.stringify({
    beats,
    teamColor: teamColor(preview.team_abbr),
    logoUrl: preview.logo_url,
    brollSrc: brollPath ? toPublicSrc(brollPath) : undefined,
    brollDurationInFrames,
  }));
  execSync(
    `npx remotion render src/index.ts pick-preview "${outPath}" --props="${propsFile}"`,
    { cwd: resolve(__dirname, '..'), stdio: 'inherit', timeout: 300000 },
  );

  // 4. re-check the gate at UPLOAD time — rendering just consumed minutes
  const minutesLeft = (new Date(preview.game_time).getTime() - Date.now()) / 60000;
  const clearance = mayPublish({
    adapterPublishable: tts.publishable,
    minutesToFirstPitch: minutesLeft,
    fixture: gate.fromFixture,
    dryRun: gate.dryRun,
    postsThisRun,
    maxPostsPerRun: gate.maxPostsPerRun,
  });
  if (!clearance.allowed) {
    console.log(
      `WITHHELD upload ${preview.game_id}: ${describeDecision(clearance)} ` +
      `(publishable=${tts.publishable}, ${minutesLeft.toFixed(0)}min to first pitch). ` +
      `Render kept at ${outPath}`,
    );
    return decidePreviewResult(clearance, null);
  }

  const caption = buildPreviewCaption({
    teamName: preview.team_name,
    oddsAmerican: preview.odds_american,
    turnNarration: preview.beats.find((b) => b.key === 'turn')?.narration,
  });

  // Defence in depth: the backend enforces the banned-word contract and
  // fails closed (NarrationContractError), so the realistic failure mode is
  // "nothing renders", not "banned word ships" — but this caption is the one
  // thing that actually posts publicly, and it embeds the turn beat's
  // narration unvalidated. Cheap insurance against a backend regression.
  if (/edge/i.test(caption)) {
    console.error(`REFUSED upload ${preview.game_id}: caption contains banned word "edge":\n${caption}`);
    return decidePreviewResult(clearance, null);
  }

  const upload = await uploadToBlotato(outPath, caption, blotatoCfg);
  const result = decidePreviewResult(clearance, upload);
  if (result.outcome === 'posted') {
    console.log(`Posted: ${preview.game_id} -> ${upload.posted.join(', ')}`);
  } else {
    console.error(
      `Upload for ${preview.game_id} did not confirm any platform post ` +
      `(uploaded=${upload.uploaded}, posted=[${upload.posted.join(', ')}]) — leaving un-deduped for retry.`,
    );
  }
  return result;
}

async function main() {
  mkdirSync(AUDIO_DIR, { recursive: true });
  mkdirSync(BROLL_DIR, { recursive: true });
  mkdirSync(OUT_DIR, { recursive: true });
  const tts = selectAdapter(process.env);
  console.log(`TTS provider: ${tts.id} (publishable: ${tts.publishable})`);

  const blotatoCfg = blotatoConfigFromEnv();
  if (blotatoCfg.apiKey && !blotatoCfg.tiktokAccountId && !blotatoCfg.instagramAccountId) {
    console.warn(
      'WARNING: BLOTATO_API_KEY is set but neither BLOTATO_TIKTOK_ACCOUNT_ID nor ' +
      'BLOTATO_INSTAGRAM_ACCOUNT_ID is configured. No platform can confirm a post this run — ' +
      'every upload will be treated as failed (and left un-deduped) rather than posted.',
    );
  }

  const posted = loadPosted();
  const { previews, fromFixture } = await loadPreviews();
  console.log(`${previews.length} eligible pick(s)`);

  const gate: RunGate = {
    fromFixture,
    dryRun: dryRunFromEnv(),
    maxPostsPerRun: maxPostsPerRunFromEnv(),
  };
  if (fromFixture) {
    console.log(
      'FIXTURE MODE: uploading is refused for this entire run. Fixture data ' +
      'never passed the backend narration contract, so it must not reach a ' +
      'public account. Renders will still be produced and kept on disk.',
    );
  } else if (gate.dryRun) {
    console.log('DRY_RUN: renders will be produced and kept on disk; nothing will be uploaded.');
  }
  console.log(`Post cap: ${gate.maxPostsPerRun} per run`);

  let tally = emptyTally();
  let postsThisRun = 0;

  for (const preview of previews) {
    if (posted.includes(preview.game_id)) continue;

    // Ask the gate BEFORE spending render time. The cap is the only refusal
    // worth acting on this early: a capped preview's render would just be
    // thrown away (nothing reuses the mp4 — the next run re-renders from
    // fresh odds), whereas a fixture/dry-run/lead-time render is still
    // wanted on disk. Capped picks are never deduped, so the next run picks
    // them up exactly as if this run had not seen them.
    const preflight = mayPublish({
      adapterPublishable: tts.publishable,
      minutesToFirstPitch: (new Date(preview.game_time).getTime() - Date.now()) / 60000,
      fixture: gate.fromFixture,
      dryRun: gate.dryRun,
      postsThisRun,
      maxPostsPerRun: gate.maxPostsPerRun,
    });
    if (refusedBeforeRender(preflight)) {
      console.log(
        `CAPPED ${preview.game_id}: ${describeDecision(preflight)} ` +
        `(${postsThisRun}/${gate.maxPostsPerRun}). Not rendered, not deduped — eligible next run.`,
      );
      tally = addOutcome(tally, 'capped');
      continue;
    }

    // One bad pick — banned narration copy that slipped past the backend,
    // a synthesis failure, a render crash — must not take down the rest of
    // the slate. Log clearly and move on to the next pick.
    try {
      const result = await processPreview(preview, tts, blotatoCfg, gate, postsThisRun);
      tally = addOutcome(tally, result.outcome);
      if (result.outcome === 'posted') postsThisRun += 1;
      if (result.dedupe) {
        posted.push(preview.game_id);
        savePosted(posted);
      }
    } catch (err: any) {
      tally = addOutcome(tally, 'failed');
      console.error(`FAILED ${preview.game_id}: ${err?.message || err}`);
    }
  }

  console.log(formatTally(tally));
}

main().catch((err: any) => {
  console.error(`Fatal error: ${err?.message || err}`);
  process.exit(1);
});
