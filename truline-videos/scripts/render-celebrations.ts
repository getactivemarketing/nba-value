/**
 * Check for new underdog ML wins and render celebration videos.
 *
 * Run manually: npx tsx scripts/render-celebrations.ts
 * Run via cron:  star/30 * * * * cd /path/to/truline-videos && npx tsx scripts/render-celebrations.ts
 * (replace "star" with asterisk)
 */

import { config } from 'dotenv';
import axios from 'axios';
import { execSync } from 'child_process';
import { readFileSync, writeFileSync, existsSync } from 'fs';
import { resolve } from 'path';

config({ path: resolve(__dirname, '..', '.env') });

const API_BASE = 'https://nba-value-production.up.railway.app/api/v1';
const BLOTATO_API = 'https://backend.blotato.com/v2';
const BLOTATO_API_KEY = process.env.BLOTATO_API_KEY || '';
const BLOTATO_TIKTOK_ACCOUNT_ID = process.env.BLOTATO_TIKTOK_ACCOUNT_ID || '';
const BLOTATO_INSTAGRAM_ACCOUNT_ID = process.env.BLOTATO_INSTAGRAM_ACCOUNT_ID || '';

const RENDERED_FILE = resolve(__dirname, '..', 'rendered.json');
const RENDERED_DIR = resolve(__dirname, '..', 'rendered');

const TEAM_COLORS: Record<string, string> = {
  ARI: '#A71930', ATL: '#CE1141', BAL: '#DF4601', BOS: '#BD3039',
  CHC: '#0E3386', CWS: '#27251F', CIN: '#C6011F', CLE: '#00385D',
  COL: '#33006F', DET: '#0C2340', HOU: '#002D62', KC: '#004687',
  LAA: '#BA0021', LAD: '#005A9C', MIA: '#00A3E0', MIL: '#12284B',
  MIN: '#002B5C', NYM: '#002D72', NYY: '#003087', OAK: '#003831',
  PHI: '#E81828', PIT: '#FDB827', SD: '#2F241D', SF: '#FD5A1E',
  SEA: '#005C5C', STL: '#C41E3A', TB: '#092C5C', TEX: '#003278',
  TOR: '#134A8E', WSH: '#AB0003',
  GSW: '#1D428A', LAL: '#552583', BKN: '#000000', BOS_NBA: '#007A33',
  MIA_NBA: '#98002E', MIL_NBA: '#00471B', DEN: '#0D2240', PHX: '#1D1160',
  DAL: '#0053BC', MEM: '#5D76A9', SAC: '#5B2B82', OKC: '#007DC3',
  CLE_NBA: '#860038', IND: '#002D62', ORL: '#0077C0', CHA: '#1D1160',
  CHI_NBA: '#CE1141', TOR_NBA: '#CE1141', POR: '#E03A3E', SAS: '#C4CED4',
  NOP: '#002B5C', NYK: '#006BB6', UTA: '#002B5C', WAS: '#002B5C',
  LAC: '#C8102E',
};

const TEAM_NAMES: Record<string, string> = {
  ARI: 'D-backs', ATL: 'Braves', BAL: 'Orioles', BOS: 'Red Sox',
  CHC: 'Cubs', CWS: 'White Sox', CIN: 'Reds', CLE: 'Guardians',
  COL: 'Rockies', DET: 'Tigers', HOU: 'Astros', KC: 'Royals',
  LAA: 'Angels', LAD: 'Dodgers', MIA: 'Marlins', MIL: 'Brewers',
  MIN: 'Twins', NYM: 'Mets', NYY: 'Yankees', OAK: 'Athletics',
  PHI: 'Phillies', PIT: 'Pirates', SD: 'Padres', SF: 'Giants',
  SEA: 'Mariners', STL: 'Cardinals', TB: 'Rays', TEX: 'Rangers',
  TOR: 'Blue Jays', WSH: 'Nationals',
  GSW: 'Warriors', LAL: 'Lakers', BKN: 'Nets', NYK: 'Knicks',
  MIA_NBA: 'Heat', MIL_NBA: 'Bucks', DEN: 'Nuggets', PHX: 'Suns',
  DAL: 'Mavericks', MEM: 'Grizzlies', SAC: 'Kings', OKC: 'Thunder',
  CLE_NBA: 'Cavaliers', IND: 'Pacers', ORL: 'Magic', CHA: 'Hornets',
  DET_NBA: 'Pistons', CHI_NBA: 'Bulls', TOR_NBA: 'Raptors', POR: 'Trail Blazers',
  SAS: 'Spurs', NOP: 'Pelicans', UTA: 'Jazz', WAS: 'Wizards',
  LAC: 'Clippers',
};

interface UnderdogWin {
  date: string | null;
  team: string;
  odds_american: number;
  profit: number;
  score: string | null;
}

function loadRendered(): string[] {
  if (!existsSync(RENDERED_FILE)) return [];
  return JSON.parse(readFileSync(RENDERED_FILE, 'utf-8'));
}

function saveRendered(ids: string[]) {
  writeFileSync(RENDERED_FILE, JSON.stringify(ids, null, 2));
}

async function fetchUnderdogWins(): Promise<UnderdogWin[]> {
  try {
    const resp = await axios.get(`${API_BASE}/mlb/evaluation/underdogs?days=2`, { timeout: 15000 });
    return resp.data.biggest_wins || [];
  } catch (err: any) {
    console.error('Failed to fetch underdogs:', err.message);
    return [];
  }
}

function renderVideo(props: Record<string, any>, outputPath: string) {
  const propsJson = JSON.stringify(props);
  // Write props to temp file to avoid shell escaping issues
  const propsFile = resolve(RENDERED_DIR, '_props.json');
  writeFileSync(propsFile, propsJson);
  const cmd = `npx remotion render src/index.ts model-hit "${outputPath}" --props="${propsFile}"`;
  console.log(`Rendering: ${outputPath}`);
  execSync(cmd, { cwd: resolve(__dirname, '..'), stdio: 'inherit', timeout: 120000 });
}

async function uploadToBlotato(videoPath: string, caption: string) {
  if (!BLOTATO_API_KEY) {
    console.log('[DRY-RUN] Would upload:', videoPath);
    console.log('[DRY-RUN] Caption:', caption);
    return;
  }

  const headers = {
    'blotato-api-key': BLOTATO_API_KEY,
    'Content-Type': 'application/json',
  };

  try {
    const videoData = readFileSync(videoPath);
    const filename = videoPath.split('/').pop() || 'video.mp4';

    const uploadResp = await axios.post(`${BLOTATO_API}/media/uploads`, { filename }, { headers, timeout: 60000 });
    const { presignedUrl, publicUrl } = uploadResp.data;

    await axios.put(presignedUrl, videoData, {
      headers: { 'Content-Type': 'video/mp4' },
      timeout: 60000,
      maxBodyLength: Infinity,
    });

    console.log(`Uploaded: ${publicUrl}`);

    for (const [platform, accountId] of [
      ['tiktok', BLOTATO_TIKTOK_ACCOUNT_ID],
      ['instagram', BLOTATO_INSTAGRAM_ACCOUNT_ID],
    ]) {
      if (!accountId) continue;

      const payload = {
        post: {
          accountId,
          content: { text: caption, mediaUrls: [publicUrl], platform },
          target: { targetType: platform },
        },
        useNextFreeSlot: true,
      };

      try {
        const resp = await axios.post(`${BLOTATO_API}/posts`, payload, { headers, timeout: 30000 });
        console.log(`Posted to ${platform}:`, resp.data.postSubmissionId);
      } catch (err: any) {
        console.error(`Failed to post to ${platform}:`, err?.response?.data || err.message);
      }
    }
  } catch (err: any) {
    console.error('Upload failed:', err?.response?.data || err.message);
  }
}

async function main() {
  console.log(`[${new Date().toISOString()}] Checking for new underdog wins...`);

  const rendered = loadRendered();
  let newRenders = 0;

  const wins = await fetchUnderdogWins();

  for (const win of wins) {
    const id = `mlb_${win.team}_${win.date}`;
    if (rendered.includes(id)) continue;

    const teamName = TEAM_NAMES[win.team] || win.team;
    const teamColor = TEAM_COLORS[win.team] || '#059669';
    const outputPath = resolve(RENDERED_DIR, `${id}.mp4`);

    const profitUnits = win.profit / 100;

    const props = {
      winnerTeam: win.team,
      winnerName: teamName,
      oddsAmerican: win.odds_american,
      profitUnits,
      scoreText: win.score ? `Final: ${win.score}` : '',
      sport: 'mlb',
      teamColor,
    };

    try {
      renderVideo(props, outputPath);

      const caption = [
        `${teamName} hit at +${win.odds_american}.`,
        '',
        `Model called the ${teamName} ML.`,
        '',
        `+${profitUnits.toFixed(2)}u on a unit bet.`,
        '',
        '#MLB #SportsBetting #Underdogs',
      ].join('\n');

      await uploadToBlotato(outputPath, caption);

      rendered.push(id);
      saveRendered(rendered);
      newRenders++;
      console.log(`Done: ${id}`);
    } catch (err) {
      console.error(`Failed to render ${id}:`, err);
    }
  }

  console.log(`Done. ${newRenders} new video(s) rendered.`);
}

main().catch(console.error);
