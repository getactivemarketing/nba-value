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
import { TEAM_COLORS, TEAM_NAMES } from '../src/teams';
import { blotatoConfigFromEnv, uploadToBlotato } from '../src/blotato';

config({ path: resolve(__dirname, '..', '.env') });

const API_BASE = 'https://nba-value-production.up.railway.app/api/v1';

const RENDERED_FILE = resolve(__dirname, '..', 'rendered.json');
const RENDERED_DIR = resolve(__dirname, '..', 'rendered');

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

      await uploadToBlotato(outputPath, caption, blotatoConfigFromEnv());

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
