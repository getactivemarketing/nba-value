# "MODEL HIT" Celebration Video Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Auto-generate 15-second vertical celebration videos when underdog ML picks cash, rendered locally via Remotion and uploaded to Blotato for TikTok/Instagram.

**Architecture:** New `truline-videos/` directory at repo root with a Remotion project. One composition (`ModelHit`) with animated team logo, odds, profit counter, and CTA. A render script checks the Railway API for new wins, renders videos, and uploads to Blotato.

**Tech Stack:** Remotion 4.x, React, TypeScript, axios, node-cron (local)

**Spec:** `docs/superpowers/specs/2026-04-17-model-hit-video.md`

---

### Task 1: Scaffold the Remotion project

**Files:**
- Create: `truline-videos/package.json`
- Create: `truline-videos/remotion.config.ts`
- Create: `truline-videos/tsconfig.json`
- Create: `truline-videos/src/index.ts`
- Create: `truline-videos/src/constants.ts`
- Create: `truline-videos/.gitignore`

- [ ] **Step 1: Create the directory and initialize**

```bash
mkdir -p /Applications/XAMPP/xamppfiles/htdocs/Sites/Truline/truline-videos/src
mkdir -p /Applications/XAMPP/xamppfiles/htdocs/Sites/Truline/truline-videos/rendered
mkdir -p /Applications/XAMPP/xamppfiles/htdocs/Sites/Truline/truline-videos/scripts
```

- [ ] **Step 2: Create package.json**

Create `truline-videos/package.json`:

```json
{
  "name": "truline-videos",
  "version": "1.0.0",
  "private": true,
  "scripts": {
    "studio": "remotion studio",
    "render:test": "remotion render src/index.ts model-hit rendered/test.mp4 --props='{\"winnerTeam\":\"GSW\",\"winnerName\":\"Warriors\",\"oddsAmerican\":180,\"profitUnits\":1.80,\"scoreText\":\"GSW 118, LAC 105\",\"sport\":\"nba\",\"teamColor\":\"#1D428A\"}'",
    "celebrate": "tsx scripts/render-celebrations.ts"
  },
  "dependencies": {
    "@remotion/bundler": "^4.0.447",
    "@remotion/cli": "^4.0.447",
    "@remotion/google-fonts": "^4.0.447",
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "remotion": "^4.0.447",
    "axios": "^1.7.0",
    "tsx": "^4.7.0"
  },
  "devDependencies": {
    "@types/react": "^18.3.0",
    "typescript": "^5.7.0"
  }
}
```

- [ ] **Step 3: Create remotion.config.ts**

Create `truline-videos/remotion.config.ts`:

```typescript
import { Config } from '@remotion/cli/config';

Config.setVideoImageFormat('jpeg');
Config.setConcurrency(1);
Config.setPixelFormat('yuv420p');
Config.setCodec('h264');
Config.setCrf(18);
```

- [ ] **Step 4: Create tsconfig.json**

Create `truline-videos/tsconfig.json`:

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "jsx": "react-jsx",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "allowSyntheticDefaultImports": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "lib": ["ES2022", "DOM"]
  },
  "include": ["src/**/*", "remotion.config.ts", "scripts/**/*"]
}
```

- [ ] **Step 5: Create constants.ts**

Create `truline-videos/src/constants.ts`:

```typescript
export const FPS = 30;
export const WIDTH = 1080;
export const HEIGHT = 1920;

export const COLORS = {
  bg: '#0A0E17',
  surface: '#151B2B',
  text: '#F1F5F9',
  muted: '#64748B',
  green: '#059669',
  greenGlow: 'rgba(5, 150, 105, 0.3)',
  accent: '#A4E6FF',
  white: '#FFFFFF',
};

export const FONTS = {
  display: '"Inter Tight", -apple-system, BlinkMacSystemFont, "Helvetica Neue", sans-serif',
  body: '"Inter", -apple-system, BlinkMacSystemFont, "Helvetica Neue", sans-serif',
  mono: '"JetBrains Mono", "SF Mono", "Fira Code", monospace',
};

export const seconds = (s: number): number => Math.round(s * FPS);

// ESPN logo CDN
export const espnLogoUrl = (abbr: string, sport: string = 'mlb') => {
  const espnMap: Record<string, string> = {
    // MLB
    ARI: 'ari', ATL: 'atl', BAL: 'bal', BOS: 'bos', CHC: 'chc', CWS: 'chw',
    CIN: 'cin', CLE: 'cle', COL: 'col', DET: 'det', HOU: 'hou', KC: 'kc',
    LAA: 'laa', LAD: 'lad', MIA: 'mia', MIL: 'mil', MIN: 'min', NYM: 'nym',
    NYY: 'nyy', OAK: 'oak', PHI: 'phi', PIT: 'pit', SD: 'sd', SF: 'sf',
    SEA: 'sea', STL: 'stl', TB: 'tb', TEX: 'tex', TOR: 'tor', WSH: 'wsh',
    // NBA
    GSW: 'gs', NOP: 'no', NYK: 'ny', SAS: 'sa', UTA: 'utah', WAS: 'wsh',
  };
  const mapped = espnMap[abbr] || abbr.toLowerCase();
  return `https://a.espncdn.com/i/teamlogos/${sport}/500/${mapped}.png`;
};
```

- [ ] **Step 6: Create index.ts and .gitignore**

Create `truline-videos/src/index.ts`:

```typescript
import { registerRoot } from 'remotion';
import { Root } from './Root';

registerRoot(Root);
```

Create `truline-videos/.gitignore`:

```
node_modules/
rendered/*.mp4
```

- [ ] **Step 7: Install dependencies**

```bash
cd /Applications/XAMPP/xamppfiles/htdocs/Sites/Truline/truline-videos
npm install
```

- [ ] **Step 8: Commit**

```bash
cd /Applications/XAMPP/xamppfiles/htdocs/Sites/Truline
git add truline-videos/package.json truline-videos/remotion.config.ts truline-videos/tsconfig.json truline-videos/src/index.ts truline-videos/src/constants.ts truline-videos/.gitignore
git commit -m "feat: scaffold truline-videos Remotion project"
```

---

### Task 2: Create the ModelHit composition and components

**Files:**
- Create: `truline-videos/src/Root.tsx`
- Create: `truline-videos/src/compositions/ModelHit.tsx`

- [ ] **Step 1: Create Root.tsx**

Create `truline-videos/src/Root.tsx`:

```tsx
import React from 'react';
import { Composition } from 'remotion';
import { ModelHit, type ModelHitProps } from './compositions/ModelHit';
import { FPS, WIDTH, HEIGHT, seconds } from './constants';

export const Root: React.FC = () => {
  const defaultProps: ModelHitProps = {
    winnerTeam: 'GSW',
    winnerName: 'Warriors',
    oddsAmerican: 180,
    profitUnits: 1.80,
    scoreText: 'GSW 118, LAC 105',
    sport: 'nba',
    teamColor: '#1D428A',
  };

  return (
    <Composition
      id="model-hit"
      component={ModelHit}
      durationInFrames={seconds(15)}
      fps={FPS}
      width={WIDTH}
      height={HEIGHT}
      defaultProps={defaultProps}
    />
  );
};
```

- [ ] **Step 2: Create ModelHit.tsx**

Create `truline-videos/src/compositions/ModelHit.tsx`:

```tsx
import React from 'react';
import {
  AbsoluteFill,
  Sequence,
  Img,
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig,
} from 'remotion';
import { COLORS, FONTS, seconds, espnLogoUrl } from '../constants';

export interface ModelHitProps {
  winnerTeam: string;
  winnerName: string;
  oddsAmerican: number;
  profitUnits: number;
  scoreText: string;
  sport: 'mlb' | 'nba';
  teamColor: string;
}

const AnimatedText: React.FC<{
  children: string;
  delay?: number;
  size?: number;
  color?: string;
  font?: string;
  weight?: number;
  y?: number;
}> = ({ children, delay = 0, size = 80, color = COLORS.text, font = FONTS.display, weight = 800, y }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const local = frame - delay;
  if (local < 0) return null;

  const progress = spring({
    frame: local,
    fps,
    config: { damping: 14, stiffness: 100, mass: 0.8 },
  });

  const scale = interpolate(progress, [0, 1], [0.6, 1]);
  const opacity = interpolate(progress, [0, 1], [0, 1]);

  return (
    <div
      style={{
        position: y !== undefined ? 'absolute' : 'relative',
        top: y,
        left: 0,
        right: 0,
        textAlign: 'center',
        fontSize: size,
        fontFamily: font,
        fontWeight: weight,
        color,
        opacity,
        transform: `scale(${scale})`,
        letterSpacing: '-0.02em',
      }}
    >
      {children}
    </div>
  );
};

const ProfitCounter: React.FC<{
  target: number;
  delay?: number;
}> = ({ target, delay = 0 }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const local = frame - delay;
  if (local < 0) return null;

  const progress = spring({
    frame: local,
    fps,
    config: { damping: 20, stiffness: 80, mass: 1 },
  });

  const value = interpolate(progress, [0, 1], [0, target]);
  const opacity = interpolate(progress, [0, 1], [0, 1]);

  return (
    <div
      style={{
        textAlign: 'center',
        fontSize: 72,
        fontFamily: FONTS.mono,
        fontWeight: 700,
        color: COLORS.green,
        opacity,
      }}
    >
      +{value.toFixed(2)}u
    </div>
  );
};

export const ModelHit: React.FC<ModelHitProps> = ({
  winnerTeam,
  winnerName,
  oddsAmerican,
  profitUnits,
  scoreText,
  sport,
  teamColor,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Background glow animation
  const glowPulse = interpolate(
    frame % 60,
    [0, 30, 60],
    [0.15, 0.25, 0.15],
  );

  const logoUrl = espnLogoUrl(winnerTeam, sport);
  const oddsStr = `+${oddsAmerican}`;

  return (
    <AbsoluteFill style={{ backgroundColor: COLORS.bg }}>
      {/* Team color radial glow (pulsing) */}
      <div
        style={{
          position: 'absolute',
          top: '30%',
          left: '50%',
          width: 800,
          height: 800,
          transform: 'translate(-50%, -50%)',
          borderRadius: '50%',
          background: `radial-gradient(circle, ${teamColor} 0%, transparent 70%)`,
          opacity: glowPulse,
          filter: 'blur(80px)',
        }}
      />

      {/* Phase 1: 0-3s — "MODEL HIT" title */}
      <Sequence from={0} durationInFrames={seconds(15)}>
        <AnimatedText
          size={140}
          color={COLORS.green}
          weight={900}
          y={280}
          delay={5}
        >
          MODEL HIT
        </AnimatedText>
        <AnimatedText
          size={56}
          color={COLORS.muted}
          weight={500}
          y={440}
          delay={15}
        >
          Underdog cashed
        </AnimatedText>
      </Sequence>

      {/* Phase 2: 3-7s — Team logo + name */}
      <Sequence from={seconds(3)} durationInFrames={seconds(12)}>
        {(() => {
          const localFrame = frame - seconds(3);
          if (localFrame < 0) return null;

          const logoProgress = spring({
            frame: localFrame,
            fps,
            config: { damping: 12, stiffness: 80, mass: 1 },
          });

          const logoScale = interpolate(logoProgress, [0, 1], [0.3, 1]);
          const logoOpacity = interpolate(logoProgress, [0, 1], [0, 1]);

          return (
            <>
              <div
                style={{
                  position: 'absolute',
                  top: 580,
                  left: '50%',
                  transform: `translateX(-50%) scale(${logoScale})`,
                  opacity: logoOpacity,
                  filter: `drop-shadow(0 0 60px ${teamColor})`,
                }}
              >
                <Img src={logoUrl} width={380} height={380} />
              </div>
              <AnimatedText
                size={96}
                color={COLORS.text}
                weight={800}
                y={1000}
                delay={seconds(3) + 15}
              >
                {winnerName.toUpperCase()}
              </AnimatedText>
            </>
          );
        })()}
      </Sequence>

      {/* Phase 3: 7-11s — Odds + profit */}
      <Sequence from={seconds(7)} durationInFrames={seconds(8)}>
        <AnimatedText
          size={160}
          color={COLORS.accent}
          font={FONTS.mono}
          weight={900}
          y={1160}
          delay={seconds(7)}
        >
          {oddsStr}
        </AnimatedText>
        <AnimatedText
          size={48}
          color={COLORS.muted}
          weight={600}
          y={1340}
          delay={seconds(7) + 10}
        >
          UNDERDOG CASHED
        </AnimatedText>
        <div style={{ position: 'absolute', top: 1420, left: 0, right: 0 }}>
          <ProfitCounter target={profitUnits} delay={seconds(7) + 20} />
        </div>
      </Sequence>

      {/* Phase 4: 11-15s — CTA */}
      <Sequence from={seconds(11)} durationInFrames={seconds(4)}>
        <AnimatedText
          size={42}
          color={COLORS.muted}
          weight={500}
          y={1560}
          delay={seconds(11)}
        >
          {scoreText}
        </AnimatedText>
        <AnimatedText
          size={56}
          color={COLORS.accent}
          weight={700}
          y={1680}
          delay={seconds(11) + 10}
        >
          truline.app
        </AnimatedText>
        <AnimatedText
          size={40}
          color={COLORS.muted}
          weight={500}
          y={1760}
          delay={seconds(11) + 15}
        >
          Follow for daily picks
        </AnimatedText>
      </Sequence>
    </AbsoluteFill>
  );
};
```

- [ ] **Step 3: Preview in Remotion Studio**

```bash
cd /Applications/XAMPP/xamppfiles/htdocs/Sites/Truline/truline-videos
npx remotion studio
```

Opens browser at `http://localhost:3000`. Verify the "model-hit" composition renders with default props (Warriors +180). Check all 4 phases animate correctly.

- [ ] **Step 4: Render a test video**

```bash
npm run render:test
```

Verify `rendered/test.mp4` exists, is ~15 seconds, plays at 1080x1920.

- [ ] **Step 5: Commit**

```bash
cd /Applications/XAMPP/xamppfiles/htdocs/Sites/Truline
git add truline-videos/src/Root.tsx truline-videos/src/compositions/ModelHit.tsx
git commit -m "feat: ModelHit celebration video composition — 15-sec vertical with team logo, odds, profit counter"
```

---

### Task 3: Create the render-celebrations script

**Files:**
- Create: `truline-videos/scripts/render-celebrations.ts`
- Create: `truline-videos/rendered.json`

- [ ] **Step 1: Create rendered.json (empty tracker)**

Create `truline-videos/rendered.json`:

```json
[]
```

- [ ] **Step 2: Create render-celebrations.ts**

Create `truline-videos/scripts/render-celebrations.ts`:

```typescript
/**
 * Check for new underdog ML wins and render celebration videos.
 *
 * Run manually: npx tsx scripts/render-celebrations.ts
 * Run via cron:  */30 * * * * cd /path/to/truline-videos && npx tsx scripts/render-celebrations.ts
 */

import axios from 'axios';
import { execSync } from 'child_process';
import { readFileSync, writeFileSync, existsSync } from 'fs';
import { resolve } from 'path';

const API_BASE = 'https://nba-value-production.up.railway.app/api/v1';
const BLOTATO_API = 'https://backend.blotato.com/v2';
const BLOTATO_API_KEY = process.env.BLOTATO_API_KEY || '';
const BLOTATO_TIKTOK_ACCOUNT_ID = process.env.BLOTATO_TIKTOK_ACCOUNT_ID || '';
const BLOTATO_INSTAGRAM_ACCOUNT_ID = process.env.BLOTATO_INSTAGRAM_ACCOUNT_ID || '';

const RENDERED_FILE = resolve(__dirname, '..', 'rendered.json');
const RENDERED_DIR = resolve(__dirname, '..', 'rendered');

// Team colors (primary)
const TEAM_COLORS: Record<string, string> = {
  // MLB
  ARI: '#A71930', ATL: '#CE1141', BAL: '#DF4601', BOS: '#BD3039',
  CHC: '#0E3386', CWS: '#27251F', CIN: '#C6011F', CLE: '#00385D',
  COL: '#33006F', DET: '#0C2340', HOU: '#002D62', KC: '#004687',
  LAA: '#BA0021', LAD: '#005A9C', MIA: '#00A3E0', MIL: '#12284B',
  MIN: '#002B5C', NYM: '#002D72', NYY: '#003087', OAK: '#003831',
  PHI: '#E81828', PIT: '#FDB827', SD: '#2F241D', SF: '#FD5A1E',
  SEA: '#005C5C', STL: '#C41E3A', TB: '#092C5C', TEX: '#003278',
  TOR: '#134A8E', WSH: '#AB0003',
  // NBA
  ATL: '#E14434', BOS: '#007A33', BKN: '#000000', CHA: '#1D1160',
  CHI: '#CE1141', CLE: '#860038', DAL: '#0053BC', DEN: '#0D2240',
  DET: '#C8102E', GSW: '#1D428A', HOU: '#CE1141', IND: '#002D62',
  LAC: '#C8102E', LAL: '#552583', MEM: '#5D76A9', MIA: '#98002E',
  MIL: '#00471B', NOP: '#002B5C', NYK: '#006BB6', OKC: '#007DC3',
  ORL: '#0077C0', PHX: '#1D1160', POR: '#E03A3E', SAC: '#5B2B82',
  SAS: '#C4CED4', TOR: '#CE1141', UTA: '#002B5C', WAS: '#002B5C',
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
  // NBA
  GSW: 'Warriors', LAL: 'Lakers', BKN: 'Nets', NYK: 'Knicks',
  MIA: 'Heat', MIL: 'Bucks', DEN: 'Nuggets', PHX: 'Suns',
  DAL: 'Mavericks', MEM: 'Grizzlies', SAC: 'Kings', OKC: 'Thunder',
  CLE: 'Cavaliers', IND: 'Pacers', ORL: 'Magic', CHA: 'Hornets',
  DET: 'Pistons', CHI: 'Bulls', TOR: 'Raptors', POR: 'Trail Blazers',
  SAS: 'Spurs', NOP: 'Pelicans', UTA: 'Jazz', WAS: 'Wizards',
  LAC: 'Clippers', BOS: 'Celtics',
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

async function fetchUnderdogWins(sport: 'mlb' | 'nba'): Promise<UnderdogWin[]> {
  const endpoint = sport === 'mlb'
    ? `${API_BASE}/mlb/evaluation/underdogs?days=2`
    : `${API_BASE}/evaluation/predictions?days=2`;

  try {
    const resp = await axios.get(endpoint, { timeout: 15000 });
    if (sport === 'mlb') {
      return resp.data.biggest_wins || [];
    }
    return [];
  } catch (err) {
    console.error(`Failed to fetch ${sport} underdogs:`, err);
    return [];
  }
}

function renderVideo(props: {
  winnerTeam: string;
  winnerName: string;
  oddsAmerican: number;
  profitUnits: number;
  scoreText: string;
  sport: string;
  teamColor: string;
}, outputPath: string) {
  const propsJson = JSON.stringify(props).replace(/'/g, "'\\''");
  const cmd = `npx remotion render src/index.ts model-hit "${outputPath}" --props='${propsJson}'`;
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

  // Upload media
  const videoData = readFileSync(videoPath);
  const filename = videoPath.split('/').pop() || 'video.mp4';

  const uploadResp = await axios.post(`${BLOTATO_API}/media/uploads`, { filename }, { headers, timeout: 60000 });
  const { presignedUrl, publicUrl } = uploadResp.data;

  await axios.put(presignedUrl, videoData, {
    headers: { 'Content-Type': 'video/mp4' },
    timeout: 60000,
  });

  console.log(`Uploaded: ${publicUrl}`);

  // Post to each platform
  for (const [platform, accountId] of [
    ['tiktok', BLOTATO_TIKTOK_ACCOUNT_ID],
    ['instagram', BLOTATO_INSTAGRAM_ACCOUNT_ID],
  ]) {
    if (!accountId) continue;

    const payload = {
      post: {
        accountId,
        content: {
          text: caption,
          mediaUrls: [publicUrl],
          platform,
        },
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
}

async function main() {
  console.log(`[${new Date().toISOString()}] Checking for new underdog wins...`);

  const rendered = loadRendered();
  let newRenders = 0;

  // Check MLB
  const mlbWins = await fetchUnderdogWins('mlb');
  for (const win of mlbWins) {
    const id = `mlb_${win.team}_${win.date}`;
    if (rendered.includes(id)) continue;

    const teamName = TEAM_NAMES[win.team] || win.team;
    const teamColor = TEAM_COLORS[win.team] || '#059669';
    const outputPath = resolve(RENDERED_DIR, `${id}.mp4`);

    const props = {
      winnerTeam: win.team,
      winnerName: teamName,
      oddsAmerican: win.odds_american,
      profitUnits: win.profit / 100, // API returns profit on $100 bet
      scoreText: win.score ? `Final: ${win.score}` : '',
      sport: 'mlb',
      teamColor,
    };

    try {
      renderVideo(props, outputPath);

      const caption = `${teamName} hit at +${win.odds_american}.\n\nModel called the ${teamName} ML.\n\n+${props.profitUnits.toFixed(2)}u on a unit bet.\n\n#MLB #SportsBetting #Underdogs`;
      await uploadToBlotato(outputPath, caption);

      rendered.push(id);
      saveRendered(rendered);
      newRenders++;
      console.log(`Rendered + uploaded: ${id}`);
    } catch (err) {
      console.error(`Failed to render ${id}:`, err);
    }
  }

  console.log(`Done. ${newRenders} new video(s) rendered.`);
}

main().catch(console.error);
```

- [ ] **Step 3: Test dry-run (no Blotato key)**

```bash
cd /Applications/XAMPP/xamppfiles/htdocs/Sites/Truline/truline-videos
npx tsx scripts/render-celebrations.ts
```

Expected: script checks API, finds wins (if any), renders video, prints `[DRY-RUN]` for upload since no `BLOTATO_API_KEY` is set.

- [ ] **Step 4: Commit**

```bash
cd /Applications/XAMPP/xamppfiles/htdocs/Sites/Truline
git add truline-videos/scripts/render-celebrations.ts truline-videos/rendered.json
git commit -m "feat: render-celebrations script — checks API, renders videos, uploads to Blotato"
```

---

### Task 4: Set up local cron and env

- [ ] **Step 1: Create a .env file for Blotato credentials**

Create `truline-videos/.env` (DO NOT commit):

```bash
BLOTATO_API_KEY=your_blotato_api_key_here
BLOTATO_TIKTOK_ACCOUNT_ID=your_tiktok_account_id
BLOTATO_INSTAGRAM_ACCOUNT_ID=your_instagram_account_id
```

Add to `truline-videos/.gitignore`:

```
.env
```

- [ ] **Step 2: Update the script to load .env**

Add to the top of `scripts/render-celebrations.ts` (after the import block):

```typescript
import { config } from 'dotenv';
config({ path: resolve(__dirname, '..', '.env') });
```

Add `dotenv` to package.json dependencies:

```bash
cd /Applications/XAMPP/xamppfiles/htdocs/Sites/Truline/truline-videos
npm install dotenv
```

- [ ] **Step 3: Set up cron**

Add to your Mac crontab:

```bash
crontab -e
```

Add this line:

```
*/30 * * * * cd /Applications/XAMPP/xamppfiles/htdocs/Sites/Truline/truline-videos && /usr/local/bin/npx tsx scripts/render-celebrations.ts >> /tmp/truline-video.log 2>&1
```

Note: adjust the `npx` path if needed — run `which npx` to find the correct path.

- [ ] **Step 4: Verify cron runs**

Wait 30 minutes, then check:

```bash
cat /tmp/truline-video.log
```

Should show the script ran and checked for wins.

- [ ] **Step 5: Commit .gitignore update**

```bash
cd /Applications/XAMPP/xamppfiles/htdocs/Sites/Truline
git add truline-videos/.gitignore truline-videos/scripts/render-celebrations.ts truline-videos/package.json
git commit -m "feat: add dotenv support and cron setup for celebration video pipeline"
```
