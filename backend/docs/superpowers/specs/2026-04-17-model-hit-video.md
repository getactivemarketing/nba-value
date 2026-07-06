# "MODEL HIT" Celebration Video — Design Spec

## Goal

Auto-generate a 15-second vertical video (1080x1920) when an underdog ML pick at +130 or better cashes. Rendered locally via Remotion, uploaded to Blotato for TikTok + Instagram Reels.

## Video Timeline (15 seconds, 30fps = 450 frames)

- **0-3s (frames 0-89)**: Dark background with team primary color radial glow. "MODEL HIT" text scales in from 0.5x to 1x with spring easing. Subtle pulse on the glow.
- **3-7s (frames 90-209)**: Winning team logo fades in center at ~400px. Team name types in below letter by letter. Team color gradient wash behind logo.
- **7-11s (frames 210-329)**: Odds badge slides up from bottom (+180 in large text). "UNDERDOG CASHED" label fades in. Profit counter animates from $0 → +1.80u with a counting effect.
- **11-15s (frames 330-449)**: TruLine logo + "truline.app" + "Follow for daily picks" CTA. Score text below. Holds static for screenshots.

## Architecture

```
truline-videos/                    (new directory at repo root)
├── package.json                   (remotion + axios deps)
├── remotion.config.ts
├── tsconfig.json
├── src/
│   ├── Root.tsx                   (registers ModelHit composition)
│   ├── constants.ts               (TruLine brand colors, fonts, FPS=30, 1080x1920)
│   ├── compositions/
│   │   └── ModelHit.tsx           (15-sec celebration video component)
│   └── components/
│       ├── TeamLogo.tsx           (fetches ESPN logo via Img, renders with glow shadow)
│       ├── OddsBadge.tsx          (animated odds display + profit counter)
│       └── TruLineCTA.tsx         (closing frame: logo, URL, follow CTA)
├── scripts/
│   └── render-celebrations.ts     (cron script: check API → render → upload → mark done)
├── rendered/                      (.gitignored output dir for .mp4 files)
└── rendered.json                  (tracks which wins have been rendered — committed)
```

## Constants

```typescript
export const FPS = 30;
export const WIDTH = 1080;
export const HEIGHT = 1920;

export const COLORS = {
  bg: '#0A0E17',
  surface: '#151B2B',
  text: '#F1F5F9',
  muted: '#64748B',
  green: '#05966980',   // profit counter
  greenSolid: '#059669',
  accent: '#A4E6FF',
  white: '#FFFFFF',
};

export const seconds = (s: number) => Math.round(s * FPS);
```

## Team Colors

Reuse the same `MLB_TEAMS` / `NBA_TEAM_COLORS` mappings from the existing frontend/backend. The render script passes the team's primary color as a prop.

## Composition Props

```typescript
interface ModelHitProps {
  winnerTeam: string;      // "GSW"
  winnerName: string;      // "Warriors"
  oddsAmerican: number;    // 180
  profitUnits: number;     // 1.80
  scoreText: string;       // "GSW 118, LAC 105"
  sport: 'mlb' | 'nba';
  teamColor: string;       // "#1D428A"
}
```

## Render Script (`scripts/render-celebrations.ts`)

1. Reads `rendered.json` (array of snapshot IDs already processed)
2. Fetches `GET /api/v1/mlb/evaluation/underdogs?days=2` from Railway
3. Filters for wins not in `rendered.json`
4. For each new win:
   a. Calls `npx remotion render src/index.ts model-hit rendered/{team}_{date}.mp4 --props='{"winnerTeam":"GSW",...}'`
   b. Uploads `.mp4` to Blotato via media upload + post (reuse Blotato API pattern from backend)
   c. Appends snapshot ID to `rendered.json`
5. Also checks NBA underdogs endpoint (same logic)

## Cron Setup

macOS launchd or crontab:
```
*/30 * * * * cd /path/to/truline-videos && npx tsx scripts/render-celebrations.ts >> /tmp/truline-video.log 2>&1
```

Runs every 30 minutes. If no new wins, exits immediately (~1 second). If a win exists, renders (~45 sec) and uploads (~5 sec).

## Blotato Upload

Same pattern as backend `blotato.py`:
1. `POST /v2/media/uploads` → get presigned URL + public URL
2. `PUT` raw mp4 bytes to presigned URL
3. `POST /v2/posts` with `mediaUrls: [publicUrl]` targeting TikTok + Instagram accounts

Needs `BLOTATO_API_KEY` and account IDs for TikTok/Instagram (from Blotato dashboard).

## Files Changed

- Create: `truline-videos/` directory with all files listed above (~8 files)
- No changes to existing backend or frontend code
- `.gitignore`: add `truline-videos/rendered/*.mp4`

## Testing

- `cd truline-videos && npx remotion studio` — preview the video in browser
- `npx remotion render src/index.ts model-hit test.mp4 --props='...'` — render a test video
- Verify the rendered .mp4 plays correctly at 1080x1920 / 15 seconds
- Run `render-celebrations.ts` manually to test the full pipeline
