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

export const espnLogoUrl = (abbr: string, sport: string = 'mlb') => {
  const espnMap: Record<string, string> = {
    ARI: 'ari', ATL: 'atl', BAL: 'bal', BOS: 'bos', CHC: 'chc', CWS: 'chw',
    CIN: 'cin', CLE: 'cle', COL: 'col', DET: 'det', HOU: 'hou', KC: 'kc',
    LAA: 'laa', LAD: 'lad', MIA: 'mia', MIL: 'mil', MIN: 'min', NYM: 'nym',
    NYY: 'nyy', OAK: 'oak', PHI: 'phi', PIT: 'pit', SD: 'sd', SF: 'sf',
    SEA: 'sea', STL: 'stl', TB: 'tb', TEX: 'tex', TOR: 'tor', WSH: 'wsh',
    GSW: 'gs', NOP: 'no', NYK: 'ny', SAS: 'sa', UTA: 'utah', WAS: 'wsh',
  };
  const mapped = espnMap[abbr] || abbr.toLowerCase();
  return `https://a.espncdn.com/i/teamlogos/${sport}/500/${mapped}.png`;
};
