/**
 * MLB Team logos
 * Uses ESPN's CDN for official team logos.
 *
 * URL format: https://a.espncdn.com/i/teamlogos/mlb/500/{abbr}.png
 * ESPN uses lowercase abbreviations, mostly matching ours but with a few differences.
 */

// Map our team abbreviations to ESPN's
const MLB_ESPN_ABBR: Record<string, string> = {
  ARI: 'ari',
  ATL: 'atl',
  BAL: 'bal',
  BOS: 'bos',
  CHC: 'chc',
  CWS: 'chw', // ESPN uses CHW
  CIN: 'cin',
  CLE: 'cle',
  COL: 'col',
  DET: 'det',
  HOU: 'hou',
  KC: 'kc',
  LAA: 'laa',
  LAD: 'lad',
  MIA: 'mia',
  MIL: 'mil',
  MIN: 'min',
  NYM: 'nym',
  NYY: 'nyy',
  OAK: 'oak',
  PHI: 'phi',
  PIT: 'pit',
  SD: 'sd',
  SF: 'sf',
  SEA: 'sea',
  STL: 'stl',
  TB: 'tb',
  TEX: 'tex',
  TOR: 'tor',
  WSH: 'wsh',
};

export function getMLBLogo(abbreviation: string): string {
  const abbr = abbreviation.toUpperCase();
  const espnAbbr = MLB_ESPN_ABBR[abbr];
  if (!espnAbbr) return '';
  return `https://a.espncdn.com/i/teamlogos/mlb/500/${espnAbbr}.png`;
}
