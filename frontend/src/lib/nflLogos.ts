/** NFL team logos via ESPN CDN: https://a.espncdn.com/i/teamlogos/nfl/500/{abbr}.png */
const NFL_ESPN_ABBR: Record<string, string> = {
  ARI:'ari',ATL:'atl',BAL:'bal',BUF:'buf',CAR:'car',CHI:'chi',CIN:'cin',CLE:'cle',DAL:'dal',DEN:'den',
  DET:'det',GB:'gb',HOU:'hou',IND:'ind',JAX:'jax',KC:'kc',LA:'lar',LAC:'lac',LV:'lv',MIA:'mia',
  MIN:'min',NE:'ne',NO:'no',NYG:'nyg',NYJ:'nyj',PHI:'phi',PIT:'pit',SEA:'sea',SF:'sf',TB:'tb',TEN:'ten',WAS:'wsh',
};
export function getTeamLogo(abbr: string): string {
  const e = NFL_ESPN_ABBR[abbr] || abbr.toLowerCase();
  return `https://a.espncdn.com/i/teamlogos/nfl/500/${e}.png`;
}
