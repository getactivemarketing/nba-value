/** MLB and NBA team colours and display names, shared by the video scripts. */
export const TEAM_COLORS: Record<string, string> = {
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

export const TEAM_NAMES: Record<string, string> = {
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

export const teamColor = (abbr: string): string => TEAM_COLORS[abbr] || '#059669';
export const teamName = (abbr: string): string => TEAM_NAMES[abbr] || abbr;
