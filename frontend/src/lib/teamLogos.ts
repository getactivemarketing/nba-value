/**
 * NBA Team logos and colors
 * Using ESPN's CDN for official team logos
 */

interface TeamInfo {
  name: string;
  city: string;
  fullName: string;
  logo: string;
  primaryColor: string;
  secondaryColor: string;
}

// ESPN team IDs mapped to abbreviations
const ESPN_TEAM_IDS: Record<string, string> = {
  ATL: '1',
  BOS: '2',
  BKN: '17',
  CHA: '30',
  CHI: '4',
  CLE: '5',
  DAL: '6',
  DEN: '7',
  DET: '8',
  GSW: '9',
  HOU: '10',
  IND: '11',
  LAC: '12',
  LAL: '13',
  MEM: '29',
  MIA: '14',
  MIL: '15',
  MIN: '16',
  NOP: '3',
  NYK: '18',
  OKC: '25',
  ORL: '19',
  PHI: '20',
  PHX: '21',
  POR: '22',
  SAC: '23',
  SAS: '24',
  TOR: '28',
  UTA: '26',
  WAS: '27',
};

const TEAM_INFO: Record<string, TeamInfo> = {
  ATL: { name: 'Hawks', city: 'Atlanta', fullName: 'Atlanta Hawks', logo: '', primaryColor: '#E03A3E', secondaryColor: '#C1D32F' },
  BOS: { name: 'Celtics', city: 'Boston', fullName: 'Boston Celtics', logo: '', primaryColor: '#007A33', secondaryColor: '#BA9653' },
  BKN: { name: 'Nets', city: 'Brooklyn', fullName: 'Brooklyn Nets', logo: '', primaryColor: '#000000', secondaryColor: '#FFFFFF' },
  CHA: { name: 'Hornets', city: 'Charlotte', fullName: 'Charlotte Hornets', logo: '', primaryColor: '#1D1160', secondaryColor: '#00788C' },
  CHI: { name: 'Bulls', city: 'Chicago', fullName: 'Chicago Bulls', logo: '', primaryColor: '#CE1141', secondaryColor: '#000000' },
  CLE: { name: 'Cavaliers', city: 'Cleveland', fullName: 'Cleveland Cavaliers', logo: '', primaryColor: '#860038', secondaryColor: '#FDBB30' },
  DAL: { name: 'Mavericks', city: 'Dallas', fullName: 'Dallas Mavericks', logo: '', primaryColor: '#00538C', secondaryColor: '#002B5E' },
  DEN: { name: 'Nuggets', city: 'Denver', fullName: 'Denver Nuggets', logo: '', primaryColor: '#0E2240', secondaryColor: '#FEC524' },
  DET: { name: 'Pistons', city: 'Detroit', fullName: 'Detroit Pistons', logo: '', primaryColor: '#C8102E', secondaryColor: '#1D42BA' },
  GSW: { name: 'Warriors', city: 'Golden State', fullName: 'Golden State Warriors', logo: '', primaryColor: '#1D428A', secondaryColor: '#FFC72C' },
  HOU: { name: 'Rockets', city: 'Houston', fullName: 'Houston Rockets', logo: '', primaryColor: '#CE1141', secondaryColor: '#000000' },
  IND: { name: 'Pacers', city: 'Indiana', fullName: 'Indiana Pacers', logo: '', primaryColor: '#002D62', secondaryColor: '#FDBB30' },
  LAC: { name: 'Clippers', city: 'Los Angeles', fullName: 'Los Angeles Clippers', logo: '', primaryColor: '#C8102E', secondaryColor: '#1D428A' },
  LAL: { name: 'Lakers', city: 'Los Angeles', fullName: 'Los Angeles Lakers', logo: '', primaryColor: '#552583', secondaryColor: '#FDB927' },
  MEM: { name: 'Grizzlies', city: 'Memphis', fullName: 'Memphis Grizzlies', logo: '', primaryColor: '#5D76A9', secondaryColor: '#12173F' },
  MIA: { name: 'Heat', city: 'Miami', fullName: 'Miami Heat', logo: '', primaryColor: '#98002E', secondaryColor: '#F9A01B' },
  MIL: { name: 'Bucks', city: 'Milwaukee', fullName: 'Milwaukee Bucks', logo: '', primaryColor: '#00471B', secondaryColor: '#EEE1C6' },
  MIN: { name: 'Timberwolves', city: 'Minnesota', fullName: 'Minnesota Timberwolves', logo: '', primaryColor: '#0C2340', secondaryColor: '#236192' },
  NOP: { name: 'Pelicans', city: 'New Orleans', fullName: 'New Orleans Pelicans', logo: '', primaryColor: '#0C2340', secondaryColor: '#C8102E' },
  NYK: { name: 'Knicks', city: 'New York', fullName: 'New York Knicks', logo: '', primaryColor: '#006BB6', secondaryColor: '#F58426' },
  OKC: { name: 'Thunder', city: 'Oklahoma City', fullName: 'Oklahoma City Thunder', logo: '', primaryColor: '#007AC1', secondaryColor: '#EF3B24' },
  ORL: { name: 'Magic', city: 'Orlando', fullName: 'Orlando Magic', logo: '', primaryColor: '#0077C0', secondaryColor: '#C4CED4' },
  PHI: { name: '76ers', city: 'Philadelphia', fullName: 'Philadelphia 76ers', logo: '', primaryColor: '#006BB6', secondaryColor: '#ED174C' },
  PHX: { name: 'Suns', city: 'Phoenix', fullName: 'Phoenix Suns', logo: '', primaryColor: '#1D1160', secondaryColor: '#E56020' },
  POR: { name: 'Trail Blazers', city: 'Portland', fullName: 'Portland Trail Blazers', logo: '', primaryColor: '#E03A3E', secondaryColor: '#000000' },
  SAC: { name: 'Kings', city: 'Sacramento', fullName: 'Sacramento Kings', logo: '', primaryColor: '#5A2D81', secondaryColor: '#63727A' },
  SAS: { name: 'Spurs', city: 'San Antonio', fullName: 'San Antonio Spurs', logo: '', primaryColor: '#C4CED4', secondaryColor: '#000000' },
  TOR: { name: 'Raptors', city: 'Toronto', fullName: 'Toronto Raptors', logo: '', primaryColor: '#CE1141', secondaryColor: '#000000' },
  UTA: { name: 'Jazz', city: 'Utah', fullName: 'Utah Jazz', logo: '', primaryColor: '#002B5C', secondaryColor: '#00471B' },
  WAS: { name: 'Wizards', city: 'Washington', fullName: 'Washington Wizards', logo: '', primaryColor: '#002B5C', secondaryColor: '#E31837' },
};

// Generate logo URLs
Object.keys(ESPN_TEAM_IDS).forEach((abbr) => {
  if (TEAM_INFO[abbr]) {
    TEAM_INFO[abbr].logo = `https://a.espncdn.com/i/teamlogos/nba/500/${abbr.toLowerCase()}.png`;
  }
});

export function getTeamLogo(abbreviation: string): string {
  const abbr = abbreviation.toUpperCase();
  return TEAM_INFO[abbr]?.logo || '';
}

export function getTeamInfo(abbreviation: string): TeamInfo | null {
  const abbr = abbreviation.toUpperCase();
  return TEAM_INFO[abbr] || null;
}

export function getTeamColor(abbreviation: string): string {
  const abbr = abbreviation.toUpperCase();
  return TEAM_INFO[abbr]?.primaryColor || '#6B7280';
}

export function getTeamFullName(abbreviation: string): string {
  const abbr = abbreviation.toUpperCase();
  return TEAM_INFO[abbr]?.fullName || abbreviation;
}

export { TEAM_INFO };
