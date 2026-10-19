// frontend/src/lib/edgeFormat.ts
const BOOKS: Record<string, string> = {
  pinnacle: 'Pinnacle', novig: 'Novig', prophetx: 'ProphetX', draftkings: 'DraftKings',
  fanduel: 'FanDuel', betmgm: 'BetMGM', williamhill_us: 'Caesars', espnbet: 'ESPN BET',
  fanatics: 'Fanatics', betrivers: 'BetRivers', hardrockbet: 'Hard Rock', betonlineag: 'BetOnline',
  bovada: 'Bovada', lowvig: 'LowVig', prizepicks: 'PrizePicks', underdog: 'Underdog',
};
const MARKETS: Record<string, string> = {
  player_points: 'Points', player_rebounds: 'Rebounds', player_assists: 'Assists',
  player_threes: 'Threes', player_points_rebounds_assists: 'Pts+Reb+Ast',
  player_points_rebounds: 'Pts+Reb', player_points_assists: 'Pts+Ast',
  player_rebounds_assists: 'Reb+Ast', h2h: 'Moneyline', spreads: 'Spread', totals: 'Total',
  player_pass_yds: 'Pass Yds', player_pass_tds: 'Pass TDs', player_pass_completions: 'Completions',
  player_pass_attempts: 'Pass Att', player_pass_interceptions: 'INTs', player_rush_yds: 'Rush Yds',
  player_rush_attempts: 'Rush Att', player_receptions: 'Receptions', player_reception_yds: 'Rec Yds',
  player_rush_reception_yds: 'Rush+Rec Yds', player_anytime_td: 'Anytime TD',
};
const NBA_MARKETS = ['h2h', 'spreads', 'totals', 'player_points', 'player_rebounds', 'player_assists',
  'player_threes', 'player_points_rebounds_assists', 'player_points_rebounds', 'player_points_assists',
  'player_rebounds_assists'];
const NFL_MARKETS = ['h2h', 'spreads', 'totals', 'player_pass_yds', 'player_pass_tds',
  'player_pass_completions', 'player_pass_attempts', 'player_pass_interceptions', 'player_rush_yds',
  'player_rush_attempts', 'player_receptions', 'player_reception_yds', 'player_rush_reception_yds',
  'player_anytime_td'];
export type Sport = 'nba' | 'nfl';
export const marketKeysFor = (sport: Sport) => (sport === 'nfl' ? NFL_MARKETS : NBA_MARKETS);
export const propMarketsFor = (sport: Sport) => marketKeysFor(sport).filter((m) => m.startsWith('player_'));
export const edgeBase = (sport: Sport) => (sport === 'nfl' ? '/nfl/edge' : '/edge');
export const BOOK_KEYS = Object.keys(BOOKS);
export const MARKET_KEYS = NBA_MARKETS;
export const DFS_BOOKS = new Set(['prizepicks', 'underdog']);

export const fmtAmerican = (n: number | null | undefined) =>
  n == null ? '—' : n > 0 ? `+${n}` : `${n}`;
export const fmtEv = (x: number) => `${x >= 0 ? '+' : ''}${(x * 100).toFixed(1)}%`;
export const fmtPct = (x: number | null | undefined) => (x == null ? '—' : `${Math.round(x * 100)}%`);
export function fmtAge(iso: string, now: Date = new Date()): string {
  const s = Math.max(0, Math.round((now.getTime() - new Date(iso).getTime()) / 1000));
  return s < 60 ? `${s}s` : s < 3600 ? `${Math.floor(s / 60)}m` : `${Math.floor(s / 3600)}h`;
}
export const bookLabel = (k: string) => BOOKS[k] ?? k;
export const marketLabel = (k: string) => MARKETS[k] ?? k;

/** Spread lines are stored from the HOME side; an away row shows the negated line. */
export function outcomeLabel(market: string, side: string, line: number,
  ev: { home_team: string; away_team: string }): string {
  const team = side === 'home' ? ev.home_team : ev.away_team;
  if (market === 'h2h') return team;
  if (market === 'spreads') {
    const l = side === 'home' ? line : -line;
    return `${team} ${l > 0 ? '+' : ''}${l}`;
  }
  if (market === 'player_anytime_td') return side === 'over' ? 'Yes' : 'No'; // stored as over/under 0.5
  return `${side === 'over' ? 'Over' : 'Under'} ${line}`;
}
