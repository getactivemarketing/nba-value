import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_URL || '/api/v1';

const client = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add auth token to requests
client.interceptors.request.use((config) => {
  const token = localStorage.getItem('auth_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Types
export interface NFLPick {
  game_id: string; home_team: string; away_team: string; kickoff_utc: string | null;
  best_bet_type: string | null; best_bet_team: string | null; best_total_direction: string | null;
  best_bet_line: number | null;
  best_bet_odds: number | null; best_bet_value_score: number | null; best_bet_edge: number | null;
  predicted_margin: number | null; predicted_total: number | null;
}
export interface NFLPicksResponse { picks: NFLPick[]; total: number; min_value_score: number; }

export interface NFLGameSummary {
  game_id: string; season: number; week: number; home_team: string; away_team: string;
  kickoff_utc: string | null; is_divisional: boolean | null; is_primetime: boolean | null;
  best_bet_type: string | null; best_bet_team: string | null; best_total_direction: string | null;
  best_bet_line: number | null; best_bet_value_score: number | null;
}
export interface NFLGamesResponse { games: NFLGameSummary[]; total: number; }

export interface NFLDailyPerformance {
  date: string; predictions: number; wins: number; losses: number; pushes: number;
  win_rate: number | null; profit: number;
}
export interface NFLMarketRecord {
  wins: number; losses: number; pushes: number; profit: number; win_rate: number | null; count: number;
}
export interface NFLEvaluationSummary {
  total_predictions: number; graded: number; wins: number; losses: number; pushes: number;
  win_rate: number | null; total_profit: number;
  by_market: Record<'best_bet' | 'spread' | 'ml', NFLMarketRecord>;
}

export const nflApi = {
  async getPicks(minValueScore = 40, limit = 20): Promise<NFLPicksResponse> {
    const r = await client.get<NFLPicksResponse>(`/nfl/picks?min_value_score=${minValueScore}&limit=${limit}`);
    return r.data;
  },
  async getGames(season?: number, week?: number): Promise<NFLGamesResponse> {
    const p = new URLSearchParams();
    if (season != null) p.set('season', String(season));
    if (week != null) p.set('week', String(week));
    const q = p.toString();
    const r = await client.get<NFLGamesResponse>(`/nfl/games${q ? `?${q}` : ''}`);
    return r.data;
  },
  async getDailyEvaluation(days = 30): Promise<NFLDailyPerformance[]> {
    const r = await client.get<NFLDailyPerformance[]>(`/nfl/evaluation/daily?days=${days}`);
    return r.data;
  },
  async getEvaluationSummary(): Promise<NFLEvaluationSummary> {
    const r = await client.get<NFLEvaluationSummary>('/nfl/evaluation/summary');
    return r.data;
  },
};

// Team colours (primary, secondary). Abbrs = our canonical NFL keys.
export const NFL_TEAMS: Record<string, { name: string; primary: string; secondary: string }> = {
  ARI:{name:'Cardinals',primary:'#97233F',secondary:'#000000'}, ATL:{name:'Falcons',primary:'#A71930',secondary:'#000000'},
  BAL:{name:'Ravens',primary:'#241773',secondary:'#9E7C0C'}, BUF:{name:'Bills',primary:'#00338D',secondary:'#C60C30'},
  CAR:{name:'Panthers',primary:'#0085CA',secondary:'#101820'}, CHI:{name:'Bears',primary:'#0B162A',secondary:'#C83803'},
  CIN:{name:'Bengals',primary:'#FB4F14',secondary:'#000000'}, CLE:{name:'Browns',primary:'#311D00',secondary:'#FF3C00'},
  DAL:{name:'Cowboys',primary:'#003594',secondary:'#869397'}, DEN:{name:'Broncos',primary:'#FB4F14',secondary:'#002244'},
  DET:{name:'Lions',primary:'#0076B6',secondary:'#B0B7BC'}, GB:{name:'Packers',primary:'#203731',secondary:'#FFB81C'},
  HOU:{name:'Texans',primary:'#03202F',secondary:'#A71930'}, IND:{name:'Colts',primary:'#002C5F',secondary:'#A2AAAD'},
  JAX:{name:'Jaguars',primary:'#101820',secondary:'#D7A22A'}, KC:{name:'Chiefs',primary:'#E31837',secondary:'#FFB81C'},
  LA:{name:'Rams',primary:'#003594',secondary:'#FFA300'}, LAC:{name:'Chargers',primary:'#0080C6',secondary:'#FFC20E'},
  LV:{name:'Raiders',primary:'#000000',secondary:'#A5ACAF'}, MIA:{name:'Dolphins',primary:'#008E97',secondary:'#FC4C02'},
  MIN:{name:'Vikings',primary:'#4F2683',secondary:'#FFC62F'}, NE:{name:'Patriots',primary:'#002244',secondary:'#C60C30'},
  NO:{name:'Saints',primary:'#D3BC8D',secondary:'#101820'}, NYG:{name:'Giants',primary:'#0B2265',secondary:'#A71930'},
  NYJ:{name:'Jets',primary:'#125740',secondary:'#000000'}, PHI:{name:'Eagles',primary:'#004C54',secondary:'#A5ACAF'},
  PIT:{name:'Steelers',primary:'#FFB612',secondary:'#101820'}, SEA:{name:'Seahawks',primary:'#002244',secondary:'#69BE28'},
  SF:{name:'49ers',primary:'#AA0000',secondary:'#B3995D'}, TB:{name:'Buccaneers',primary:'#D50A0A',secondary:'#34302B'},
  TEN:{name:'Titans',primary:'#0C2340',secondary:'#4B92DB'}, WAS:{name:'Commanders',primary:'#5A1414',secondary:'#FFB612'},
};
export function getTeamInfo(abbr: string) {
  return NFL_TEAMS[abbr] || { name: abbr, primary: '#333', secondary: '#777' };
}
export function formatOdds(decimal: number): string {
  return decimal >= 2.0 ? `+${Math.round((decimal - 1) * 100)}` : `${Math.round(-100 / (decimal - 1))}`;
}
