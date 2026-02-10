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
export interface PitcherInfo {
  pitcher_id: number;
  name: string;
  team: string | null;
  throws: string | null;
  era: number | null;
  whip: number | null;
  k_per_9: number | null;
  quality_score: number | null;
}

export interface GameContextInfo {
  venue_name: string | null;
  park_factor: number | null;
  temperature: number | null;
  wind_speed: number | null;
  is_dome: boolean;
  weather_factor: number | null;
}

export interface MarketInfo {
  market_type: string;
  line: number | null;
  home_odds: number | null;
  away_odds: number | null;
  over_odds: number | null;
  under_odds: number | null;
  book: string | null;
}

export interface ValueBetInfo {
  market_type: string;
  bet_type: string;
  team: string | null;
  line: number | null;
  odds_decimal: number;
  odds_american: number;
  model_prob: number;
  market_prob: number;
  edge: number;
  value_score: number;
  confidence: string;
}

export interface MLBGame {
  game_id: string;
  game_date: string;
  game_time: string | null;
  home_team: string;
  away_team: string;
  status: string;
  home_starter: PitcherInfo | null;
  away_starter: PitcherInfo | null;
  context: GameContextInfo | null;
  markets: MarketInfo[];
  predicted_run_diff: number | null;
  predicted_total: number | null;
  p_home_win: number | null;
  p_away_win: number | null;
  best_ml: ValueBetInfo | null;
  best_rl: ValueBetInfo | null;
  best_total: ValueBetInfo | null;
  best_bet: ValueBetInfo | null;
  home_score: number | null;
  away_score: number | null;
}

export interface MLBGamesResponse {
  games: MLBGame[];
  total: number;
  date: string;
}

export interface MLBTopPick {
  game_id: string;
  game_date: string;
  game_time: string | null;
  home_team: string;
  away_team: string;
  home_starter: string | null;
  away_starter: string | null;
  bet_type: string;
  team: string | null;
  line: number | null;
  odds_decimal: number;
  odds_american: number;
  value_score: number;
  edge: number;
  confidence: string;
  predicted_run_diff: number | null;
}

export interface MLBTopPicksResponse {
  picks: MLBTopPick[];
  total: number;
  min_value_score: number;
}

export interface MLBDailyPerformance {
  date: string;
  predictions: number;
  wins: number;
  losses: number;
  pushes: number;
  win_rate: number | null;
  profit: number;
}

export interface MLBEvaluationSummary {
  total_predictions: number;
  graded_predictions: number;
  wins: number;
  losses: number;
  pushes: number;
  overall_win_rate: number | null;
  total_profit: number;
  by_value_tier: Record<string, {
    wins: number;
    losses: number;
    pushes: number;
    profit: number;
    win_rate: number | null;
    count: number;
  }>;
}

// API functions
export const mlbApi = {
  async getGames(date?: string): Promise<MLBGamesResponse> {
    const params = date ? `?game_date=${date}` : '';
    const response = await client.get<MLBGamesResponse>(`/mlb/games${params}`);
    return response.data;
  },

  async getGame(gameId: string): Promise<MLBGame> {
    const response = await client.get<MLBGame>(`/mlb/games/${gameId}`);
    return response.data;
  },

  async getTopPicks(minValueScore: number = 65, date?: string, limit: number = 20): Promise<MLBTopPicksResponse> {
    let params = `?min_value_score=${minValueScore}&limit=${limit}`;
    if (date) params += `&game_date=${date}`;
    const response = await client.get<MLBTopPicksResponse>(`/mlb/picks/top${params}`);
    return response.data;
  },

  async getPitcher(name: string): Promise<PitcherInfo> {
    const response = await client.get<PitcherInfo>(`/mlb/pitchers/${encodeURIComponent(name)}`);
    return response.data;
  },

  async getDailyEvaluation(days: number = 7): Promise<MLBDailyPerformance[]> {
    const response = await client.get<MLBDailyPerformance[]>(`/mlb/evaluation/daily?days=${days}`);
    return response.data;
  },

  async getEvaluationSummary(): Promise<MLBEvaluationSummary> {
    const response = await client.get<MLBEvaluationSummary>('/mlb/evaluation/summary');
    return response.data;
  },
};

// Team logos/colors mapping
export const MLB_TEAMS: Record<string, { name: string; color: string; logo?: string }> = {
  'ARI': { name: 'D-backs', color: '#A71930' },
  'ATL': { name: 'Braves', color: '#CE1141' },
  'BAL': { name: 'Orioles', color: '#DF4601' },
  'BOS': { name: 'Red Sox', color: '#BD3039' },
  'CHC': { name: 'Cubs', color: '#0E3386' },
  'CWS': { name: 'White Sox', color: '#27251F' },
  'CIN': { name: 'Reds', color: '#C6011F' },
  'CLE': { name: 'Guardians', color: '#00385D' },
  'COL': { name: 'Rockies', color: '#333366' },
  'DET': { name: 'Tigers', color: '#0C2340' },
  'HOU': { name: 'Astros', color: '#002D62' },
  'KC': { name: 'Royals', color: '#004687' },
  'LAA': { name: 'Angels', color: '#BA0021' },
  'LAD': { name: 'Dodgers', color: '#005A9C' },
  'MIA': { name: 'Marlins', color: '#00A3E0' },
  'MIL': { name: 'Brewers', color: '#12284B' },
  'MIN': { name: 'Twins', color: '#002B5C' },
  'NYM': { name: 'Mets', color: '#002D72' },
  'NYY': { name: 'Yankees', color: '#003087' },
  'OAK': { name: 'Athletics', color: '#003831' },
  'PHI': { name: 'Phillies', color: '#E81828' },
  'PIT': { name: 'Pirates', color: '#27251F' },
  'SD': { name: 'Padres', color: '#2F241D' },
  'SF': { name: 'Giants', color: '#FD5A1E' },
  'SEA': { name: 'Mariners', color: '#0C2C56' },
  'STL': { name: 'Cardinals', color: '#C41E3A' },
  'TB': { name: 'Rays', color: '#092C5C' },
  'TEX': { name: 'Rangers', color: '#003278' },
  'TOR': { name: 'Blue Jays', color: '#134A8E' },
  'WSH': { name: 'Nationals', color: '#AB0003' },
};

export function getTeamInfo(abbr: string) {
  return MLB_TEAMS[abbr] || { name: abbr, color: '#333' };
}

export function formatOdds(decimal: number): string {
  if (decimal >= 2.0) {
    return `+${Math.round((decimal - 1) * 100)}`;
  } else {
    return `${Math.round(-100 / (decimal - 1))}`;
  }
}
