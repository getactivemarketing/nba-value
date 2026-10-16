// frontend/src/types/edge.ts
export interface EvRow {
  event_id: string; matchup: string; commence_time: string; market: string; market_label: string;
  player: string; side: string; line: number; bet: string; book: string; price: number;
  link: string | null; fair_prob: number; fair_price: number; ev: number; n_books: number;
  observed_at: string; verify: boolean; bdl_player_id: number | null;
}
export interface EdgeGame {
  event_id: string; home_team: string; away_team: string; commence_time: string;
  last_lines_poll: string | null; last_props_poll: string | null; prop_markets: number;
}
export interface PriceCell { price: number; link: string | null; observed_at: string; stale: boolean; }
export interface OddsGridRow {
  player: string; line: number; side: string; prices: Record<string, PriceCell>;
  best_book?: string; fair_prob: number | null; fair_price: number | null;
}
export interface OddsResponse {
  event: { event_id: string; home_team: string; away_team: string; commence_time: string } | null;
  market: string; rows: OddsGridRow[];
}
export interface HitRate { hits: number; misses: number; pushes: number; games: number; rate: number | null; }
export interface PlayerGame {
  game_date: string; opponent: string; is_home: boolean; minutes: number; value: number;
}
export interface PlayerResponse {
  bdl_player_id: number; name: string | null; team: string | null; market: string; line: number | null;
  opponent: string | null; is_home: boolean | null;
  hit_rates: Record<'l5' | 'l10' | 'l20' | 'season' | 'vs_opponent' | 'home_away', HitRate> | null;
  games: PlayerGame[]; prices: { book: string; side: string; line: number; price: number; link: string | null }[];
}
export interface ClvBlock {
  measured: number; bets: number; mean_clv: number | null; median_clv: number | null; beat_close_rate: number | null;
  std_error: number | null; verdict: string; ci95: [number, number] | null;
}
export interface RecordBlock {
  wins: number; losses: number; pushes: number; voids: number; win_rate: number | null;
  units: number; roi: number | null;
}
export interface ResultsBlock { n: number; clv: ClvBlock; record: RecordBlock; }
export interface ResultsResponse extends ResultsBlock {
  by_market: Record<string, ResultsBlock>; by_book: Record<string, ResultsBlock>;
  daily_clv: { date: string; mean_clv: number; n: number }[];
}
