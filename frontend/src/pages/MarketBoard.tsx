import { useState, useMemo } from 'react';
import { useMarkets, useUpcomingGames, useGameHistory } from '@/hooks/useMarkets';
import { useDebounce } from '@/hooks/useDebounce';
import { GameCard } from '@/components/MarketBoard/GameCard';
import { HistoricalGameCard } from '@/components/MarketBoard/HistoricalGameCard';
import { TopPicks } from '@/components/MarketBoard/TopPicks';
import { ErrorMessage } from '@/components/ui/ErrorMessage';
import type { MarketFilters as Filters, Market, Algorithm } from '@/types/market';
import type { TeamTrends, GamePrediction, TornadoFactor } from '@/lib/api';

interface GameGroup {
  gameId: string;
  homeTeam: string;
  awayTeam: string;
  tipTime: string;
  markets: Market[];
  homeTrends?: TeamTrends;
  awayTrends?: TeamTrends;
  prediction?: GamePrediction | null;
  tornadoChart?: TornadoFactor[];
}

// Generate dates for the date picker
function getDateRange(): Date[] {
  const dates: Date[] = [];
  const today = new Date();

  // 3 days before and 4 days after
  for (let i = -3; i <= 4; i++) {
    const date = new Date(today);
    date.setDate(today.getDate() + i);
    dates.push(date);
  }

  return dates;
}

function formatDateLabel(date: Date): { day: string; date: string } {
  const today = new Date();
  const isToday = date.toDateString() === today.toDateString();

  if (isToday) {
    return {
      day: 'TODAY',
      date: date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }).toUpperCase(),
    };
  }

  return {
    day: date.toLocaleDateString('en-US', { weekday: 'short' }).toUpperCase(),
    date: date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }).toUpperCase(),
  };
}

function isPastDate(date: Date): boolean {
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const check = new Date(date);
  check.setHours(0, 0, 0, 0);
  return check < today;
}

function GameCardSkeleton() {
  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden animate-pulse">
      <div className="bg-slate-800 h-10" />
      <div className="p-4">
        <div className="flex justify-between items-center">
          <div className="flex items-center gap-3">
            <div className="w-14 h-14 bg-gray-200 rounded-full" />
            <div>
              <div className="h-6 bg-gray-200 rounded w-16 mb-2" />
              <div className="h-8 bg-gray-200 rounded w-24" />
            </div>
          </div>
          <div className="text-center">
            <div className="h-4 bg-gray-200 rounded w-24 mb-2 mx-auto" />
            <div className="h-4 bg-gray-100 rounded w-16 mx-auto" />
          </div>
          <div className="flex items-center gap-3">
            <div className="text-right">
              <div className="h-6 bg-gray-200 rounded w-16 mb-2 ml-auto" />
              <div className="h-8 bg-gray-200 rounded w-24" />
            </div>
            <div className="w-14 h-14 bg-gray-200 rounded-full" />
          </div>
        </div>
      </div>
    </div>
  );
}

export function MarketBoard() {
  const [algorithm, setAlgorithm] = useState<Algorithm>('b');
  const [selectedDate, setSelectedDate] = useState<Date>(new Date());

  const dates = useMemo(() => getDateRange(), []);
  const isViewingPast = useMemo(() => isPastDate(selectedDate), [selectedDate]);

  const filters: Partial<Filters> = { algorithm };
  const debouncedFilters = useDebounce(filters, 300);

  const { data: markets, isLoading, error, isFetching } = useMarkets(debouncedFilters);
  const { data: gamesWithTrends } = useUpcomingGames(24);
  const { data: historicalGames, isLoading: isLoadingHistory } = useGameHistory(7);

  // Create a map of game trends and predictions by game_id
  const trendsMap = useMemo(() => {
    const map = new Map<string, { home: TeamTrends; away: TeamTrends; prediction: GamePrediction | null; tornadoChart: TornadoFactor[] }>();
    if (gamesWithTrends) {
      for (const game of gamesWithTrends) {
        map.set(game.game_id, {
          home: game.home_trends,
          away: game.away_trends,
          prediction: game.prediction,
          tornadoChart: game.tornado_chart || [],
        });
      }
    }
    return map;
  }, [gamesWithTrends]);

  // Group markets by game and filter by selected date
  const games = useMemo(() => {
    if (!markets || markets.length === 0) return [];

    const gameMap = new Map<string, GameGroup>();

    for (const market of markets) {
      const existing = gameMap.get(market.game_id);
      if (existing) {
        existing.markets.push(market);
      } else {
        const homeTeam = (market as any).home_team || 'HOME';
        const awayTeam = (market as any).away_team || 'AWAY';
        const tipTime = (market as any).tip_time || new Date().toISOString();

        const trends = trendsMap.get(market.game_id);

        gameMap.set(market.game_id, {
          gameId: market.game_id,
          homeTeam,
          awayTeam,
          tipTime,
          markets: [market],
          homeTrends: trends?.home,
          awayTrends: trends?.away,
          prediction: trends?.prediction,
          tornadoChart: trends?.tornadoChart,
        });
      }
    }

    // Filter by selected date (using local date comparison)
    const selectedYear = selectedDate.getFullYear();
    const selectedMonth = selectedDate.getMonth();
    const selectedDay = selectedDate.getDate();

    const filteredGames = Array.from(gameMap.values()).filter(game => {
      const gameDate = new Date(game.tipTime);
      // Compare using local date components
      return gameDate.getFullYear() === selectedYear &&
             gameDate.getMonth() === selectedMonth &&
             gameDate.getDate() === selectedDay;
    });

    // Sort by tip time
    return filteredGames.sort(
      (a, b) => new Date(a.tipTime).getTime() - new Date(b.tipTime).getTime()
    );
  }, [markets, trendsMap, selectedDate]);

  // Count games for each date
  const gameCountByDate = useMemo(() => {
    if (!markets || markets.length === 0) return new Map<string, number>();

    const gameMap = new Map<string, Set<string>>();

    for (const market of markets) {
      const tipTime = (market as any).tip_time;
      if (tipTime) {
        const dateStr = new Date(tipTime).toDateString();
        if (!gameMap.has(dateStr)) {
          gameMap.set(dateStr, new Set());
        }
        gameMap.get(dateStr)!.add(market.game_id);
      }
    }

    const counts = new Map<string, number>();
    gameMap.forEach((games, date) => {
      counts.set(date, games.size);
    });

    return counts;
  }, [markets]);

  // Filter historical games by selected date
  const filteredHistoricalGames = useMemo(() => {
    if (!historicalGames || !isViewingPast) return [];

    const selectedYear = selectedDate.getFullYear();
    const selectedMonth = selectedDate.getMonth();
    const selectedDay = selectedDate.getDate();

    return historicalGames.filter(game => {
      const gameDate = new Date(game.game_date + 'T12:00:00');
      return gameDate.getFullYear() === selectedYear &&
             gameDate.getMonth() === selectedMonth &&
             gameDate.getDate() === selectedDay;
    });
  }, [historicalGames, selectedDate, isViewingPast]);

  // Count historical games for past dates
  const historicalCountByDate = useMemo(() => {
    if (!historicalGames) return new Map<string, number>();

    const counts = new Map<string, number>();
    for (const game of historicalGames) {
      const dateStr = new Date(game.game_date + 'T12:00:00').toDateString();
      counts.set(dateStr, (counts.get(dateStr) || 0) + 1);
    }
    return counts;
  }, [historicalGames]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">NBA Market Board</h1>
        </div>
        <div className="flex items-center gap-4">
          {/* Filter dropdown placeholder */}
          <button className="flex items-center gap-2 px-3 py-2 border border-gray-300 rounded-lg text-sm text-gray-700 hover:bg-gray-50">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2.586a1 1 0 01-.293.707l-6.414 6.414a1 1 0 00-.293.707V17l-4 4v-6.586a1 1 0 00-.293-.707L3.293 7.293A1 1 0 013 6.586V4z" />
            </svg>
            Filter
          </button>

          {/* Search placeholder */}
          <div className="relative">
            <svg className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
            <input
              type="text"
              placeholder="Search"
              className="pl-10 pr-4 py-2 border border-gray-300 rounded-lg text-sm w-40 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>
        </div>
      </div>

      {/* Date Picker */}
      <div className="flex items-center gap-1 bg-white rounded-lg border border-gray-200 p-1">
        <button className="p-2 hover:bg-gray-100 rounded">
          <svg className="w-5 h-5 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
          </svg>
        </button>

        {dates.map((date) => {
          const { day, date: dateLabel } = formatDateLabel(date);
          const isSelected = date.toDateString() === selectedDate.toDateString();
          const isPast = isPastDate(date);
          const gameCount = isPast
            ? historicalCountByDate.get(date.toDateString()) || 0
            : gameCountByDate.get(date.toDateString()) || 0;

          return (
            <button
              key={date.toISOString()}
              onClick={() => setSelectedDate(date)}
              className={`flex-1 py-2 px-3 rounded-lg text-center transition-colors ${
                isSelected
                  ? 'bg-slate-800 text-white'
                  : 'hover:bg-gray-100 text-gray-700'
              }`}
            >
              <div className="text-xs font-medium">{day}</div>
              <div className="text-xs">{dateLabel}</div>
              {gameCount > 0 && (
                <div className={`text-[10px] mt-0.5 ${isSelected ? 'text-gray-300' : 'text-gray-400'}`}>
                  {gameCount} games
                </div>
              )}
            </button>
          );
        })}

        <button className="p-2 hover:bg-gray-100 rounded">
          <svg className="w-5 h-5 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
          </svg>
        </button>

        {/* Calendar button */}
        <button className="p-2 hover:bg-gray-100 rounded border-l border-gray-200 ml-1">
          <svg className="w-5 h-5 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
          </svg>
        </button>
      </div>

      {/* Algorithm Toggle & Loading */}
      <div className="flex justify-between items-center">
        <div className="flex items-center gap-4">
          <span className="text-sm text-gray-500">Value Algorithm:</span>
          <div className="flex rounded-lg overflow-hidden border border-gray-200">
            <button
              onClick={() => setAlgorithm('a')}
              className={`px-4 py-2 text-sm font-medium transition-colors ${
                algorithm === 'a'
                  ? 'bg-blue-600 text-white'
                  : 'bg-white text-gray-700 hover:bg-gray-50'
              }`}
            >
              Algo A
            </button>
            <button
              onClick={() => setAlgorithm('b')}
              className={`px-4 py-2 text-sm font-medium transition-colors ${
                algorithm === 'b'
                  ? 'bg-emerald-600 text-white'
                  : 'bg-white text-gray-700 hover:bg-gray-50'
              }`}
            >
              Algo B
            </button>
          </div>
        </div>

        {isFetching && !isLoading && (
          <div className="flex items-center gap-2 text-sm text-gray-500">
            <div className="w-4 h-4 border-2 border-blue-600 border-t-transparent rounded-full animate-spin" />
            Updating...
          </div>
        )}
      </div>

      {/* Legend */}
      <div className="flex flex-wrap gap-6 text-sm text-gray-500 bg-white rounded-lg border border-gray-200 p-3">
        <span className="font-medium text-gray-700">Value Score:</span>
        <div className="flex items-center gap-2">
          <span className="px-2 py-0.5 bg-emerald-600 text-white rounded text-xs font-semibold">70%+</span>
          <span>Strong Value</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="px-2 py-0.5 bg-amber-500 text-white rounded text-xs font-semibold">50-69%</span>
          <span>Moderate</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="px-2 py-0.5 bg-slate-400 text-white rounded text-xs font-semibold">&lt;50%</span>
          <span>Low Value</span>
        </div>
      </div>

      {/* Top Picks Summary - Only show for today */}
      {!isViewingPast && selectedDate.toDateString() === new Date().toDateString() && (
        <TopPicks />
      )}

      {/* Error State */}
      {error && <ErrorMessage error={error as Error} />}

      {/* Loading State */}
      {(isLoading || (isViewingPast && isLoadingHistory)) && (
        <div className="grid gap-6 lg:grid-cols-2">
          {[1, 2, 3, 4].map((i) => (
            <GameCardSkeleton key={i} />
          ))}
        </div>
      )}

      {/* Historical Games (Past Date) */}
      {isViewingPast && !isLoadingHistory && (
        <>
          {filteredHistoricalGames.length === 0 ? (
            <div className="text-center py-12 bg-white rounded-lg border border-gray-200">
              <svg className="w-16 h-16 mx-auto text-gray-300 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
              <p className="text-gray-500 text-lg">No game results for this date</p>
              <p className="text-sm text-gray-400 mt-1">
                Historical data is available for the past 7 days
              </p>
            </div>
          ) : (
            <>
              <div className="bg-slate-100 rounded-lg p-3 text-sm text-slate-700">
                Viewing completed games with results. Spread and total outcomes are based on closing lines.
              </div>
              <div className="grid gap-6 lg:grid-cols-2">
                {filteredHistoricalGames.map((game) => (
                  <HistoricalGameCard key={game.game_id} game={game} />
                ))}
              </div>
            </>
          )}
        </>
      )}

      {/* Upcoming Games (Today/Future) */}
      {!isViewingPast && !isLoading && (
        <>
          {games.length === 0 ? (
            <div className="text-center py-12 bg-white rounded-lg border border-gray-200">
              <svg className="w-16 h-16 mx-auto text-gray-300 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
              <p className="text-gray-500 text-lg">No games scheduled for this date</p>
              <p className="text-sm text-gray-400 mt-1">
                Select another date or check back later
              </p>
            </div>
          ) : (
            <div className="grid gap-6 lg:grid-cols-2">
              {games.map((game) => (
                <GameCard
                  key={game.gameId}
                  gameId={game.gameId}
                  homeTeam={game.homeTeam}
                  awayTeam={game.awayTeam}
                  tipTime={game.tipTime}
                  markets={game.markets}
                  algorithm={algorithm}
                  homeTrends={game.homeTrends}
                  awayTrends={game.awayTrends}
                  prediction={game.prediction}
                  tornadoChart={game.tornadoChart}
                />
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
