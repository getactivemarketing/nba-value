import { useState, useMemo } from 'react';
import { useMarkets } from '@/hooks/useMarkets';
import { useDebounce } from '@/hooks/useDebounce';
import { GameCard } from '@/components/MarketBoard/GameCard';
import { ErrorMessage } from '@/components/ui/ErrorMessage';
import type { MarketFilters as Filters, Market, Algorithm } from '@/types/market';

interface GameGroup {
  gameId: string;
  homeTeam: string;
  awayTeam: string;
  tipTime: string;
  markets: Market[];
}

function GameCardSkeleton() {
  return (
    <div className="card animate-pulse">
      <div className="flex justify-between items-start mb-4">
        <div>
          <div className="h-6 bg-gray-200 rounded w-32 mb-2" />
          <div className="h-4 bg-gray-100 rounded w-24" />
        </div>
        <div className="h-8 bg-gray-200 rounded-full w-12" />
      </div>
      <div className="space-y-3">
        {[1, 2, 3].map((i) => (
          <div key={i} className="flex justify-between items-center py-2">
            <div className="h-4 bg-gray-200 rounded w-20" />
            <div className="flex space-x-8">
              <div className="h-6 bg-gray-200 rounded-full w-10" />
              <div className="h-6 bg-gray-200 rounded-full w-10" />
              <div className="h-6 bg-gray-200 rounded-full w-10" />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export function MarketBoard() {
  const [algorithm, setAlgorithm] = useState<Algorithm>('a');

  const filters: Partial<Filters> = { algorithm };
  const debouncedFilters = useDebounce(filters, 300);

  const { data: markets, isLoading, error, isFetching } = useMarkets(debouncedFilters);

  // Group markets by game
  const games = useMemo(() => {
    if (!markets || markets.length === 0) return [];

    const gameMap = new Map<string, GameGroup>();

    for (const market of markets) {
      const existing = gameMap.get(market.game_id);
      if (existing) {
        existing.markets.push(market);
      } else {
        // Extract home/away from the market data
        // The API returns home_team and away_team on the market object
        const homeTeam = (market as any).home_team || 'HOME';
        const awayTeam = (market as any).away_team || 'AWAY';
        const tipTime = (market as any).tip_time || new Date().toISOString();

        gameMap.set(market.game_id, {
          gameId: market.game_id,
          homeTeam,
          awayTeam,
          tipTime,
          markets: [market],
        });
      }
    }

    // Sort by tip time
    return Array.from(gameMap.values()).sort(
      (a, b) => new Date(a.tipTime).getTime() - new Date(b.tipTime).getTime()
    );
  }, [markets]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Today's Games</h1>
          <p className="text-sm text-gray-500 mt-1">
            Value Scores by sportsbook
          </p>
        </div>
        <div className="flex items-center space-x-4">
          {isFetching && !isLoading && (
            <div className="w-4 h-4 border-2 border-blue-600 border-t-transparent rounded-full animate-spin" />
          )}

          {/* Algorithm Toggle */}
          <div className="flex rounded-lg overflow-hidden border border-gray-200">
            <button
              onClick={() => setAlgorithm('a')}
              className={`px-3 py-1.5 text-sm font-medium transition-colors ${
                algorithm === 'a'
                  ? 'bg-blue-600 text-white'
                  : 'bg-white text-gray-700 hover:bg-gray-50'
              }`}
            >
              Algo A
            </button>
            <button
              onClick={() => setAlgorithm('b')}
              className={`px-3 py-1.5 text-sm font-medium transition-colors ${
                algorithm === 'b'
                  ? 'bg-green-600 text-white'
                  : 'bg-white text-gray-700 hover:bg-gray-50'
              }`}
            >
              Algo B
            </button>
          </div>
        </div>
      </div>

      {/* Legend */}
      <div className="flex flex-wrap gap-4 text-xs text-gray-500">
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 rounded-full bg-green-500" />
          <span>80+ Strong Value</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 rounded-full bg-green-400" />
          <span>60-79 Good Value</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 rounded-full bg-yellow-400" />
          <span>40-59 Moderate</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 rounded-full bg-gray-300" />
          <span>&lt;40 Low Value</span>
        </div>
      </div>

      {/* Error State */}
      {error && <ErrorMessage error={error as Error} />}

      {/* Loading State */}
      {isLoading && (
        <div className="grid gap-6 md:grid-cols-2">
          {[1, 2, 3, 4].map((i) => (
            <GameCardSkeleton key={i} />
          ))}
        </div>
      )}

      {/* Empty State */}
      {!isLoading && !error && games.length === 0 && (
        <div className="text-center py-12">
          <p className="text-gray-500">No games scheduled for today</p>
          <p className="text-sm text-gray-400 mt-1">
            Check back later for upcoming games
          </p>
        </div>
      )}

      {/* Game Cards */}
      {!isLoading && games.length > 0 && (
        <div className="grid gap-6 md:grid-cols-2">
          {games.map((game) => (
            <GameCard
              key={game.gameId}
              gameId={game.gameId}
              homeTeam={game.homeTeam}
              awayTeam={game.awayTeam}
              tipTime={game.tipTime}
              markets={game.markets}
              algorithm={algorithm}
            />
          ))}
        </div>
      )}
    </div>
  );
}
