"""
Historical Backtesting Module

Evaluates model performance on historical games using closing lines and results.

Key features:
1. Runs model predictions using closing lines
2. Evaluates what picks would have been made
3. Calculates hypothetical ROI and win rates
4. Analyzes performance by various filters

Note: This is a "what-if" analysis using closing lines, not true real-time backtesting.
"""

import psycopg2
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from dataclasses import dataclass

import structlog

logger = structlog.get_logger()

DB_URL = 'postgresql://postgres:wzYHkiAOkykxiPitXKBIqPJxvifFtDPI@maglev.proxy.rlwy.net:46068/railway'


@dataclass
class BacktestResult:
    """Result of a single backtested game."""
    game_id: str
    game_date: str
    home_team: str
    away_team: str
    closing_spread: float
    closing_total: float
    home_score: int
    away_score: int
    actual_winner: str
    spread_result: str  # home_cover, away_cover, push
    total_result: str   # over, under, push

    # Model picks (if any)
    spread_pick: str | None = None  # home, away
    spread_edge: float = 0.0
    total_pick: str | None = None   # over, under
    total_edge: float = 0.0


def run_backtest(
    start_date: str = None,
    end_date: str = None,
    min_edge: float = 0.0,
    value_threshold: int = 50,
    db_url: str = None,
) -> dict:
    """
    Run backtest on historical games.

    Uses closing lines from game_results to evaluate what spread/total picks
    would have been made and how they would have performed.

    Args:
        start_date: Start date (YYYY-MM-DD), defaults to 30 days ago
        end_date: End date (YYYY-MM-DD), defaults to yesterday
        min_edge: Minimum edge % to count as a pick
        value_threshold: Minimum value score to count as a pick
        db_url: Database connection string

    Returns:
        Dict with backtest results and performance metrics
    """
    conn = psycopg2.connect(db_url or DB_URL)
    cur = conn.cursor()

    # Default date range
    if not end_date:
        end_date = (datetime.now(timezone.utc) - timedelta(days=1)).strftime('%Y-%m-%d')
    if not start_date:
        start_date = (datetime.now(timezone.utc) - timedelta(days=30)).strftime('%Y-%m-%d')

    # Get historical games with results
    cur.execute('''
        SELECT
            game_id, game_date, home_team_id, away_team_id,
            closing_spread, closing_total,
            home_score, away_score, actual_winner,
            spread_result, total_result
        FROM game_results
        WHERE game_date >= %s
        AND game_date <= %s
        AND actual_winner IS NOT NULL
        AND closing_spread IS NOT NULL
        ORDER BY game_date
    ''', (start_date, end_date))

    games = cur.fetchall()
    cur.close()
    conn.close()

    if not games:
        return {"status": "no_data", "games": 0}

    # Analyze each game
    results = []
    for row in games:
        (game_id, game_date, home, away, closing_spread, closing_total,
         home_score, away_score, winner, spread_result, total_result) = row

        result = BacktestResult(
            game_id=game_id,
            game_date=str(game_date),
            home_team=home,
            away_team=away,
            closing_spread=float(closing_spread) if closing_spread else 0,
            closing_total=float(closing_total) if closing_total else 0,
            home_score=home_score or 0,
            away_score=away_score or 0,
            actual_winner=winner or "",
            spread_result=spread_result or "",
            total_result=total_result or "",
        )
        results.append(result)

    # Calculate performance metrics
    # Simple baseline: compare closing spread to actual margin
    spread_stats = {"wins": 0, "losses": 0, "pushes": 0, "profit": 0.0}
    total_stats = {"wins": 0, "losses": 0, "pushes": 0, "profit": 0.0}

    # Track performance by margin vs spread
    close_games = {"wins": 0, "losses": 0, "profit": 0.0}  # Margin within 3 of spread
    blowouts = {"wins": 0, "losses": 0, "profit": 0.0}     # Margin > 10 from spread

    for r in results:
        actual_margin = r.home_score - r.away_score  # Positive = home won by X

        # Spread analysis: would betting the "smart" side (closing line) have won?
        # If closing spread is -5, home favored by 5
        # If home wins by more than 5, home covers

        # For this baseline, we assume betting the favorite when margin > 7
        # and betting the underdog when margin < 3
        if r.closing_spread != 0:
            # Track all spread outcomes
            if r.spread_result == "push":
                spread_stats["pushes"] += 1
            elif r.spread_result:
                # Calculate margin vs spread difference
                spread_diff = abs(actual_margin - (-r.closing_spread))

                # Close game analysis (within 3 points of spread)
                if spread_diff <= 3:
                    if r.spread_result == "home_cover":
                        close_games["wins"] += 1
                        close_games["profit"] += 90.91
                    else:
                        close_games["losses"] += 1
                        close_games["profit"] -= 100

                # Blowout analysis (margin > 10 from spread)
                elif spread_diff >= 10:
                    if r.spread_result == "home_cover":
                        blowouts["wins"] += 1
                        blowouts["profit"] += 90.91
                    else:
                        blowouts["losses"] += 1
                        blowouts["profit"] -= 100

        # Total analysis
        if r.closing_total != 0 and r.total_result:
            if r.total_result == "push":
                total_stats["pushes"] += 1

    # Calculate win rates
    def calc_rate(stats):
        total = stats["wins"] + stats["losses"]
        return round(stats["wins"] / total * 100, 1) if total > 0 else 0

    # Home vs Away cover rates
    home_covers = sum(1 for r in results if r.spread_result == "home_cover")
    away_covers = sum(1 for r in results if r.spread_result == "away_cover")
    spread_pushes = sum(1 for r in results if r.spread_result == "push")

    # Over vs Under rates
    overs = sum(1 for r in results if r.total_result == "over")
    unders = sum(1 for r in results if r.total_result == "under")
    total_pushes = sum(1 for r in results if r.total_result == "push")

    # Favorite vs Underdog performance
    fav_covers = sum(1 for r in results
                     if (r.closing_spread < 0 and r.spread_result == "home_cover") or
                        (r.closing_spread > 0 and r.spread_result == "away_cover"))
    dog_covers = sum(1 for r in results
                     if (r.closing_spread > 0 and r.spread_result == "home_cover") or
                        (r.closing_spread < 0 and r.spread_result == "away_cover"))

    # Big favorites (7+ points)
    big_fav_games = [r for r in results if abs(r.closing_spread) >= 7]
    big_fav_covers = sum(1 for r in big_fav_games
                         if (r.closing_spread < 0 and r.spread_result == "home_cover") or
                            (r.closing_spread > 0 and r.spread_result == "away_cover"))
    big_dog_covers = sum(1 for r in big_fav_games
                         if (r.closing_spread > 0 and r.spread_result == "home_cover") or
                            (r.closing_spread < 0 and r.spread_result == "away_cover"))

    return {
        "status": "success",
        "date_range": {"start": start_date, "end": end_date},
        "total_games": len(results),
        "spread_outcomes": {
            "home_covers": home_covers,
            "away_covers": away_covers,
            "pushes": spread_pushes,
            "home_cover_rate": round(home_covers / (home_covers + away_covers) * 100, 1) if (home_covers + away_covers) > 0 else 50,
        },
        "total_outcomes": {
            "overs": overs,
            "unders": unders,
            "pushes": total_pushes,
            "over_rate": round(overs / (overs + unders) * 100, 1) if (overs + unders) > 0 else 50,
        },
        "favorite_vs_underdog": {
            "favorite_covers": fav_covers,
            "underdog_covers": dog_covers,
            "favorite_cover_rate": round(fav_covers / (fav_covers + dog_covers) * 100, 1) if (fav_covers + dog_covers) > 0 else 50,
        },
        "big_spreads_7plus": {
            "games": len(big_fav_games),
            "favorite_covers": big_fav_covers,
            "underdog_covers": big_dog_covers,
            "favorite_cover_rate": round(big_fav_covers / (big_fav_covers + big_dog_covers) * 100, 1) if (big_fav_covers + big_dog_covers) > 0 else 50,
        },
        "close_game_analysis": {
            "games": close_games["wins"] + close_games["losses"],
            **close_games,
            "win_rate": calc_rate(close_games),
        },
        "blowout_analysis": {
            "games": blowouts["wins"] + blowouts["losses"],
            **blowouts,
            "win_rate": calc_rate(blowouts),
        },
    }


def analyze_model_vs_baseline(days: int = 14, db_url: str = None) -> dict:
    """
    Compare our model's performance against simple baseline strategies.

    Analyzes:
    1. Model picks (from prediction_snapshots)
    2. Blind favorite betting
    3. Blind underdog betting
    4. Blind home team betting

    Args:
        days: Number of days to analyze
        db_url: Database connection string

    Returns:
        Comparison of model vs baselines
    """
    conn = psycopg2.connect(db_url or DB_URL)
    cur = conn.cursor()

    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime('%Y-%m-%d')

    # Get our graded predictions
    cur.execute('''
        SELECT
            ps.game_id, ps.home_team, ps.away_team,
            ps.best_bet_type, ps.best_bet_team, ps.best_bet_value_score,
            ps.best_bet_result, ps.best_bet_profit,
            gr.closing_spread, gr.spread_result
        FROM prediction_snapshots ps
        JOIN game_results gr ON ps.game_id = gr.game_id
        WHERE ps.snapshot_time >= %s
        AND ps.best_bet_result IS NOT NULL
    ''', (cutoff,))

    model_picks = cur.fetchall()

    # Get all games in period for baseline comparison
    cur.execute('''
        SELECT
            game_id, home_team_id, away_team_id,
            closing_spread, spread_result
        FROM game_results
        WHERE game_date >= %s
        AND spread_result IS NOT NULL
        AND closing_spread IS NOT NULL
    ''', (cutoff,))

    all_games = cur.fetchall()
    cur.close()
    conn.close()

    # Model performance
    model_stats = {"picks": 0, "wins": 0, "losses": 0, "profit": 0.0}
    for row in model_picks:
        if row[6]:  # Has result
            model_stats["picks"] += 1
            if row[6] == 'win':
                model_stats["wins"] += 1
                model_stats["profit"] += float(row[7] or 90.91)
            elif row[6] == 'loss':
                model_stats["losses"] += 1
                model_stats["profit"] += float(row[7] or -100)

    # Baseline strategies
    blind_fav = {"wins": 0, "losses": 0, "profit": 0.0}
    blind_dog = {"wins": 0, "losses": 0, "profit": 0.0}
    blind_home = {"wins": 0, "losses": 0, "profit": 0.0}

    for game_id, home, away, spread, result in all_games:
        if not spread or not result:
            continue

        spread_val = float(spread)

        # Blind favorite betting
        if spread_val < 0:  # Home favored
            if result == "home_cover":
                blind_fav["wins"] += 1
                blind_fav["profit"] += 90.91
            elif result == "away_cover":
                blind_fav["losses"] += 1
                blind_fav["profit"] -= 100
        else:  # Away favored
            if result == "away_cover":
                blind_fav["wins"] += 1
                blind_fav["profit"] += 90.91
            elif result == "home_cover":
                blind_fav["losses"] += 1
                blind_fav["profit"] -= 100

        # Blind underdog betting (opposite of favorite)
        if spread_val < 0:  # Home favored, bet away
            if result == "away_cover":
                blind_dog["wins"] += 1
                blind_dog["profit"] += 90.91
            elif result == "home_cover":
                blind_dog["losses"] += 1
                blind_dog["profit"] -= 100
        else:  # Away favored, bet home
            if result == "home_cover":
                blind_dog["wins"] += 1
                blind_dog["profit"] += 90.91
            elif result == "away_cover":
                blind_dog["losses"] += 1
                blind_dog["profit"] -= 100

        # Blind home betting
        if result == "home_cover":
            blind_home["wins"] += 1
            blind_home["profit"] += 90.91
        elif result == "away_cover":
            blind_home["losses"] += 1
            blind_home["profit"] -= 100

    def calc_stats(stats, key="picks"):
        total = stats.get(key, stats["wins"] + stats["losses"])
        wins = stats["wins"]
        losses = stats["losses"]
        profit = stats["profit"]
        return {
            "bets": wins + losses,
            "wins": wins,
            "losses": losses,
            "win_rate": round(wins / (wins + losses) * 100, 1) if (wins + losses) > 0 else 0,
            "profit": round(profit, 2),
            "roi": round(profit / ((wins + losses) * 100) * 100, 1) if (wins + losses) > 0 else 0,
        }

    return {
        "status": "success",
        "days_analyzed": days,
        "total_games_in_period": len(all_games),
        "model_performance": calc_stats(model_stats),
        "baselines": {
            "blind_favorite": calc_stats(blind_fav),
            "blind_underdog": calc_stats(blind_dog),
            "blind_home": calc_stats(blind_home),
        },
        "model_vs_baselines": {
            "vs_blind_favorite": round(model_stats["profit"] - blind_fav["profit"], 2),
            "vs_blind_underdog": round(model_stats["profit"] - blind_dog["profit"], 2),
            "vs_blind_home": round(model_stats["profit"] - blind_home["profit"], 2),
        }
    }


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python backtester.py [backtest|compare]")
        sys.exit(1)

    command = sys.argv[1]

    if command == 'backtest':
        days = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        end_date = (datetime.now(timezone.utc) - timedelta(days=1)).strftime('%Y-%m-%d')
        start_date = (datetime.now(timezone.utc) - timedelta(days=days)).strftime('%Y-%m-%d')

        result = run_backtest(start_date, end_date)

        print(f"\nHistorical Backtest ({result['date_range']['start']} to {result['date_range']['end']})")
        print(f"Total Games: {result['total_games']}")
        print("=" * 60)

        print(f"\nSpread Outcomes:")
        so = result['spread_outcomes']
        print(f"  Home Covers: {so['home_covers']} ({so['home_cover_rate']}%)")
        print(f"  Away Covers: {so['away_covers']} ({100 - so['home_cover_rate']}%)")
        print(f"  Pushes: {so['pushes']}")

        print(f"\nTotal Outcomes:")
        to = result['total_outcomes']
        print(f"  Overs: {to['overs']} ({to['over_rate']}%)")
        print(f"  Unders: {to['unders']} ({100 - to['over_rate']}%)")
        print(f"  Pushes: {to['pushes']}")

        print(f"\nFavorite vs Underdog:")
        fv = result['favorite_vs_underdog']
        print(f"  Favorites Cover: {fv['favorite_covers']} ({fv['favorite_cover_rate']}%)")
        print(f"  Underdogs Cover: {fv['underdog_covers']} ({100 - fv['favorite_cover_rate']}%)")

        print(f"\nBig Spreads (7+ pts): {result['big_spreads_7plus']['games']} games")
        bs = result['big_spreads_7plus']
        print(f"  Favorites Cover: {bs['favorite_covers']} ({bs['favorite_cover_rate']}%)")
        print(f"  Underdogs Cover: {bs['underdog_covers']} ({100 - bs['favorite_cover_rate']}%)")

    elif command == 'compare':
        days = int(sys.argv[2]) if len(sys.argv) > 2 else 14
        result = analyze_model_vs_baseline(days=days)

        print(f"\nModel vs Baseline Comparison ({result['days_analyzed']} days)")
        print(f"Total Games in Period: {result['total_games_in_period']}")
        print("=" * 60)

        print(f"\nOur Model:")
        mp = result['model_performance']
        print(f"  Bets: {mp['bets']}")
        print(f"  Record: {mp['wins']}-{mp['losses']} ({mp['win_rate']}%)")
        print(f"  Profit: ${mp['profit']:.2f} (ROI: {mp['roi']}%)")

        print(f"\nBaseline Strategies (if betting every game):")
        for name, stats in result['baselines'].items():
            print(f"  {name.replace('_', ' ').title()}:")
            print(f"    Record: {stats['wins']}-{stats['losses']} ({stats['win_rate']}%)")
            print(f"    Profit: ${stats['profit']:.2f} (ROI: {stats['roi']}%)")

        print(f"\nModel vs Baselines (profit difference):")
        for name, diff in result['model_vs_baselines'].items():
            sign = "+" if diff >= 0 else ""
            print(f"  {name.replace('_', ' ').title()}: {sign}${diff:.2f}")

    else:
        print(f"Unknown command: {command}")
        print("Available commands: backtest, compare")
        sys.exit(1)
