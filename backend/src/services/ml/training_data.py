"""Build training dataset for MOV model from NBA API and historical odds."""

import asyncio
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
import pandas as pd
import numpy as np
from pathlib import Path
import structlog
import time

# NBA API imports
from nba_api.stats.endpoints import (
    LeagueGameLog,
    TeamEstimatedMetrics,
)
from nba_api.stats.static import teams

logger = structlog.get_logger()

# Team name to abbreviation mapping
TEAM_ABBREV = {t['full_name']: t['abbreviation'] for t in teams.get_teams()}
TEAM_ID_MAP = {t['id']: t['abbreviation'] for t in teams.get_teams()}


@dataclass
class TrainingGame:
    """A game with features and target for training."""
    game_id: str
    game_date: str
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    margin: int  # home - away (positive = home win)
    closing_spread: float | None

    # Home team features
    home_ortg_10: float | None
    home_drtg_10: float | None
    home_net_rtg_10: float | None
    home_pace_10: float | None
    home_win_pct_10: float | None
    home_rest_days: int | None  # Days since last game
    home_b2b: int | None  # 1 if back-to-back, 0 otherwise

    # Away team features
    away_ortg_10: float | None
    away_drtg_10: float | None
    away_net_rtg_10: float | None
    away_pace_10: float | None
    away_win_pct_10: float | None
    away_rest_days: int | None
    away_b2b: int | None


def fetch_season_games(season: str = "2024-25") -> pd.DataFrame:
    """Fetch all games for a season from NBA API."""
    logger.info(f"Fetching games for season {season}")

    # Small delay to avoid rate limiting
    time.sleep(0.5)

    game_log = LeagueGameLog(
        season=season,
        season_type_all_star="Regular Season",
    )

    df = game_log.get_data_frames()[0]
    logger.info(f"Fetched {len(df)} game records")

    return df


def calculate_rolling_stats(games_df: pd.DataFrame, window: int = 10) -> dict:
    """
    Calculate rolling team stats for each game.

    Returns dict mapping (team_id, game_date) -> stats dict
    """
    logger.info(f"Calculating rolling stats with window={window}")

    # Sort by date
    games_df = games_df.sort_values('GAME_DATE')

    team_stats = {}

    for team_id in games_df['TEAM_ID'].unique():
        team_games = games_df[games_df['TEAM_ID'] == team_id].copy()
        team_games = team_games.sort_values('GAME_DATE')

        # Calculate rolling stats
        team_games['PTS_ROLL'] = team_games['PTS'].rolling(window, min_periods=3).mean()
        team_games['OPP_PTS_ROLL'] = (team_games['PTS'] - team_games['PLUS_MINUS']).rolling(window, min_periods=3).mean()
        team_games['WIN_ROLL'] = team_games['WL'].map({'W': 1, 'L': 0}).rolling(window, min_periods=3).mean()

        # Estimate ratings from points (simplified)
        # ORtg ≈ (PTS / Poss) * 100, using pace ≈ 100
        team_games['ORTG_EST'] = team_games['PTS_ROLL'] * 100 / 100  # Simplified
        team_games['DRTG_EST'] = team_games['OPP_PTS_ROLL'] * 100 / 100

        # Calculate rest days (days since last game)
        team_games['GAME_DATE_DT'] = pd.to_datetime(team_games['GAME_DATE'])
        team_games['PREV_GAME_DATE'] = team_games['GAME_DATE_DT'].shift(1)
        team_games['REST_DAYS'] = (team_games['GAME_DATE_DT'] - team_games['PREV_GAME_DATE']).dt.days
        team_games['IS_B2B'] = (team_games['REST_DAYS'] == 1).astype(int)

        for _, row in team_games.iterrows():
            key = (team_id, row['GAME_DATE'])
            rest_days = row['REST_DAYS'] if pd.notna(row['REST_DAYS']) else 3  # Default to 3 for first game
            team_stats[key] = {
                'ortg_10': row['ORTG_EST'],
                'drtg_10': row['DRTG_EST'],
                'net_rtg_10': row['ORTG_EST'] - row['DRTG_EST'] if pd.notna(row['ORTG_EST']) else None,
                'pace_10': 100.0,  # Placeholder
                'win_pct_10': row['WIN_ROLL'],
                'rest_days': int(min(rest_days, 7)),  # Cap at 7 days
                'is_b2b': int(row['IS_B2B']) if pd.notna(row['IS_B2B']) else 0,
            }

    return team_stats


def build_training_dataset(
    seasons: list[str] = ["2023-24", "2024-25"],
    closing_spreads: dict | None = None,
) -> list[TrainingGame]:
    """
    Build training dataset from NBA API data.

    Args:
        seasons: List of seasons to include
        closing_spreads: Optional dict mapping game_id -> closing spread

    Returns:
        List of TrainingGame objects
    """
    all_games = []

    for season in seasons:
        try:
            games_df = fetch_season_games(season)
            time.sleep(1)  # Rate limit protection

            # Calculate rolling stats
            team_stats = calculate_rolling_stats(games_df, window=10)

            # Group by game_id to get both teams
            grouped = games_df.groupby('GAME_ID')

            for game_id, game_rows in grouped:
                if len(game_rows) != 2:
                    continue

                # Determine home/away
                home_row = game_rows[game_rows['MATCHUP'].str.contains(' vs. ')].iloc[0] if len(game_rows[game_rows['MATCHUP'].str.contains(' vs. ')]) > 0 else None
                away_row = game_rows[game_rows['MATCHUP'].str.contains(' @ ')].iloc[0] if len(game_rows[game_rows['MATCHUP'].str.contains(' @ ')]) > 0 else None

                if home_row is None or away_row is None:
                    continue

                game_date = home_row['GAME_DATE']
                home_team_id = home_row['TEAM_ID']
                away_team_id = away_row['TEAM_ID']

                home_team = TEAM_ID_MAP.get(home_team_id, str(home_team_id))
                away_team = TEAM_ID_MAP.get(away_team_id, str(away_team_id))

                home_score = int(home_row['PTS'])
                away_score = int(away_row['PTS'])
                margin = home_score - away_score

                # Get rolling stats at game time
                home_stats = team_stats.get((home_team_id, game_date), {})
                away_stats = team_stats.get((away_team_id, game_date), {})

                # Get closing spread if available
                spread = closing_spreads.get(game_id) if closing_spreads else None

                training_game = TrainingGame(
                    game_id=game_id,
                    game_date=game_date,
                    home_team=home_team,
                    away_team=away_team,
                    home_score=home_score,
                    away_score=away_score,
                    margin=margin,
                    closing_spread=spread,
                    home_ortg_10=home_stats.get('ortg_10'),
                    home_drtg_10=home_stats.get('drtg_10'),
                    home_net_rtg_10=home_stats.get('net_rtg_10'),
                    home_pace_10=home_stats.get('pace_10'),
                    home_win_pct_10=home_stats.get('win_pct_10'),
                    home_rest_days=home_stats.get('rest_days'),
                    home_b2b=home_stats.get('is_b2b'),
                    away_ortg_10=away_stats.get('ortg_10'),
                    away_drtg_10=away_stats.get('drtg_10'),
                    away_net_rtg_10=away_stats.get('net_rtg_10'),
                    away_pace_10=away_stats.get('pace_10'),
                    away_win_pct_10=away_stats.get('win_pct_10'),
                    away_rest_days=away_stats.get('rest_days'),
                    away_b2b=away_stats.get('is_b2b'),
                )

                all_games.append(training_game)

        except Exception as e:
            logger.error(f"Error fetching season {season}: {e}")
            continue

    logger.info(f"Built training dataset with {len(all_games)} games")
    return all_games


def games_to_dataframe(games: list[TrainingGame]) -> pd.DataFrame:
    """Convert list of TrainingGame to pandas DataFrame."""
    return pd.DataFrame([
        {
            'game_id': g.game_id,
            'game_date': g.game_date,
            'home_team': g.home_team,
            'away_team': g.away_team,
            'home_score': g.home_score,
            'away_score': g.away_score,
            'margin': g.margin,
            'closing_spread': g.closing_spread,
            'home_ortg_10': g.home_ortg_10,
            'home_drtg_10': g.home_drtg_10,
            'home_net_rtg_10': g.home_net_rtg_10,
            'home_pace_10': g.home_pace_10,
            'home_win_pct_10': g.home_win_pct_10,
            'home_rest_days': g.home_rest_days,
            'home_b2b': g.home_b2b,
            'away_ortg_10': g.away_ortg_10,
            'away_drtg_10': g.away_drtg_10,
            'away_net_rtg_10': g.away_net_rtg_10,
            'away_pace_10': g.away_pace_10,
            'away_win_pct_10': g.away_win_pct_10,
            'away_rest_days': g.away_rest_days,
            'away_b2b': g.away_b2b,
        }
        for g in games
    ])


def prepare_training_arrays(df: pd.DataFrame, include_rest: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepare feature matrix X and target vector y for model training.

    Features:
    - home_ortg_10, home_drtg_10, home_net_rtg_10, home_pace_10, home_win_pct_10
    - away_ortg_10, away_drtg_10, away_net_rtg_10, away_pace_10, away_win_pct_10
    - home_rest_days, home_b2b, away_rest_days, away_b2b (if include_rest=True)

    Target: margin (home - away)
    """
    feature_cols = [
        'home_ortg_10', 'home_drtg_10', 'home_net_rtg_10', 'home_pace_10', 'home_win_pct_10',
        'away_ortg_10', 'away_drtg_10', 'away_net_rtg_10', 'away_pace_10', 'away_win_pct_10',
    ]

    if include_rest:
        feature_cols.extend([
            'home_rest_days', 'home_b2b',
            'away_rest_days', 'away_b2b',
        ])

    # Drop rows with missing features
    df_clean = df.dropna(subset=feature_cols)

    X = df_clean[feature_cols].values
    y = df_clean['margin'].values

    logger.info(f"Prepared {len(X)} training samples with {len(feature_cols)} features, dropped {len(df) - len(df_clean)} with missing data")

    return X, y, df_clean, feature_cols


def save_training_data(df: pd.DataFrame, path: str | Path) -> None:
    """Save training data to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Saved training data to {path}")


def load_training_data(path: str | Path) -> pd.DataFrame:
    """Load training data from CSV."""
    return pd.read_csv(path)


if __name__ == "__main__":
    # Build training dataset
    print("Building training dataset...")
    games = build_training_dataset(seasons=["2023-24", "2024-25"])

    df = games_to_dataframe(games)
    print(f"\nDataset shape: {df.shape}")
    print(f"\nSample data:")
    print(df.head())

    print(f"\nMargin distribution:")
    print(df['margin'].describe())

    # Save for later use
    save_training_data(df, "data/training_games.csv")

    # Prepare arrays
    X, y, df_clean = prepare_training_arrays(df)
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
