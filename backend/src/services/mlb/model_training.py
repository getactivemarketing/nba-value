"""MLB Model Training Pipeline.

Fetches historical game data from MLB Stats API and trains
a LightGBM model to predict run differentials.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("Warning: LightGBM not installed. Install with: pip install lightgbm")

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False
    print("Warning: joblib not installed. Install with: pip install joblib")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MLB_API_BASE = "https://statsapi.mlb.com/api/v1"

# Park factors (run scoring environment relative to average)
PARK_FACTORS = {
    "COL": 1.15,  # Coors Field - highest
    "CIN": 1.08,  # Great American
    "TEX": 1.06,  # Globe Life (retractable)
    "BOS": 1.05,  # Fenway
    "PHI": 1.04,  # Citizens Bank
    "MIL": 1.03,  # American Family
    "CHC": 1.02,  # Wrigley
    "ATL": 1.02,  # Truist Park
    "ARI": 1.01,  # Chase Field (retractable)
    "BAL": 1.01,  # Camden Yards
    "TOR": 1.00,  # Rogers Centre (dome)
    "NYY": 1.00,  # Yankee Stadium
    "MIN": 1.00,  # Target Field
    "LAA": 0.99,  # Angel Stadium
    "DET": 0.99,  # Comerica
    "CLE": 0.98,  # Progressive
    "KC": 0.98,   # Kauffman
    "WSH": 0.98,  # Nationals Park
    "CHW": 0.97,  # Guaranteed Rate
    "HOU": 0.97,  # Minute Maid (retractable)
    "STL": 0.97,  # Busch Stadium
    "SD": 0.96,   # Petco
    "PIT": 0.96,  # PNC Park
    "LAD": 0.96,  # Dodger Stadium
    "SF": 0.95,   # Oracle Park
    "NYM": 0.95,  # Citi Field
    "TB": 0.94,   # Tropicana (dome)
    "SEA": 0.93,  # T-Mobile (retractable)
    "MIA": 0.92,  # LoanDepot (retractable)
    "OAK": 0.92,  # Oakland Coliseum
}


class MLBHistoricalDataFetcher:
    """Fetches historical MLB game data from the Stats API."""

    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        self.team_stats_cache: dict[str, dict] = {}
        self.pitcher_stats_cache: dict[int, dict] = {}

    async def close(self):
        await self.client.aclose()

    async def fetch_season_schedule(self, season: int, game_type: str = "R") -> list[dict]:
        """Fetch all games for a season.

        Args:
            season: Year (e.g., 2024)
            game_type: R=Regular, P=Postseason, S=Spring Training
        """
        url = f"{MLB_API_BASE}/schedule"
        params = {
            "sportId": 1,
            "season": season,
            "gameType": game_type,
            "hydrate": "team,linescore,decisions,probablePitcher",
        }

        all_games = []

        # Fetch by date range to avoid timeout
        start_date = datetime(season, 3, 20)  # Opening day area
        end_date = datetime(season, 10, 5)    # End of regular season

        current = start_date
        while current <= end_date:
            chunk_end = min(current + timedelta(days=30), end_date)
            params["startDate"] = current.strftime("%Y-%m-%d")
            params["endDate"] = chunk_end.strftime("%Y-%m-%d")

            try:
                resp = await self.client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()

                for date_entry in data.get("dates", []):
                    for game in date_entry.get("games", []):
                        if game.get("status", {}).get("abstractGameState") == "Final":
                            all_games.append(game)

                logger.info(f"Fetched {current.strftime('%Y-%m')} - {len(all_games)} total games")
            except Exception as e:
                logger.error(f"Error fetching {current}: {e}")

            current = chunk_end + timedelta(days=1)
            await asyncio.sleep(0.5)  # Rate limiting

        return all_games

    async def fetch_team_season_stats(self, team_id: int, season: int) -> dict:
        """Fetch team season statistics."""
        cache_key = f"{team_id}_{season}"
        if cache_key in self.team_stats_cache:
            return self.team_stats_cache[cache_key]

        url = f"{MLB_API_BASE}/teams/{team_id}/stats"
        params = {
            "stats": "season",
            "season": season,
            "group": "hitting,pitching",
        }

        try:
            resp = await self.client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

            stats = {}
            for stat_group in data.get("stats", []):
                group_name = stat_group.get("group", {}).get("displayName", "")
                splits = stat_group.get("splits", [])
                if splits:
                    stats[group_name.lower()] = splits[0].get("stat", {})

            self.team_stats_cache[cache_key] = stats
            return stats
        except Exception as e:
            logger.error(f"Error fetching team {team_id} stats: {e}")
            return {}

    async def fetch_pitcher_season_stats(self, pitcher_id: int, season: int) -> dict:
        """Fetch pitcher season statistics."""
        cache_key = f"{pitcher_id}_{season}"
        if cache_key in self.pitcher_stats_cache:
            return self.pitcher_stats_cache[cache_key]

        url = f"{MLB_API_BASE}/people/{pitcher_id}/stats"
        params = {
            "stats": "season",
            "season": season,
            "group": "pitching",
        }

        try:
            resp = await self.client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

            stats = {}
            for stat_group in data.get("stats", []):
                splits = stat_group.get("splits", [])
                if splits:
                    stats = splits[0].get("stat", {})
                    break

            self.pitcher_stats_cache[cache_key] = stats
            return stats
        except Exception as e:
            logger.error(f"Error fetching pitcher {pitcher_id} stats: {e}")
            return {}

    def extract_game_features(self, game: dict, home_stats: dict, away_stats: dict,
                              home_pitcher_stats: dict, away_pitcher_stats: dict) -> dict:
        """Extract features from a game for model training."""
        home_team = game.get("teams", {}).get("home", {})
        away_team = game.get("teams", {}).get("away", {})

        home_abbr = home_team.get("team", {}).get("abbreviation", "UNK")
        away_abbr = away_team.get("team", {}).get("abbreviation", "UNK")

        # Scores
        home_score = home_team.get("score", 0)
        away_score = away_team.get("score", 0)
        run_diff = home_score - away_score
        total_runs = home_score + away_score

        # Team hitting stats
        home_hitting = home_stats.get("hitting", {})
        away_hitting = away_stats.get("hitting", {})

        # Team pitching stats
        home_pitching = home_stats.get("pitching", {})
        away_pitching = away_stats.get("pitching", {})

        # Helper to safely get float
        def safe_float(d, key, default=0.0):
            try:
                val = d.get(key, default)
                return float(val) if val else default
            except (ValueError, TypeError):
                return default

        features = {
            # Target
            "run_diff": run_diff,
            "total_runs": total_runs,
            "home_win": 1 if run_diff > 0 else 0,

            # Team offense
            "home_runs_per_game": safe_float(home_hitting, "runs") / max(safe_float(home_hitting, "gamesPlayed", 1), 1),
            "away_runs_per_game": safe_float(away_hitting, "runs") / max(safe_float(away_hitting, "gamesPlayed", 1), 1),
            "home_ops": safe_float(home_hitting, "ops"),
            "away_ops": safe_float(away_hitting, "ops"),
            "home_avg": safe_float(home_hitting, "avg"),
            "away_avg": safe_float(away_hitting, "avg"),
            "home_obp": safe_float(home_hitting, "obp"),
            "away_obp": safe_float(away_hitting, "obp"),
            "home_slg": safe_float(home_hitting, "slg"),
            "away_slg": safe_float(away_hitting, "slg"),

            # Team pitching
            "home_era": safe_float(home_pitching, "era", 4.50),
            "away_era": safe_float(away_pitching, "era", 4.50),
            "home_whip": safe_float(home_pitching, "whip", 1.30),
            "away_whip": safe_float(away_pitching, "whip", 1.30),

            # Starter stats
            "home_starter_era": safe_float(home_pitcher_stats, "era", 4.50),
            "away_starter_era": safe_float(away_pitcher_stats, "era", 4.50),
            "home_starter_whip": safe_float(home_pitcher_stats, "whip", 1.30),
            "away_starter_whip": safe_float(away_pitcher_stats, "whip", 1.30),
            "home_starter_k9": safe_float(home_pitcher_stats, "strikeOutsPer9Inn", 8.0),
            "away_starter_k9": safe_float(away_pitcher_stats, "strikeOutsPer9Inn", 8.0),
            "home_starter_bb9": safe_float(home_pitcher_stats, "walksPer9Inn", 3.0),
            "away_starter_bb9": safe_float(away_pitcher_stats, "walksPer9Inn", 3.0),
            "home_starter_ip": safe_float(home_pitcher_stats, "inningsPitched", 0),
            "away_starter_ip": safe_float(away_pitcher_stats, "inningsPitched", 0),

            # Park factor
            "park_factor": PARK_FACTORS.get(home_abbr, 1.0),

            # Derived features
            "offense_diff": safe_float(home_hitting, "ops") - safe_float(away_hitting, "ops"),
            "starter_era_diff": safe_float(away_pitcher_stats, "era", 4.5) - safe_float(home_pitcher_stats, "era", 4.5),
            "team_era_diff": safe_float(away_pitching, "era", 4.5) - safe_float(home_pitching, "era", 4.5),
        }

        return features


class MLBModelTrainer:
    """Trains LightGBM models for MLB predictions."""

    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.fetcher = MLBHistoricalDataFetcher()

    async def collect_training_data(self, seasons: list[int]) -> pd.DataFrame:
        """Collect training data from multiple seasons."""
        all_features = []

        for season in seasons:
            logger.info(f"Fetching {season} season data...")
            games = await self.fetcher.fetch_season_schedule(season)
            logger.info(f"Found {len(games)} completed games for {season}")

            for i, game in enumerate(games):
                try:
                    # Get team IDs
                    home_team = game.get("teams", {}).get("home", {})
                    away_team = game.get("teams", {}).get("away", {})
                    home_team_id = home_team.get("team", {}).get("id")
                    away_team_id = away_team.get("team", {}).get("id")

                    if not home_team_id or not away_team_id:
                        continue

                    # Get probable pitchers
                    home_pitcher = home_team.get("probablePitcher", {})
                    away_pitcher = away_team.get("probablePitcher", {})
                    home_pitcher_id = home_pitcher.get("id")
                    away_pitcher_id = away_pitcher.get("id")

                    # Fetch stats
                    home_stats = await self.fetcher.fetch_team_season_stats(home_team_id, season)
                    away_stats = await self.fetcher.fetch_team_season_stats(away_team_id, season)

                    home_pitcher_stats = {}
                    away_pitcher_stats = {}
                    if home_pitcher_id:
                        home_pitcher_stats = await self.fetcher.fetch_pitcher_season_stats(home_pitcher_id, season)
                    if away_pitcher_id:
                        away_pitcher_stats = await self.fetcher.fetch_pitcher_season_stats(away_pitcher_id, season)

                    # Extract features
                    features = self.fetcher.extract_game_features(
                        game, home_stats, away_stats,
                        home_pitcher_stats, away_pitcher_stats
                    )
                    features["season"] = season
                    features["game_id"] = game.get("gamePk")
                    all_features.append(features)

                    if (i + 1) % 100 == 0:
                        logger.info(f"  Processed {i + 1}/{len(games)} games")

                    # Small delay for rate limiting
                    if (i + 1) % 50 == 0:
                        await asyncio.sleep(0.5)

                except Exception as e:
                    logger.error(f"Error processing game: {e}")
                    continue

        await self.fetcher.close()

        df = pd.DataFrame(all_features)
        logger.info(f"Collected {len(df)} games with features")
        return df

    def train_run_diff_model(self, df: pd.DataFrame) -> tuple:
        """Train the run differential prediction model."""
        if not HAS_LIGHTGBM:
            raise ImportError("LightGBM is required for model training")

        # Feature columns (exclude targets and metadata)
        exclude_cols = ["run_diff", "total_runs", "home_win", "season", "game_id"]
        feature_cols = [c for c in df.columns if c not in exclude_cols]

        X = df[feature_cols].fillna(0)
        y = df["run_diff"]

        # Train/test split by time (use last 20% as test)
        split_idx = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        logger.info(f"Training set: {len(X_train)} games")
        logger.info(f"Test set: {len(X_test)} games")

        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

        # Model parameters
        params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_child_samples": 20,
            "verbose": -1,
            "seed": 42,
        }

        # Train
        model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[train_data, test_data],
            valid_names=["train", "test"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=100),
            ],
        )

        # Evaluate
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        logger.info(f"Run Diff Model - RMSE: {rmse:.3f}, MAE: {mae:.3f}")

        # Feature importance
        importance = pd.DataFrame({
            "feature": feature_cols,
            "importance": model.feature_importance(),
        }).sort_values("importance", ascending=False)
        logger.info(f"Top features:\n{importance.head(10)}")

        return model, feature_cols, {"rmse": rmse, "mae": mae}

    def train_totals_model(self, df: pd.DataFrame) -> tuple:
        """Train the total runs prediction model."""
        if not HAS_LIGHTGBM:
            raise ImportError("LightGBM is required for model training")

        # Feature columns
        exclude_cols = ["run_diff", "total_runs", "home_win", "season", "game_id"]
        feature_cols = [c for c in df.columns if c not in exclude_cols]

        X = df[feature_cols].fillna(0)
        y = df["total_runs"]

        # Train/test split
        split_idx = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

        # Model parameters
        params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_child_samples": 20,
            "verbose": -1,
            "seed": 42,
        }

        # Train
        model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[train_data, test_data],
            valid_names=["train", "test"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=100),
            ],
        )

        # Evaluate
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        logger.info(f"Totals Model - RMSE: {rmse:.3f}, MAE: {mae:.3f}")

        return model, feature_cols, {"rmse": rmse, "mae": mae}

    def save_models(self, run_diff_model, totals_model, feature_cols: list[str], metrics: dict):
        """Save trained models to disk."""
        if not HAS_JOBLIB:
            raise ImportError("joblib is required for saving models")

        # Save run diff model
        run_diff_path = self.model_dir / "mlb_run_diff_v1.joblib"
        joblib.dump({
            "model": run_diff_model,
            "feature_cols": feature_cols,
            "metrics": metrics.get("run_diff", {}),
            "trained_at": datetime.now().isoformat(),
            "version": "1.0",
        }, run_diff_path)
        logger.info(f"Saved run diff model to {run_diff_path}")

        # Save totals model
        totals_path = self.model_dir / "mlb_totals_v1.joblib"
        joblib.dump({
            "model": totals_model,
            "feature_cols": feature_cols,
            "metrics": metrics.get("totals", {}),
            "trained_at": datetime.now().isoformat(),
            "version": "1.0",
        }, totals_path)
        logger.info(f"Saved totals model to {totals_path}")


async def train_mlb_models(seasons: list[int] = None, save_data: bool = True):
    """Main function to train MLB models."""
    if seasons is None:
        seasons = [2024, 2025]  # Default to recent seasons

    trainer = MLBModelTrainer(model_dir="models")

    # Check if we have cached data
    data_path = Path("models/mlb_training_data.parquet")

    if data_path.exists():
        logger.info(f"Loading cached training data from {data_path}")
        df = pd.read_parquet(data_path)
    else:
        logger.info(f"Collecting training data for seasons: {seasons}")
        df = await trainer.collect_training_data(seasons)

        if save_data and len(df) > 0:
            df.to_parquet(data_path)
            logger.info(f"Saved training data to {data_path}")

    if len(df) < 100:
        logger.error("Not enough training data collected")
        return

    logger.info(f"Training on {len(df)} games")

    # Train models
    run_diff_model, feature_cols, run_diff_metrics = trainer.train_run_diff_model(df)
    totals_model, _, totals_metrics = trainer.train_totals_model(df)

    # Save models
    trainer.save_models(
        run_diff_model,
        totals_model,
        feature_cols,
        {"run_diff": run_diff_metrics, "totals": totals_metrics}
    )

    logger.info("Model training complete!")

    return {
        "games_trained": len(df),
        "run_diff_rmse": run_diff_metrics["rmse"],
        "totals_rmse": totals_metrics["rmse"],
    }


if __name__ == "__main__":
    import sys

    # Parse seasons from command line
    if len(sys.argv) > 1:
        seasons = [int(s) for s in sys.argv[1:]]
    else:
        seasons = [2024, 2025]

    print(f"Training MLB models using seasons: {seasons}")
    result = asyncio.run(train_mlb_models(seasons))

    if result:
        print(f"\nTraining Results:")
        print(f"  Games: {result['games_trained']}")
        print(f"  Run Diff RMSE: {result['run_diff_rmse']:.3f}")
        print(f"  Totals RMSE: {result['totals_rmse']:.3f}")
