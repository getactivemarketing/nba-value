"""SQLAlchemy database models."""

from src.models.game import Game
from src.models.team import Team
from src.models.team_stats import TeamStats
from src.models.market import Market
from src.models.odds_snapshot import OddsSnapshot
from src.models.prediction import ModelPrediction
from src.models.score import ValueScore
from src.models.evaluation import Evaluation
from src.models.user import User
from src.models.game_result import GameResult
from src.models.prediction_snapshot import PredictionSnapshot
from src.models.player_prop import PlayerProp
from src.models.prop_snapshot import PropSnapshot

# MLB Models
from src.models.mlb_team import MLBTeam
from src.models.mlb_pitcher import MLBPitcher
from src.models.mlb_pitcher_stats import MLBPitcherStats
from src.models.mlb_team_stats import MLBTeamStats
from src.models.mlb_game import MLBGame
from src.models.mlb_game_context import MLBGameContext
from src.models.mlb_market import MLBMarket
from src.models.mlb_prediction import MLBPrediction
from src.models.mlb_prediction_snapshot import MLBPredictionSnapshot

__all__ = [
    # NBA Models
    "Game",
    "Team",
    "TeamStats",
    "Market",
    "OddsSnapshot",
    "ModelPrediction",
    "ValueScore",
    "Evaluation",
    "User",
    "GameResult",
    "PredictionSnapshot",
    "PlayerProp",
    "PropSnapshot",
    # MLB Models
    "MLBTeam",
    "MLBPitcher",
    "MLBPitcherStats",
    "MLBTeamStats",
    "MLBGame",
    "MLBGameContext",
    "MLBMarket",
    "MLBPrediction",
    "MLBPredictionSnapshot",
]
