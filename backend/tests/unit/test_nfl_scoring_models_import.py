def test_nfl_scoring_models_named_and_columns():
    from src.models import NFLMarket, NFLPredictionSnapshot
    assert NFLMarket.__tablename__ == "nfl_markets"
    assert NFLPredictionSnapshot.__tablename__ == "nfl_prediction_snapshots"
    mcols = set(NFLMarket.__table__.columns.keys())
    assert {"game_id", "market_type", "line", "home_odds", "away_odds",
            "over_odds", "under_odds"}.issubset(mcols)
    scols = set(NFLPredictionSnapshot.__table__.columns.keys())
    assert {"best_spread_team", "best_ml_team", "best_total_direction",
            "best_bet_type", "best_bet_profit", "actual_margin", "actual_total"}.issubset(scols)


def test_snapshot_game_id_unique():
    from src.models import NFLPredictionSnapshot
    assert NFLPredictionSnapshot.__table__.columns["game_id"].unique
