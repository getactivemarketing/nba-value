def test_nfl_models_importable_and_named():
    from src.models import NFLTeam, NFLGame, NFLGameContext, NFLTeamStats
    assert NFLTeam.__tablename__ == "nfl_teams"
    assert NFLGame.__tablename__ == "nfl_games"
    assert NFLGameContext.__tablename__ == "nfl_game_context"
    assert NFLTeamStats.__tablename__ == "nfl_team_stats"


def test_nfl_game_has_situational_flag_columns():
    from src.models import NFLGame
    cols = set(NFLGame.__table__.columns.keys())
    assert {"is_divisional", "is_primetime", "primetime_slot",
            "home_qb", "away_qb"}.issubset(cols)


def test_nfl_team_stats_unique_constraint():
    from src.models import NFLTeamStats
    uniques = {tuple(c.name for c in con.columns)
               for con in NFLTeamStats.__table__.constraints
               if con.__class__.__name__ == "UniqueConstraint"}
    assert ("team", "season", "through_week") in uniques
