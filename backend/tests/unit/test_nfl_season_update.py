from src.services.nfl.season_update import match_event_to_game


def test_match_event_to_game_by_teams_and_date():
    games = [{"game_id": "2026_02_CIN_KC", "home_team": "KC", "away_team": "CIN",
              "kickoff_date": "2026-09-13"}]
    ev = {"home_team_abbr": "KC", "away_team_abbr": "CIN", "commence_date": "2026-09-13"}
    assert match_event_to_game(ev, games) == "2026_02_CIN_KC"
    ev2 = {"home_team_abbr": "KC", "away_team_abbr": "BUF", "commence_date": "2026-09-13"}
    assert match_event_to_game(ev2, games) is None


def test_match_event_to_game_same_teams_different_date_returns_none():
    # Two teams can play twice in a season (e.g. divisional rematch) — date
    # must be part of the match key, not just teams.
    games = [{"game_id": "2026_02_CIN_KC", "home_team": "KC", "away_team": "CIN",
              "kickoff_date": "2026-09-13"}]
    ev = {"home_team_abbr": "KC", "away_team_abbr": "CIN", "commence_date": "2026-12-20"}
    assert match_event_to_game(ev, games) is None
