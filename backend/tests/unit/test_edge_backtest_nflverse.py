from datetime import datetime, timezone

import pandas as pd

from src.edge.backtest import SEASONS, nflverse


def test_seasons():
    assert SEASONS == (2023, 2024, 2025)


def test_schedule_games_converts_et_to_utc_maps_teams_and_skips_playoffs():
    frame = pd.DataFrame([
        {"game_id": "2023_10_CLE_BAL", "season": 2023, "game_type": "REG", "week": 10, "gameday": "2023-11-12",
         "gametime": "13:00", "home_team": "BAL", "away_team": "CLE", "home_score": 31.0, "away_score": 33.0},
        {"game_id": "2023_10_WAS_LA", "season": 2023, "game_type": "REG", "week": 10, "gameday": "2023-11-13",
         "gametime": "20:15", "home_team": "LA", "away_team": "WAS", "home_score": float("nan"),
         "away_score": float("nan")},
        {"game_id": "2023_19_X_Y", "season": 2023, "game_type": "WC", "week": 19, "gameday": "2024-01-13",
         "gametime": "16:30", "home_team": "KC", "away_team": "MIA", "home_score": 26.0, "away_score": 7.0},
    ])
    games = nflverse.schedule_games(frame)
    assert [g.game_id for g in games] == ["2023_10_CLE_BAL", "2023_10_WAS_LA"]
    a, b = games
    assert a.kickoff == datetime(2023, 11, 12, 18, 0, tzinfo=timezone.utc)
    assert (a.home, a.away, a.home_score, a.away_score) == ("BAL", "CLE", 31, 33)
    assert b.kickoff == datetime(2023, 11, 14, 1, 15, tzinfo=timezone.utc)  # Monday 20:15 ET
    assert (b.home, b.away, b.home_score, b.away_score) == ("LAR", "WSH", None, None)


def test_stats_index_maps_columns_normalizes_names_and_keeps_regular_season():
    frame = pd.DataFrame([
        {"game_id": "g1", "season_type": "REG", "player_display_name": "Ja'Marr Chase", "team": "CIN",
         "attempts": 0, "completions": 0, "passing_yards": 0, "passing_tds": 0, "passing_interceptions": 0,
         "carries": 1, "rushing_yards": 4.0, "receptions": 7, "receiving_yards": 81.0},
        {"game_id": "g1", "season_type": "REG", "player_display_name": "Kenneth Walker III", "team": "SEA",
         "attempts": 0, "completions": 0, "passing_yards": 0, "passing_tds": 0, "passing_interceptions": 0,
         "carries": float("nan"), "rushing_yards": 55, "receptions": 1, "receiving_yards": 6},
        {"game_id": "g2", "season_type": "POST", "player_display_name": "Joe Burrow", "team": "CIN",
         "attempts": 30, "completions": 20, "passing_yards": 250, "passing_tds": 2, "passing_interceptions": 1,
         "carries": 2, "rushing_yards": 5, "receptions": 0, "receiving_yards": 0},
    ])
    idx = nflverse.stats_index(frame)
    assert set(idx) == {"g1"}
    chase, walker = idx["g1"]
    assert chase == {"name": "jamarr chase", "team": "CIN", "pass_att": 0, "pass_cmp": 0, "pass_yds": 0,
                     "pass_tds": 0, "pass_int": 0, "rush_att": 1, "rush_yds": 4, "receptions": 7, "rec_yds": 81}
    assert walker["name"] == "kenneth walker" and walker["rush_att"] == 0 and walker["rush_yds"] == 55


def test_snaps_index_keeps_players_with_snaps_only():
    frame = pd.DataFrame([
        {"game_id": "g1", "game_type": "REG", "player": "D.J. Moore", "offense_snaps": 60.0,
         "defense_snaps": 0.0, "st_snaps": 0.0},
        {"game_id": "g1", "game_type": "REG", "player": "Bench Guy", "offense_snaps": 0.0,
         "defense_snaps": 0.0, "st_snaps": 0.0},
        {"game_id": "g9", "game_type": "POST", "player": "X", "offense_snaps": 10.0,
         "defense_snaps": 0.0, "st_snaps": 0.0},
    ])
    assert nflverse.snaps_index(frame) == {"g1": {"dj moore"}}


def test_stats_index_skips_nameless_team_rows():
    # The release file carries ~22 team-level rows per season with no player (seen 2026-10-16).
    frame = pd.DataFrame([
        {"game_id": "g1", "season_type": "REG", "player_display_name": None, "team": "DET",
         "attempts": 0, "completions": 0, "passing_yards": 0, "passing_tds": 0, "passing_interceptions": 0,
         "carries": 0, "rushing_yards": 0, "receptions": 0, "receiving_yards": 0},
    ])
    assert nflverse.stats_index(frame) == {}
