from datetime import date

from src.utils.season import current_nba_season, nba_season_start


def test_october_starts_the_new_season():
    assert current_nba_season(date(2026, 10, 1)) == 2026
    assert current_nba_season(date(2026, 10, 20)) == 2026


def test_spring_and_summer_belong_to_the_previous_season():
    assert current_nba_season(date(2027, 4, 15)) == 2026
    assert current_nba_season(date(2026, 9, 30)) == 2025


def test_season_start_is_october_first():
    assert nba_season_start(2026) == date(2026, 10, 1)
