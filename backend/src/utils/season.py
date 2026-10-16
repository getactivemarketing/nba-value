"""NBA season label = the calendar year the season starts in (2026-27 -> 2026).

BallDontLie uses the same convention for `seasons[]`. Oct 1 is the rollover:
preseason begins early October and nothing in Jul-Sep belongs to a new season.
"""
from datetime import date


def current_nba_season(today: date | None = None) -> int:
    d = today or date.today()
    return d.year if d.month >= 10 else d.year - 1


def nba_season_start(season: int) -> date:
    return date(season, 10, 1)
