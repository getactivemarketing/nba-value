"""Build the leakage-free NFL modeling feature matrix.

Features come from nfl_team_stats at through_week = game.week - 1 (point-in-time);
targets (margin, total) from nfl_games scores; betting lines from nflverse
schedules. Week-1 games have no through_week=0 stats row and are excluded.
"""
import pandas as pd
import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models import NFLGame, NFLTeamStats, NFLGameContext
from src.services.nfl.nfl_data import load_schedules

logger = structlog.get_logger()

# Explicit map of source team-stats column -> output diff-feature name.
# Kept explicit (not derived via string munging) so the mapping is easy to
# audit against MOV_FEATURES / TOTALS_FEATURES below.
_DIFF_MAP = {
    "off_epa_play": "off_epa_diff",
    "def_epa_play": "def_epa_diff",
    "pass_epa": "pass_epa_diff",
    "rush_epa": "rush_epa_diff",
    "success_rate": "success_rate_diff",
    "pace": "pace_diff",
    "power_rating": "power_diff",
}


def build_feature_frame(
    games: pd.DataFrame, team_stats: pd.DataFrame,
    context: pd.DataFrame, lines: pd.DataFrame,
) -> pd.DataFrame:
    ts = team_stats.set_index(["team", "season", "through_week"])
    ctx = context.set_index("game_id")
    ln = lines.set_index("game_id")
    rows = []
    for _, g in games.iterrows():
        w = int(g["week"])
        if w < 2:
            continue  # no through_week = w-1 row exists
        key_h = (g["home_team"], int(g["season"]), w - 1)
        key_a = (g["away_team"], int(g["season"]), w - 1)
        if key_h not in ts.index or key_a not in ts.index:
            continue
        h, a = ts.loc[key_h], ts.loc[key_a]
        c = ctx.loc[g["game_id"]] if g["game_id"] in ctx.index else None
        li = ln.loc[g["game_id"]] if g["game_id"] in ln.index else None
        if li is None or pd.isna(li["spread_line"]) or pd.isna(li["total_line"]):
            continue  # no line -> not gradable
        row = {
            "game_id": g["game_id"], "season": int(g["season"]), "week": w,
            "home_team": g["home_team"], "away_team": g["away_team"],
            "is_divisional": int(bool(g["is_divisional"])),
            "is_primetime": int(bool(g["is_primetime"])),
            "margin": int(g["home_score"]) - int(g["away_score"]),
            "total": int(g["home_score"]) + int(g["away_score"]),
            "spread_line": float(li["spread_line"]),
            "total_line": float(li["total_line"]),
            "home_moneyline": None if pd.isna(li["home_moneyline"]) else float(li["home_moneyline"]),
            "away_moneyline": None if pd.isna(li["away_moneyline"]) else float(li["away_moneyline"]),
            "pace_sum": float(h["pace"]) + float(a["pace"]),
            "off_epa_sum": float(h["off_epa_play"]) + float(a["off_epa_play"]),
            "rest_diff": (0 if c is None or pd.isna(c["home_rest_days"]) else int(c["home_rest_days"]))
                         - (0 if c is None or pd.isna(c["away_rest_days"]) else int(c["away_rest_days"])),
            "is_dome": 0 if c is None else int(bool(c["is_dome"])),
            "wind_mph": None if c is None or pd.isna(c["wind_mph"]) else float(c["wind_mph"]),
            "temp_f": None if c is None or pd.isna(c["temp_f"]) else float(c["temp_f"]),
        }
        for col, name in _DIFF_MAP.items():
            row[name] = float(h[col]) - float(a[col])
        rows.append(row)
    return pd.DataFrame(rows)


async def load_training_frames(session: AsyncSession, seasons: list[int]):
    games = pd.DataFrame([r._mapping for r in (await session.execute(
        select(NFLGame.game_id, NFLGame.season, NFLGame.week, NFLGame.home_team,
               NFLGame.away_team, NFLGame.home_score, NFLGame.away_score,
               NFLGame.is_divisional, NFLGame.is_primetime)
        .where(NFLGame.season.in_(seasons), NFLGame.home_score.isnot(None)))).all()])
    team_stats = pd.DataFrame([dict(
        team=r.team, season=r.season, through_week=r.through_week,
        off_epa_play=r.off_epa_play, def_epa_play=r.def_epa_play, pass_epa=r.pass_epa,
        rush_epa=r.rush_epa, success_rate=r.success_rate, pace=r.pace,
        power_rating=r.power_rating)
        for r in (await session.execute(
            select(NFLTeamStats).where(NFLTeamStats.season.in_(seasons)))).scalars().all()])
    context = pd.DataFrame([dict(
        game_id=r.game_id, home_rest_days=r.home_rest_days, away_rest_days=r.away_rest_days,
        is_dome=r.is_dome, wind_mph=r.wind_mph, temp_f=r.temp_f)
        for r in (await session.execute(select(NFLGameContext))).scalars().all()])
    sched = load_schedules(seasons)
    lines = sched[["game_id", "spread_line", "total_line", "home_moneyline", "away_moneyline"]].copy()
    return games, team_stats, context, lines


# Feature columns each model consumes (kept here as the single source of truth).
# spread_line (market closing line, home favored by N pts) is a known-pre-game
# market anchor: the model learns to adjust FROM the line rather than override it.
# The v1 MOV model without it lost to the market (line RMSE 12.8 < model 13.8);
# this mirrors how the totals model uses total_line and how the MLB/NBA scorers
# regress toward the market.
MOV_FEATURES = ["off_epa_diff", "def_epa_diff", "pass_epa_diff", "rush_epa_diff",
                "success_rate_diff", "pace_diff", "power_diff", "rest_diff",
                "is_divisional", "is_primetime", "spread_line"]
TOTALS_FEATURES = ["off_epa_sum", "pace_sum", "pass_epa_diff", "is_dome",
                   "wind_mph", "temp_f", "total_line"]
