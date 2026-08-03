"""Build a multi-season, point-in-time MLB training set — directly from the API.

The 2026-only set (1,405 games) could not validate 33 features. The fix is more
seasons, and prior-season history IS fetchable: gameLog works for 2024/2025 and
the schedule endpoint hydrates starters. My earlier "wait until April" framing
was wrong.

Built entirely OFFLINE — nothing is written to production tables. Ingesting
~4,900 historical games into mlb_games would also add ~600 pitchers to
mlb_pitchers, tripling the runtime of the daily pitcher-stats job for data no
live code path needs.

Every feature is computed as of the day BEFORE each game, from that team's and
pitcher's prior appearances only.

    python -m src.tasks.build_mlb_training_history --seasons 2024,2025 \
        --out models/mlb_training_history.parquet
"""

import asyncio
import sys
from collections import defaultdict
from datetime import date, timedelta

import pandas as pd

from src.services.mlb.bullpen import (
    BullpenLine, bullpen_from_team_and_starter, cumulative_bullpen, fip, recent_workload,
)
from src.services.mlb.mlb_api import (
    DOME_VENUES, MLB_TEAM_ABBR, PARK_FACTORS, MLBStatsAPIClient,
)
from src.services.mlb.pitcher_history import cumulative_through, parse_innings
from src.services.mlb.team_history import cumulative_team_stats

# Weather is not reconstructible for past seasons, so `temperature` and
# `weather_factor` are omitted rather than defaulted across two-thirds of the
# rows — a constant column teaches the model nothing and dilutes the rest.
# `is_dome` is kept because it derives from the venue. In the 2026 retrain
# temperature scored exactly zero gain, so little is lost.
FEATURES = [
    "home_runs_per_game", "away_runs_per_game",
    "home_ops", "away_ops", "home_avg", "away_avg",
    "home_obp", "away_obp", "home_slg", "away_slg",
    "home_era", "away_era", "home_whip", "away_whip",
    "home_starter_era", "away_starter_era",
    "home_starter_whip", "away_starter_whip",
    "home_starter_k9", "away_starter_k9",
    "home_starter_bb9", "away_starter_bb9",
    "home_starter_ip", "away_starter_ip",
    "park_factor", "offense_diff", "starter_era_diff", "team_era_diff",
    "is_dome",
    "home_last_10_win_pct", "away_last_10_win_pct",
    # New (items 7 + 8)
    "home_starter_fip", "away_starter_fip", "starter_fip_diff",
    "home_bullpen_era", "away_bullpen_era",
    "home_bullpen_whip", "away_bullpen_whip",
    "home_bullpen_ip_l3", "away_bullpen_ip_l3",
]


async def _season_games(client: MLBStatsAPIClient, season: int) -> list:
    """Every completed regular-season game, in fortnight chunks."""
    out, start = [], date(season, 3, 1)
    end = date(season, 11, 15)
    while start < end:
        stop = min(start + timedelta(days=14), end)
        games = await client.get_schedule(start, stop, game_types=["R"])
        out.extend(
            g for g in games
            if g.status == "final" and g.home_score is not None
            and g.home_starter_id and g.away_starter_id
        )
        start = stop + timedelta(days=1)
    return out


def _last10(results: list[bool], as_of: str) -> float | None:
    prior = [w for d, w in results if d < as_of][-10:]
    return round(sum(prior) / len(prior), 4) if prior else None


async def build(seasons: list[int], out_path: str) -> dict:
    client = MLBStatsAPIClient()
    rows: list[dict] = []

    for season in seasons:
        print(f"[HIST] {season}: fetching schedule...", flush=True)
        games = await _season_games(client, season)
        print(f"[HIST] {season}: {len(games)} completed games", flush=True)

        # Team game logs, once per team.
        team_logs: dict[str, tuple] = {}
        for tid, abbr in MLB_TEAM_ABBR.items():
            team_logs[abbr] = await client.get_team_game_logs(tid, season)
        print(f"[HIST] {season}: team logs done", flush=True)

        # Pitcher logs, once per starter seen.
        starters = {g.home_starter_id for g in games} | {g.away_starter_id for g in games}
        starters.discard(None)
        pitcher_logs: dict[int, list] = {}
        for i, pid in enumerate(sorted(starters), 1):
            try:
                pitcher_logs[pid] = await client.get_pitcher_game_logs(int(pid), season)
            except Exception:
                pitcher_logs[pid] = []
            if i % 100 == 0:
                print(f"[HIST] {season}: {i}/{len(starters)} pitchers", flush=True)
        print(f"[HIST] {season}: {len(starters)} pitcher logs done", flush=True)

        # Per-team bullpen lines, derived as team pitching minus that game's starter.
        by_date_team_pitch = {
            abbr: {g.date: g for g in logs[1]} for abbr, logs in team_logs.items()
        }
        bullpen: dict[str, list] = defaultdict(list)
        win_hist: dict[str, list] = defaultdict(list)

        for g in sorted(games, key=lambda x: x.game_date):
            iso = g.game_date.isoformat()
            home_won = g.home_score > g.away_score
            win_hist[g.home_team].append((iso, home_won))
            win_hist[g.away_team].append((iso, not home_won))

            for team, sid in ((g.home_team, g.home_starter_id), (g.away_team, g.away_starter_id)):
                tp = by_date_team_pitch.get(team, {}).get(iso)
                sp = next((l for l in pitcher_logs.get(sid, []) if l.date == iso), None)
                if not tp or not sp:
                    continue
                line = bullpen_from_team_and_starter(
                    team_ip=parse_innings(tp.innings_pitched), team_er=tp.earned_runs,
                    team_h=tp.hits_allowed, team_bb=tp.walks_allowed,
                    starter_ip=parse_innings(sp.innings_pitched), starter_er=sp.earned_runs,
                    starter_h=sp.hits, starter_bb=sp.walks,
                )
                if line:
                    bullpen[team].append((iso, line))

        # Emit one row per game, everything as of the day before.
        for g in games:
            iso = g.game_date.isoformat()
            row: dict = {"game_id": g.game_id, "game_date": iso, "season": season}

            ok = True
            for side, team, sid in (("home", g.home_team, g.home_starter_id),
                                    ("away", g.away_team, g.away_starter_id)):
                hit, pit = team_logs.get(team, ([], []))
                ts = cumulative_team_stats(hit, pit, iso)
                sp_stats = cumulative_through(pitcher_logs.get(sid, []), iso)
                if ts is None or sp_stats is None or ts["ops"] is None or ts["era"] is None:
                    ok = False
                    break

                bp = cumulative_bullpen(bullpen.get(team, []), iso)
                row[f"{side}_ops"] = ts["ops"]
                row[f"{side}_avg"] = ts["batting_avg"]
                row[f"{side}_obp"] = ts["obp"]
                row[f"{side}_slg"] = ts["slg"]
                row[f"{side}_era"] = ts["era"]
                row[f"{side}_whip"] = ts["team_whip"]
                row[f"{side}_starter_era"] = sp_stats["era"]
                row[f"{side}_starter_whip"] = sp_stats["whip"]
                row[f"{side}_starter_k9"] = sp_stats["k_per_9"]
                row[f"{side}_starter_bb9"] = sp_stats["bb_per_9"]
                row[f"{side}_starter_ip"] = sp_stats["innings_pitched"]
                row[f"{side}_starter_fip"] = sp_stats["fip"]
                row[f"{side}_bullpen_era"] = bp["bullpen_era"] if bp else None
                row[f"{side}_bullpen_whip"] = bp["bullpen_whip"] if bp else None
                row[f"{side}_bullpen_ip_l3"] = recent_workload(bullpen.get(team, []), iso, days=3)
                row[f"{side}_last_10_win_pct"] = _last10(win_hist.get(team, []), iso)

                # Runs/game from the team's own prior scoring, not standings.
                scored = [s for d, s in _team_runs(games, team) if d < iso]
                row[f"{side}_runs_per_game"] = round(sum(scored) / len(scored), 3) if scored else None

            if not ok:
                continue

            row["park_factor"] = PARK_FACTORS.get(g.venue_name, 1.0)
            row["is_dome"] = 1.0 if g.venue_name in DOME_VENUES else 0.0
            row["offense_diff"] = (row.get("home_runs_per_game") or 0) - (row.get("away_runs_per_game") or 0)
            row["starter_era_diff"] = row["away_starter_era"] - row["home_starter_era"]
            row["starter_fip_diff"] = row["away_starter_fip"] - row["home_starter_fip"]
            row["team_era_diff"] = row["away_era"] - row["home_era"]
            row["target_run_diff"] = g.home_score - g.away_score
            row["target_total"] = g.home_score + g.away_score
            rows.append(row)

        print(f"[HIST] {season}: {len(rows)} rows so far", flush=True)

    df = pd.DataFrame(rows).dropna(subset=["target_run_diff"])
    df.to_parquet(out_path)

    print(f"\nrows: {len(df)} -> {out_path}\n")
    print(f"{'feature':<26}{'coverage':>10}{'unique':>9}")
    for c in FEATURES:
        if c not in df.columns:
            print(f"{c:<26}{'MISSING':>10}")
            continue
        n = int(df[c].notna().sum())
        print(f"{c:<26}{n/len(df)*100:>9.1f}%{df[c].nunique():>9}")
    return {"rows": len(df)}


_runs_cache: dict = {}


def _team_runs(games, team) -> list:
    """(date, runs scored) for a team, cached per build."""
    key = (id(games), team)
    if key not in _runs_cache:
        out = []
        for g in games:
            if g.home_team == team:
                out.append((g.game_date.isoformat(), g.home_score))
            elif g.away_team == team:
                out.append((g.game_date.isoformat(), g.away_score))
        _runs_cache[key] = sorted(out)
    return _runs_cache[key]


if __name__ == "__main__":
    args = sys.argv[1:]
    seasons, out = [2024, 2025], "models/mlb_training_history.parquet"
    for i, a in enumerate(args):
        if a == "--seasons" and i + 1 < len(args):
            seasons = [int(s) for s in args[i + 1].split(",")]
        elif a == "--out" and i + 1 < len(args):
            out = args[i + 1]
    asyncio.run(build(seasons, out))
