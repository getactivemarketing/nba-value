"""Content generation for TruLine social media posts."""

import structlog
from datetime import datetime, timezone, timedelta, date
from decimal import Decimal

from sqlalchemy import select, and_, desc, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.models import (
    MLBGame, MLBPredictionSnapshot, MLBPrediction, MLBPitcher, MLBGameContext,
    MLBTeamStats, MLBPitcherStats, MLBMarket,
)

logger = structlog.get_logger()

# MLB team hashtags
TEAM_HASHTAGS = {
    "ARI": "#Dbacks", "ATL": "#Braves", "BAL": "#Orioles", "BOS": "#RedSox",
    "CHC": "#Cubs", "CWS": "#WhiteSox", "CIN": "#Reds", "CLE": "#Guardians",
    "COL": "#Rockies", "DET": "#Tigers", "HOU": "#Astros", "KC": "#Royals",
    "LAA": "#Angels", "LAD": "#Dodgers", "MIA": "#Marlins", "MIL": "#Brewers",
    "MIN": "#Twins", "NYM": "#Mets", "NYY": "#Yankees", "OAK": "#Athletics",
    "PHI": "#Phillies", "PIT": "#Pirates", "SD": "#Padres", "SF": "#Giants",
    "SEA": "#Mariners", "STL": "#Cardinals", "TB": "#Rays", "TEX": "#Rangers",
    "TOR": "#BlueJays", "WSH": "#Nationals",
}

# Official MLB team Twitter handles (for tagging)
TEAM_HANDLES = {
    "ARI": "@Dbacks",
    "ATL": "@Braves",
    "BAL": "@Orioles",
    "BOS": "@RedSox",
    "CHC": "@Cubs",
    "CWS": "@whitesox",
    "CIN": "@Reds",
    "CLE": "@CleGuardians",
    "COL": "@Rockies",
    "DET": "@tigers",
    "HOU": "@astros",
    "KC": "@Royals",
    "LAA": "@Angels",
    "LAD": "@Dodgers",
    "MIA": "@Marlins",
    "MIL": "@Brewers",
    "MIN": "@Twins",
    "NYM": "@Mets",
    "NYY": "@Yankees",
    "OAK": "@Athletics",
    "PHI": "@Phillies",
    "PIT": "@Pirates",
    "SD": "@Padres",
    "SF": "@SFGiants",
    "SEA": "@Mariners",
    "STL": "@Cardinals",
    "TB": "@RaysBaseball",
    "TEX": "@Rangers",
    "TOR": "@BlueJays",
    "WSH": "@Nationals",
}

TEAM_NAMES = {
    "ARI": "D-backs", "ATL": "Braves", "BAL": "Orioles", "BOS": "Red Sox",
    "CHC": "Cubs", "CWS": "White Sox", "CIN": "Reds", "CLE": "Guardians",
    "COL": "Rockies", "DET": "Tigers", "HOU": "Astros", "KC": "Royals",
    "LAA": "Angels", "LAD": "Dodgers", "MIA": "Marlins", "MIL": "Brewers",
    "MIN": "Twins", "NYM": "Mets", "NYY": "Yankees", "OAK": "Athletics",
    "PHI": "Phillies", "PIT": "Pirates", "SD": "Padres", "SF": "Giants",
    "SEA": "Mariners", "STL": "Cardinals", "TB": "Rays", "TEX": "Rangers",
    "TOR": "Blue Jays", "WSH": "Nationals",
}


# NBA team metadata
NBA_TEAM_NAMES = {
    "ATL": "Hawks", "BOS": "Celtics", "BKN": "Nets", "CHA": "Hornets",
    "CHI": "Bulls", "CLE": "Cavaliers", "DAL": "Mavericks", "DEN": "Nuggets",
    "DET": "Pistons", "GSW": "Warriors", "HOU": "Rockets", "IND": "Pacers",
    "LAC": "Clippers", "LAL": "Lakers", "MEM": "Grizzlies", "MIA": "Heat",
    "MIL": "Bucks", "MIN": "Timberwolves", "NOP": "Pelicans", "NYK": "Knicks",
    "OKC": "Thunder", "ORL": "Magic", "PHI": "76ers", "PHX": "Suns",
    "POR": "Trail Blazers", "SAC": "Kings", "SAS": "Spurs", "TOR": "Raptors",
    "UTA": "Jazz", "WAS": "Wizards",
}

NBA_TEAM_HANDLES = {
    "ATL": "@ATLHawks", "BOS": "@celtics", "BKN": "@BrooklynNets", "CHA": "@hornets",
    "CHI": "@chicagobulls", "CLE": "@cavs", "DAL": "@dallasmavs", "DEN": "@nuggets",
    "DET": "@DetroitPistons", "GSW": "@warriors", "HOU": "@HoustonRockets", "IND": "@Pacers",
    "LAC": "@LAClippers", "LAL": "@Lakers", "MEM": "@memgrizz", "MIA": "@MiamiHEAT",
    "MIL": "@Bucks", "MIN": "@Timberwolves", "NOP": "@PelicansNBA", "NYK": "@nyknicks",
    "OKC": "@okcthunder", "ORL": "@OrlandoMagic", "PHI": "@sixers", "PHX": "@Suns",
    "POR": "@trailblazers", "SAC": "@SacramentoKings", "SAS": "@spurs", "TOR": "@Raptors",
    "UTA": "@utahjazz", "WAS": "@WashWizards",
}

NBA_TEAM_HASHTAGS = {
    "ATL": "#TrueToAtlanta", "BOS": "#DifferentHere", "BKN": "#NetsWorld",
    "CHA": "#AllFly", "CHI": "#BullsNation", "CLE": "#LetEmKnow",
    "DAL": "#MFFL", "DEN": "#MileHighBasketball", "DET": "#DetroitBasketball",
    "GSW": "#DubNation", "HOU": "#Rockets", "IND": "#BoomBaby",
    "LAC": "#Clippers", "LAL": "#LakeShow", "MEM": "#GrindCity",
    "MIA": "#HEATCulture", "MIL": "#FearTheDeer", "MIN": "#WolvesBack",
    "NOP": "#WontBowDown", "NYK": "#NewYorkForever", "OKC": "#ThunderUp",
    "ORL": "#MagicTogether", "PHI": "#HereTheyCome", "PHX": "#ValleyProud",
    "POR": "#RipCity", "SAC": "#SacramentoProud", "SAS": "#PorVida",
    "TOR": "#WeTheNorth", "UTA": "#TakeNote", "WAS": "#DCFamily",
}


async def get_team_card_stats(session: AsyncSession, team_abbr: str) -> dict:
    """Fetch team stats for social media card images.

    Returns dict with: record, l10, div_rank, ats, ou (all as formatted strings or None).
    """
    from src.models.mlb_team import MLBTeam

    result = {}

    # Get latest team stats
    stat_row = await session.execute(
        select(MLBTeamStats).where(
            MLBTeamStats.team_abbr == team_abbr
        ).order_by(desc(MLBTeamStats.stat_date)).limit(1)
    )
    stats = stat_row.scalar_one_or_none()

    if not stats:
        logger.warning("get_team_card_stats: no MLBTeamStats row", team=team_abbr)
        return result

    if stats.wins is not None and stats.losses is not None:
        result["record"] = f"{stats.wins}-{stats.losses}"
    # L10: prefer the string column, fall back to building from ints
    if stats.last_10_record:
        result["l10"] = stats.last_10_record
    elif stats.last_10_wins is not None and stats.last_10_losses is not None:
        result["l10"] = f"{stats.last_10_wins}-{stats.last_10_losses}"
    # ATS / O-U intentionally hidden — our DB-derived values only reflect games
    # we've graded (1-2 weeks), which doesn't match the season-long records
    # bettors expect. Re-enable once the backfill job runs reliably against
    # all final games + markets from opening day.

    logger.debug("get_team_card_stats: stats found", team=team_abbr, result=result)

    # Get division + compute rank (separate from core stats so a failure here
    # doesn't prevent W-L/L10/ATS/O-U from showing)
    try:
        team_row = await session.execute(
            select(MLBTeam).where(MLBTeam.team_abbr == team_abbr)
        )
        team = team_row.scalar_one_or_none()

        if not team:
            logger.warning("get_team_card_stats: no MLBTeam row — skipping div rank", team=team_abbr)
            return result

        if stats.wins is not None:
            div_teams = await session.execute(
                select(MLBTeam.team_abbr).where(
                    and_(MLBTeam.league == team.league, MLBTeam.division == team.division)
                )
            )
            div_abbrs = [r[0] for r in div_teams.fetchall()]

            div_records = []
            for abbr in div_abbrs:
                s = await session.execute(
                    select(MLBTeamStats.team_abbr, MLBTeamStats.win_pct).where(
                        MLBTeamStats.team_abbr == abbr
                    ).order_by(desc(MLBTeamStats.stat_date)).limit(1)
                )
                row = s.first()
                if row and row[1] is not None:
                    div_records.append((row[0], float(row[1])))

            div_records.sort(key=lambda x: x[1], reverse=True)
            rank = next((i + 1 for i, (a, _) in enumerate(div_records) if a == team_abbr), None)
            if rank:
                suffix = {1: "st", 2: "nd", 3: "rd"}.get(rank, "th")
                result["div_rank"] = f"{rank}{suffix} {team.league} {team.division}"
    except Exception as e:
        logger.warning("get_team_card_stats: div rank lookup failed", team=team_abbr, error=str(e))

    return result


def _fmt_odds(decimal_odds: float) -> str:
    """Format decimal odds to American."""
    if decimal_odds >= 2.0:
        return f"+{round((decimal_odds - 1) * 100)}"
    else:
        return str(round(-100 / (decimal_odds - 1)))


async def _get_pitcher_era(session: AsyncSession, pitcher_id: int | None) -> tuple[str | None, float | None]:
    """Fetch pitcher name and latest ERA. Returns (name, era)."""
    if not pitcher_id:
        return None, None
    p_row = await session.execute(
        select(MLBPitcher).where(MLBPitcher.pitcher_id == pitcher_id)
    )
    pitcher = p_row.scalar_one_or_none()
    if not pitcher:
        return None, None
    s_row = await session.execute(
        select(MLBPitcherStats)
        .where(MLBPitcherStats.pitcher_id == pitcher_id)
        .order_by(desc(MLBPitcherStats.stat_date))
        .limit(1)
    )
    stats = s_row.scalar_one_or_none()
    era = float(stats.era) if stats and stats.era is not None else None
    # Use last name only for brevity
    last_name = pitcher.player_name.split()[-1] if pitcher.player_name else None
    return last_name, era


async def _get_team_first_inning_pct(
    session: AsyncSession, team: str
) -> tuple[float | None, float | None]:
    """Return (offensive_score_pct, defensive_opp_score_pct) for a team.

    Computed directly from mlb_games first-inning runs over final regular
    season games. Returns (None, None) if no games.
    """
    from sqlalchemy import text

    result = await session.execute(
        text("""
            SELECT
                COUNT(*) AS games,
                SUM(CASE WHEN runs_for > 0 THEN 1 ELSE 0 END) AS scored,
                SUM(CASE WHEN runs_against > 0 THEN 1 ELSE 0 END) AS runs_allowed
            FROM (
                SELECT
                    COALESCE(home_first_inning_runs, 0) AS runs_for,
                    COALESCE(away_first_inning_runs, 0) AS runs_against
                FROM mlb_games
                WHERE status = 'final' AND game_type = 'R'
                  AND home_first_inning_runs IS NOT NULL
                  AND home_team = :team
                UNION ALL
                SELECT
                    COALESCE(away_first_inning_runs, 0) AS runs_for,
                    COALESCE(home_first_inning_runs, 0) AS runs_against
                FROM mlb_games
                WHERE status = 'final' AND game_type = 'R'
                  AND away_first_inning_runs IS NOT NULL
                  AND away_team = :team
            ) t
        """),
        {"team": team},
    )
    row = result.fetchone()
    if not row:
        return None, None
    games, scored, runs_allowed = row
    games = int(games or 0)
    if games == 0:
        return None, None
    off_pct = float(int(scored or 0)) / games
    def_pct = float(int(runs_allowed or 0)) / games
    return off_pct, def_pct


async def _get_pitcher_first_inning_record(
    session: AsyncSession, pitcher_id: int | None
) -> tuple[int, int] | None:
    """Return (scoreless_starts, total_starts) for a pitcher's 1st-inning record.

    Checks games where this pitcher was the starter and looks at whether
    the opposing team scored in the 1st inning.
    """
    if not pitcher_id:
        return None

    result = await session.execute(
        text("""
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN opp_first = 0 THEN 1 ELSE 0 END) AS scoreless
            FROM (
                SELECT COALESCE(away_first_inning_runs, 0) AS opp_first
                FROM mlb_games
                WHERE home_starter_id = :pid AND status = 'final'
                  AND home_first_inning_runs IS NOT NULL
                UNION ALL
                SELECT COALESCE(home_first_inning_runs, 0) AS opp_first
                FROM mlb_games
                WHERE away_starter_id = :pid AND status = 'final'
                  AND away_first_inning_runs IS NOT NULL
            ) t
        """),
        {"pid": pitcher_id},
    )
    row = result.fetchone()
    if not row or not row.total or int(row.total) == 0:
        return None
    return (int(row.scoreless or 0), int(row.total))


async def _get_team_streak(session: AsyncSession, team_abbr: str) -> str | None:
    """Return current win/loss streak like 'W4' or 'L2'. None if no data."""
    result = await session.execute(
        text("""
            SELECT
                home_team, away_team, home_score, away_score
            FROM mlb_games
            WHERE (home_team = :team OR away_team = :team)
              AND status = 'final' AND game_type = 'R'
            ORDER BY game_date DESC, game_time DESC
            LIMIT 20
        """),
        {"team": team_abbr},
    )
    rows = result.fetchall()
    if not rows:
        return None

    streak_type = None
    streak_count = 0
    for row in rows:
        if row.home_score is None or row.away_score is None:
            continue
        is_home = row.home_team == team_abbr
        won = (is_home and row.home_score > row.away_score) or \
              (not is_home and row.away_score > row.home_score)
        current = "W" if won else "L"
        if streak_type is None:
            streak_type = current
            streak_count = 1
        elif current == streak_type:
            streak_count += 1
        else:
            break

    if streak_type and streak_count >= 2:
        return f"{streak_type}{streak_count}"
    return None


async def _get_bet_type_summary(
    session: AsyncSession, days: int = 7
) -> dict[str, dict[str, int]]:
    """Return recent bet performance by type. E.g. {'moneyline': {'wins': 3, 'losses': 1}}."""
    cutoff = date.today() - timedelta(days=days)
    result = await session.execute(
        select(MLBPredictionSnapshot).where(
            and_(
                MLBPredictionSnapshot.game_date >= cutoff,
                MLBPredictionSnapshot.best_bet_result.isnot(None),
            )
        )
    )
    snapshots = list(result.scalars().all())

    summary: dict[str, dict[str, int]] = {}
    for s in snapshots:
        bt = (s.best_bet_type or "unknown").lower()
        if bt not in summary:
            summary[bt] = {"wins": 0, "losses": 0, "pushes": 0}
        if s.best_bet_result == "win":
            summary[bt]["wins"] += 1
        elif s.best_bet_result == "loss":
            summary[bt]["losses"] += 1
        else:
            summary[bt]["pushes"] += 1

    return summary


def _implied_prob_from_decimal(decimal_odds: float) -> float:
    """Raw implied probability from decimal odds (not de-vigged)."""
    if not decimal_odds or decimal_odds <= 1.0:
        return 0.0
    return 1.0 / decimal_odds


async def generate_daily_picks_thread(session: AsyncSession, game_date: date) -> list[str]:
    """Generate a Twitter thread with today's best MLB value bets.

    Uses narrative voice with data-backed analysis instead of raw stat dumps.
    Returns a list of tweet-length strings (max 280 chars each).
    """
    tweets = []

    games_result = await session.execute(
        select(MLBGame).where(
            and_(
                MLBGame.game_date == game_date,
                MLBGame.status == "scheduled",
            )
        ).order_by(MLBGame.game_time)
    )
    games = list(games_result.scalars().all())
    if not games:
        return []

    game_ids = [g.game_id for g in games]

    pred_result = await session.execute(
        select(MLBPrediction).where(
            and_(
                MLBPrediction.game_id.in_(game_ids),
                MLBPrediction.market_type == "moneyline",
            )
        )
    )
    pred_map = {p.game_id: p for p in pred_result.scalars().all()}

    mkt_result = await session.execute(
        select(MLBMarket).where(
            and_(
                MLBMarket.game_id.in_(game_ids),
                MLBMarket.market_type == "moneyline",
            )
        ).order_by(desc(MLBMarket.updated_at))
    )
    mkt_map: dict[str, MLBMarket] = {}
    for m in mkt_result.scalars().all():
        if m.game_id not in mkt_map:
            mkt_map[m.game_id] = m

    value_plays = []
    for game in games:
        pred = pred_map.get(game.game_id)
        mkt = mkt_map.get(game.game_id)
        if not pred or not mkt:
            continue
        if pred.p_home_win is None or pred.p_away_win is None:
            continue
        if mkt.home_odds is None or mkt.away_odds is None:
            continue

        p_home = float(pred.p_home_win)
        p_away = float(pred.p_away_win)
        home_odds = float(mkt.home_odds)
        away_odds = float(mkt.away_odds)
        mk_home = _implied_prob_from_decimal(home_odds)
        mk_away = _implied_prob_from_decimal(away_odds)

        home_edge = p_home - mk_home
        away_edge = p_away - mk_away

        if home_edge >= away_edge:
            team = game.home_team
            model_p = p_home
            market_p = mk_home
            odds = home_odds
            edge = home_edge
            is_underdog = home_odds >= 2.0
        else:
            team = game.away_team
            model_p = p_away
            market_p = mk_away
            odds = away_odds
            edge = away_edge
            is_underdog = away_odds >= 2.0

        if edge <= 0.05:
            continue
        confidence = (pred.confidence or "").lower()
        if confidence and confidence not in ("high", "medium"):
            continue

        value_plays.append({
            "game": game,
            "team": team,
            "model_p": model_p,
            "market_p": market_p,
            "odds": odds,
            "edge": edge,
            "is_underdog": is_underdog,
        })

    value_plays.sort(key=lambda x: x["edge"], reverse=True)

    # Header tweet
    day_name = game_date.strftime("%A")
    dogs = sum(1 for p in value_plays[:5] if p["is_underdog"])
    favs = min(len(value_plays), 5) - dogs

    pick_desc_parts = []
    if dogs:
        pick_desc_parts.append(f"{dogs} underdog{'s' if dogs != 1 else ''}")
    if favs:
        pick_desc_parts.append(f"{favs} favorite{'s' if favs != 1 else ''}")
    pick_desc = " and ".join(pick_desc_parts) if pick_desc_parts else "no value plays"

    if value_plays:
        header = (
            f"MLB picks for {day_name} — {len(games)} games, "
            f"{min(len(value_plays), 5)} cleared the model.\n\n"
            f"Today's card: {pick_desc}.\n\n"
            f"truline.app\n\n"
            f"#MLB #GamblingX"
        )
    else:
        header = (
            f"MLB picks for {day_name} — {len(games)} games analyzed, "
            f"nothing cleared the model today.\n\n"
            f"truline.app\n\n"
            f"#MLB #GamblingX"
        )
    tweets.append(header)

    if not value_plays:
        return tweets

    for play in value_plays[:5]:
        game = play["game"]
        away = game.away_team
        home = game.home_team
        team = play["team"]
        model_pct = round(play["model_p"] * 100)
        market_pct = round(play["market_p"] * 100)
        edge_pct = play["edge"] * 100
        odds_str = _fmt_odds(play["odds"])

        game_time = ""
        if game.game_time:
            et = game.game_time - timedelta(hours=4)
            try:
                game_time = et.strftime("%-I:%M %p")
            except Exception:
                game_time = et.strftime("%I:%M %p").lstrip("0")

        away_name = TEAM_NAMES.get(away, away)
        home_name = TEAM_NAMES.get(home, home)
        team_name = TEAM_NAMES.get(team, team)
        hashtag = TEAM_HASHTAGS.get(team, "")

        # Fetch context
        l10 = None
        stat_row = await session.execute(
            select(MLBTeamStats).where(
                MLBTeamStats.team_abbr == team
            ).order_by(desc(MLBTeamStats.stat_date)).limit(1)
        )
        stat = stat_row.scalar_one_or_none()
        if stat:
            l10 = stat.last_10_record or (
                f"{stat.last_10_wins}-{stat.last_10_losses}"
                if stat.last_10_wins is not None and stat.last_10_losses is not None
                else None
            )

        streak = await _get_team_streak(session, team)

        starter_id = game.home_starter_id if team == home else game.away_starter_id
        pitcher_last, pitcher_era = await _get_pitcher_era(session, starter_id)
        fi_record = await _get_pitcher_first_inning_record(session, starter_id)

        # Build narrative
        matchup_line = f"{away_name} @ {home_name}"
        if game_time:
            matchup_line += f" — {game_time}"

        context_bits = []
        if l10:
            context_bits.append(f"{l10} in their last 10")
        if streak:
            context_bits.append(f"on a {streak[1:]}-game {'win' if streak[0] == 'W' else 'losing'} streak")

        pitcher_bit = ""
        if pitcher_last and fi_record and fi_record[1] >= 3:
            pitcher_bit = f" {pitcher_last} ({pitcher_era:.2f} ERA) has held opponents scoreless in the 1st in {fi_record[0]} of {fi_record[1]} starts."

        context_str = ""
        if context_bits:
            context_str = f" They're {', '.join(context_bits)}."

        dog_or_fav = "underdog" if play["is_underdog"] else "favorite"

        tweet = (
            f"{matchup_line}\n\n"
            f"Model likes {team_name} ML at {odds_str} ({dog_or_fav}).{context_str}{pitcher_bit}\n\n"
            f"{model_pct}% model vs {market_pct}% market — {edge_pct:.1f}% edge.\n\n"
            f"{hashtag} #MLB"
        ).strip()

        if len(tweet) > 280:
            tweet = (
                f"{matchup_line}\n\n"
                f"Model likes {team_name} ML at {odds_str}.{context_str}\n\n"
                f"{model_pct}% vs {market_pct}% — {edge_pct:.1f}% edge.\n\n"
                f"{hashtag} #MLB"
            ).strip()
        if len(tweet) > 280:
            tweet = (
                f"{matchup_line}\n\n"
                f"Model likes {team_name} ML at {odds_str}.\n\n"
                f"{model_pct}% vs {market_pct}% — {edge_pct:.1f}% edge.\n\n"
                f"{hashtag} #MLB"
            ).strip()
        if len(tweet) > 280:
            tweet = tweet[:277] + "..."
        tweets.append(tweet)

    return tweets


async def generate_nrfi_tweet(session: AsyncSession, game_date: date) -> str | None:
    """
    Generate a NRFI (No Run First Inning) picks tweet for today's games.

    Uses MLBTeamStats.first_inning_score_pct directly and includes
    pitcher matchups with ERA.
    """
    # Today's scheduled games
    games_result = await session.execute(
        select(MLBGame).where(
            and_(
                MLBGame.game_date == game_date,
                MLBGame.status == "scheduled",
            )
        ).order_by(MLBGame.game_time)
    )
    games = list(games_result.scalars().all())
    if not games:
        return None

    nrfi_picks = []
    for game in games:
        home_off, home_def = await _get_team_first_inning_pct(session, game.home_team)
        away_off, away_def = await _get_team_first_inning_pct(session, game.away_team)
        if home_off is None or away_off is None or home_def is None or away_def is None:
            continue

        p_away_scores = (away_off + home_def) / 2.0
        p_home_scores = (home_off + away_def) / 2.0
        combined_nrfi = (1.0 - p_away_scores) * (1.0 - p_home_scores)

        away_last, away_era = await _get_pitcher_era(session, game.away_starter_id)
        home_last, home_era = await _get_pitcher_era(session, game.home_starter_id)

        nrfi_picks.append({
            "game": game,
            "combined_nrfi": combined_nrfi,
            "away_last": away_last,
            "away_era": away_era,
            "home_last": home_last,
            "home_era": home_era,
        })

    if not nrfi_picks:
        return None

    nrfi_picks.sort(key=lambda x: x["combined_nrfi"], reverse=True)

    date_str = game_date.strftime("%m/%d")
    footer = "\nFree picks at truline.app\n\n#NRFI #MLB #GamblingX"

    def build(picks_to_include):
        lines = [f"NRFI Plays {date_str}\n"]
        for pick in picks_to_include:
            game = pick["game"]
            away = game.away_team
            home = game.home_team
            pct = round(pick["combined_nrfi"] * 100)
            lines.append(f"* {away} @ {home} {pct}% NRFI")
            if pick["away_last"] and pick["home_last"]:
                a_era = f"{pick['away_era']:.2f}" if pick["away_era"] is not None else "-"
                h_era = f"{pick['home_era']:.2f}" if pick["home_era"] is not None else "-"
                lines.append(f"  {pick['away_last']} {a_era} vs {pick['home_last']} {h_era}")
            lines.append("")
        return "\n".join(lines).rstrip() + footer

    # Try 5 down to 3 picks until it fits
    for n in (5, 4, 3):
        tweet = build(nrfi_picks[:n])
        if len(tweet) <= 280:
            return tweet

    # Final fallback: truncate
    tweet = build(nrfi_picks[:3])
    if len(tweet) > 280:
        tweet = tweet[:277] + "..."
    return tweet


async def generate_results_tweet(session: AsyncSession, game_date: date) -> str | None:
    """Generate a results recap tweet with narrative context."""
    stmt = select(MLBPredictionSnapshot).where(
        and_(
            MLBPredictionSnapshot.game_date == game_date,
            MLBPredictionSnapshot.best_bet_result.isnot(None),
        )
    )
    result = await session.execute(stmt)
    snapshots = list(result.scalars().all())

    if not snapshots:
        return None

    wins = sum(1 for s in snapshots if s.best_bet_result == "win")
    losses = sum(1 for s in snapshots if s.best_bet_result == "loss")
    pushes = sum(1 for s in snapshots if s.best_bet_result == "push")
    profit = sum(float(s.best_bet_profit or 0) for s in snapshots)

    total = wins + losses
    win_rate = round(wins / total * 100, 1) if total > 0 else 0

    profit_str = f"+{profit:.2f}" if profit >= 0 else f"{profit:.2f}"

    # NRFI results
    nrfi_result = await session.execute(
        select(MLBGame).where(
            and_(
                MLBGame.game_date == game_date,
                MLBGame.status == "final",
                MLBGame.home_first_inning_runs.isnot(None),
            )
        )
    )
    nrfi_games = list(nrfi_result.scalars().all())
    nrfi_count = sum(1 for g in nrfi_games
                     if (g.home_first_inning_runs or 0) + (g.away_first_inning_runs or 0) == 0)
    nrfi_pct = round(nrfi_count / len(nrfi_games) * 100) if nrfi_games else 0

    # Best / worst
    graded = [s for s in snapshots if s.best_bet_profit is not None]
    best_line = ""
    worst_line = ""
    if graded:
        sorted_by_profit = sorted(graded, key=lambda s: float(s.best_bet_profit or 0), reverse=True)
        best = sorted_by_profit[0]
        worst = sorted_by_profit[-1]

        def _fmt_snap_name(s):
            team = TEAM_NAMES.get(s.best_bet_team, s.best_bet_team or "?")
            btype = (s.best_bet_type or "").lower()
            label = "ML" if btype == "moneyline" else ("runline" if btype == "runline" else ("O/U" if btype == "total" else ""))
            return f"{team} {label}".strip()

        if float(best.best_bet_profit or 0) > 0:
            best_line = f"Best: {_fmt_snap_name(best)} W\n"
        if float(worst.best_bet_profit or 0) < 0:
            worst_line = f"Worst: {_fmt_snap_name(worst)} L\n"

    # Bet type context
    type_summary = await _get_bet_type_summary(session, days=7)
    type_context = ""
    ml_data = type_summary.get("moneyline")
    if ml_data and (ml_data["wins"] + ml_data["losses"]) >= 3:
        ml_total = ml_data["wins"] + ml_data["losses"]
        type_context = f" ML picks are {ml_data['wins']}-{ml_data['losses']} this week."

    # Build tweet
    record_str = f"{wins}-{losses}"
    if pushes:
        record_str += f"-{pushes}"

    tweet = f"Yesterday: {record_str}, {profit_str}u.{type_context}\n"
    if nrfi_games:
        tweet += f"NRFI: {nrfi_count}/{len(nrfi_games)} ({nrfi_pct}%).\n"
    if best_line or worst_line:
        tweet += "\n" + best_line + worst_line

    tweet += "\ntruline.app\n\n#MLB #SportsBetting"

    if len(tweet) > 280:
        tweet = f"Yesterday: {record_str}, {profit_str}u.\n"
        if nrfi_games:
            tweet += f"NRFI: {nrfi_count}/{len(nrfi_games)} ({nrfi_pct}%).\n"
        if best_line or worst_line:
            tweet += "\n" + best_line + worst_line
        tweet += "\ntruline.app\n\n#MLB #SportsBetting"

    if len(tweet) > 280:
        tweet = tweet[:277] + "..."

    return tweet


async def generate_nrfi_results_tweet(session: AsyncSession, game_date: date) -> str | None:
    """Generate a tweet showing yesterday's NRFI results with our top picks."""
    result = await session.execute(
        select(MLBGame).where(
            and_(
                MLBGame.game_date == game_date,
                MLBGame.status == "final",
                MLBGame.home_first_inning_runs.isnot(None),
            )
        ).order_by(MLBGame.game_time)
    )
    games = list(result.scalars().all())
    if not games:
        return None

    date_str = game_date.strftime("%m/%d")
    nrfi_hit_count = 0

    # Score games for NRFI % like the picks tweet did, to reconstruct "our top picks"
    scored = []
    for g in games:
        first_inn_runs = (g.home_first_inning_runs or 0) + (g.away_first_inning_runs or 0)
        scoreless = first_inn_runs == 0
        if scoreless:
            nrfi_hit_count += 1

        home_off, home_def = await _get_team_first_inning_pct(session, g.home_team)
        away_off, away_def = await _get_team_first_inning_pct(session, g.away_team)
        if home_off is None or away_off is None or home_def is None or away_def is None:
            continue
        p_away_scores = (away_off + home_def) / 2.0
        p_home_scores = (home_off + away_def) / 2.0
        combined_nrfi = (1.0 - p_away_scores) * (1.0 - p_home_scores)
        scored.append({
            "game": g,
            "nrfi_prob": combined_nrfi,
            "scoreless": scoreless,
        })

    nrfi_pct = round(nrfi_hit_count / len(games) * 100) if games else 0

    tweet = (
        f"NRFI RESULTS {date_str}\n\n"
        f"NRFI: {nrfi_hit_count}/{len(games)} ({nrfi_pct}%)\n"
    )

    if scored:
        scored.sort(key=lambda x: x["nrfi_prob"], reverse=True)
        top_picks = scored[:3]
        tweet += "\nOur top NRFI picks:\n"
        for p in top_picks:
            g = p["game"]
            mark = "W" if p["scoreless"] else "L"
            pct = round(p["nrfi_prob"] * 100)
            tweet += f"{mark} {g.away_team} @ {g.home_team} ({pct}%)\n"

    tweet += "\nTrack daily at truline.app\n\n#NRFI #MLB"

    if len(tweet) > 280:
        tweet = tweet[:277] + "..."

    return tweet


def _nrfi_badge(nrfi_pct: float) -> str:
    """Return a badge label based on NRFI percentage (0-100)."""
    if nrfi_pct >= 70:
        return "🔥 STRONG NRFI"
    if nrfi_pct >= 55:
        return "✅ LEAN NRFI"
    if nrfi_pct >= 40:
        return "⚖️ TOSSUP"
    return "⚠️ LEAN YRFI"


def _fmt_game_time_et(game_time: datetime | None) -> str:
    """Format game_time as e.g. '7:05 PM ET'. Returns empty string if None."""
    if not game_time:
        return ""
    et = game_time - timedelta(hours=4)
    try:
        return et.strftime("%-I:%M %p ET")
    except Exception:
        return et.strftime("%I:%M %p ET").lstrip("0")


async def generate_pregame_nrfi_tweet(session: AsyncSession, game: MLBGame) -> str | None:
    """Generate a single-game NRFI pregame pick tweet with pitcher context."""
    home_off, home_def = await _get_team_first_inning_pct(session, game.home_team)
    away_off, away_def = await _get_team_first_inning_pct(session, game.away_team)
    if home_off is None or away_off is None or home_def is None or away_def is None:
        return None

    away_last, away_era = await _get_pitcher_era(session, game.away_starter_id)
    home_last, home_era = await _get_pitcher_era(session, game.home_starter_id)
    if not away_last or not home_last:
        return None

    p_away_scores = (away_off + home_def) / 2.0
    p_home_scores = (home_off + away_def) / 2.0
    nrfi_pct = (1.0 - p_away_scores) * (1.0 - p_home_scores) * 100.0
    nrfi_pct_rounded = round(nrfi_pct)

    away_name = TEAM_NAMES.get(game.away_team, game.away_team)
    home_name = TEAM_NAMES.get(game.home_team, game.home_team)
    away_handle = TEAM_HANDLES.get(game.away_team, game.away_team)
    home_handle = TEAM_HANDLES.get(game.home_team, game.home_team)

    game_time_str = _fmt_game_time_et(game.game_time)

    away_fi = await _get_pitcher_first_inning_record(session, game.away_starter_id)
    home_fi = await _get_pitcher_first_inning_record(session, game.home_starter_id)

    pitcher_context = ""
    if away_fi and home_fi and away_fi[1] >= 3 and home_fi[1] >= 3:
        pitcher_context = (
            f"{home_last} has blanked the 1st in {home_fi[0]} of {home_fi[1]} starts, "
            f"{away_last} in {away_fi[0]} of {away_fi[1]}."
        )
    elif home_fi and home_fi[1] >= 3:
        pitcher_context = f"{home_last} has blanked the 1st in {home_fi[0]} of {home_fi[1]} starts."
    elif away_fi and away_fi[1] >= 3:
        pitcher_context = f"{away_last} has blanked the 1st in {away_fi[0]} of {away_fi[1]} starts."

    matchup = f"{away_name} @ {home_name}"
    if game_time_str:
        matchup += f", {game_time_str}"

    if nrfi_pct_rounded >= 70:
        confidence = f"NRFI at {nrfi_pct_rounded}% — strong lean."
    elif nrfi_pct_rounded >= 60:
        confidence = f"NRFI at {nrfi_pct_rounded}%."
    else:
        confidence = f"NRFI at {nrfi_pct_rounded}% — slight lean."

    parts = [matchup, "", confidence]
    if pitcher_context:
        parts.append(pitcher_context)
    parts.extend(["", f"{away_handle} vs {home_handle}", "", "#NRFI #MLB"])

    tweet = "\n".join(parts)
    if len(tweet) > 280:
        parts = [matchup, "", confidence, "", f"{away_handle} vs {home_handle}", "", "#NRFI #MLB"]
        tweet = "\n".join(parts)
    if len(tweet) > 280:
        tweet = tweet[:277] + "..."
    return tweet


def generate_first_inning_recap_tweet(game: MLBGame) -> str | None:
    """Generate a 1st inning recap tweet for a single game."""
    if game.home_first_inning_runs is None or game.away_first_inning_runs is None:
        return None

    away_name = TEAM_NAMES.get(game.away_team, game.away_team)
    home_name = TEAM_NAMES.get(game.home_team, game.home_team)
    away_handle = TEAM_HANDLES.get(game.away_team, game.away_team)
    home_handle = TEAM_HANDLES.get(game.home_team, game.home_team)

    away_runs = game.away_first_inning_runs
    home_runs = game.home_first_inning_runs
    total = (away_runs or 0) + (home_runs or 0)
    result_tag = "NRFI ✅ Model called it" if total == 0 else "YRFI ❌"

    lines = [
        "1st INNING RECAP",
        "",
        f"{away_name} @ {home_name}",
        "",
        f"1st: {away_runs}-{home_runs}",
        "",
        result_tag,
        "",
        f"{away_handle} vs {home_handle}",
        "",
        "#NRFI #MLB",
    ]
    tweet = "\n".join(lines)
    if len(tweet) > 280:
        tweet = tweet[:277] + "..."
    return tweet


# =========================================================================
# NBA content generators
# =========================================================================


def _fmt_american(odds_decimal: float) -> str:
    """Decimal -> American string with sign."""
    if not odds_decimal or odds_decimal <= 1.0:
        return "-"
    if odds_decimal >= 2.0:
        return f"+{round((odds_decimal - 1) * 100)}"
    return f"{round(-100 / (odds_decimal - 1))}"


def _nba_bet_label(
    market_type: str,
    outcome_label: str,
    line: float | None,
    home_team: str,
    away_team: str,
) -> tuple[str, str]:
    """Return (pick_team_abbr, pick_label) e.g. ('LAL', 'LAL -4.5')."""
    ol = (outcome_label or "").lower()
    mt = (market_type or "").lower()
    if mt == "total":
        direction = "Over" if "over" in ol else "Under"
        line_str = f" {line}" if line is not None else ""
        return ("TOTAL", f"{direction}{line_str}")
    is_home = "home" in ol
    team = home_team if is_home else away_team
    if mt == "spread":
        if line is not None:
            sign = "+" if line > 0 else ""
            return (team, f"{team} {sign}{line}")
        return (team, f"{team} spread")
    # moneyline
    return (team, f"{team} ML")


def _nba_season_phase(game_date: date) -> tuple[str, str]:
    """Returns (title_prefix, hashtags) based on the NBA calendar.

    - Before 4/14/2026: Regular Season
    - 4/15-4/18/2026: Play-In Tournament
    - 4/19-6/25/2026: Playoffs
    - 6/5-6/25: NBA Finals (detected as late playoffs)
    """
    # Play-in tournament
    if date(2026, 4, 15) <= game_date <= date(2026, 4, 18):
        return ("Play-In Picks", "#NBA #NBAPlayIn #PlayInTournament")
    # NBA Finals (best guess)
    if date(2026, 6, 5) <= game_date <= date(2026, 6, 25):
        return ("NBA Finals Picks", "#NBAFinals #NBA")
    # Playoffs
    if date(2026, 4, 19) <= game_date <= date(2026, 6, 25):
        return ("NBA Playoff Picks", "#NBA #NBAPlayoffs")
    # Regular season
    return ("NBA Picks", "#NBA #NBABets")


async def generate_nba_picks_thread(session: AsyncSession, game_date: date) -> list[str]:
    """Generate NBA value picks thread for a given date."""
    result = await session.execute(
        text("""
            SELECT
                g.game_id, g.home_team_id, g.away_team_id, g.tip_time_utc,
                m.market_type, m.outcome_label, m.line, m.odds_decimal,
                mp.p_true, mp.p_market, mp.raw_edge,
                vs.algo_a_value_score
            FROM value_scores vs
            JOIN markets m ON m.market_id = vs.market_id
            JOIN games g ON g.game_id = m.game_id
            JOIN model_predictions mp ON mp.prediction_id = vs.prediction_id
            WHERE DATE(g.tip_time_utc AT TIME ZONE 'America/New_York') = :date
              AND vs.active_algorithm = 'A'
              AND vs.algo_a_value_score >= 60
              AND g.status = 'scheduled'
            ORDER BY vs.algo_a_value_score DESC
            LIMIT 5
        """),
        {"date": game_date},
    )
    rows = result.fetchall()

    # Count games scheduled that day for the header
    count_result = await session.execute(
        text("""
            SELECT COUNT(*) FROM games
            WHERE DATE(tip_time_utc AT TIME ZONE 'America/New_York') = :date
              AND status = 'scheduled'
        """),
        {"date": game_date},
    )
    games_count = int(count_result.scalar() or 0)

    title, header_tags = _nba_season_phase(game_date)
    date_str = game_date.strftime("%m/%d")
    header = (
        f"{title} {date_str}\n\n"
        f"AI value bets from the model.\n"
        f"{games_count} games on the board.\n\n"
        f"Full card: truline.app\n\n"
        f"{header_tags} #SportsBetting"
    )
    tweets = [header]

    if not rows:
        return tweets

    for row in rows:
        home = row.home_team_id
        away = row.away_team_id
        odds_dec = float(row.odds_decimal) if row.odds_decimal else 0.0
        line = float(row.line) if row.line is not None else None
        pick_team, pick_label = _nba_bet_label(
            row.market_type, row.outcome_label, line, home, away
        )
        model_pct = round(float(row.p_true) * 100)
        market_pct = round(float(row.p_market) * 100)
        edge_pct = float(row.raw_edge) * 100
        odds_str = _fmt_american(odds_dec)
        score = int(row.algo_a_value_score or 0)

        game_time = ""
        if row.tip_time_utc:
            et = row.tip_time_utc - timedelta(hours=4)
            try:
                game_time = et.strftime("%-I:%M %p ET")
            except Exception:
                game_time = et.strftime("%I:%M %p ET").lstrip("0")

        away_name = NBA_TEAM_NAMES.get(away, away)
        home_name = NBA_TEAM_NAMES.get(home, home)
        handle = NBA_TEAM_HANDLES.get(pick_team, "")

        header_line = f"{away_name} @ {home_name}"
        if game_time:
            header_line += f"  {game_time}"

        tweet = (
            f"{header_line}\n\n"
            f"Model pick: {pick_label}\n"
            f"Odds: {odds_str}\n"
            f"Edge: +{edge_pct:.1f}%\n\n"
            f"Model: {model_pct}% | Market: {market_pct}%\n\n"
            f"Value Score: {score}/100\n\n"
            f"{handle} {header_tags}"
        ).strip()

        if len(tweet) > 280:
            tweet = tweet[:277] + "..."
        tweets.append(tweet)

    return tweets


async def generate_nba_results_tweet(session: AsyncSession, game_date: date) -> str | None:
    """NBA results recap tweet from prediction_snapshots.

    Posts results even when no bets qualified (reports winner prediction accuracy).
    """
    from src.models.prediction_snapshot import PredictionSnapshot

    # Fetch all graded snapshots (winner_correct set), not just those with bets
    stmt = select(PredictionSnapshot).where(
        and_(
            PredictionSnapshot.game_date == game_date,
            PredictionSnapshot.winner_correct.isnot(None),
        )
    )
    result = await session.execute(stmt)
    snapshots = list(result.scalars().all())
    if not snapshots:
        return None

    date_str = game_date.strftime("%m/%d")

    # Separate snapshots with qualifying bets from those without
    with_bets = [s for s in snapshots if s.best_bet_result is not None]

    # Winner prediction stats (always available)
    winners_correct = sum(1 for s in snapshots if s.winner_correct)
    winners_total = len(snapshots)
    winners_pct = round(winners_correct / winners_total * 100, 1) if winners_total else 0

    if with_bets:
        # Normal path: we had qualifying bets
        wins = sum(1 for s in with_bets if s.best_bet_result == "win")
        losses = sum(1 for s in with_bets if s.best_bet_result == "loss")
        pushes = sum(1 for s in with_bets if s.best_bet_result == "push")
        profit = sum(float(s.best_bet_profit or 0) for s in with_bets)
        total = wins + losses
        pct = round(wins / total * 100, 1) if total else 0
        profit_str = f"+{profit:.2f}" if profit >= 0 else f"{profit:.2f}"

        graded = [s for s in with_bets if s.best_bet_profit is not None]
        best_line = ""
        worst_line = ""
        if graded:
            srt = sorted(graded, key=lambda s: float(s.best_bet_profit or 0), reverse=True)
            best, worst = srt[0], srt[-1]

            def _fmt_snap(s, mark):
                team = s.best_bet_team or "?"
                btype = (s.best_bet_type or "").lower()
                label = "ML" if btype == "moneyline" else ("SPR" if btype == "spread" else ("O/U" if btype == "total" else ""))
                line_s = ""
                if s.best_bet_line is not None and btype in ("spread", "total"):
                    line_s = f" {float(s.best_bet_line)}"
                return f"{team} {label}{line_s} {mark}".strip()

            if float(best.best_bet_profit or 0) > 0:
                best_p = float(best.best_bet_profit or 0)
                best_line = f"Best: {_fmt_snap(best, 'W')} (+{best_p:.0f})\n"
            if float(worst.best_bet_profit or 0) < 0:
                worst_p = float(worst.best_bet_profit or 0)
                worst_line = f"Worst: {_fmt_snap(worst, 'L')} ({worst_p:.0f})\n"

        tweet = f"NBA RESULTS {date_str}\n\nBets: {wins}-{losses}"
        if pushes:
            tweet += f"-{pushes}"
        tweet += f" ({pct}%)\nP/L: {profit_str}u\n"
        tweet += f"Winners: {winners_correct}/{winners_total} ({winners_pct}%)\n"
        if best_line or worst_line:
            tweet += "\n" + best_line + worst_line
    else:
        # No qualifying bets — report winner predictions only
        tweet = f"NBA RESULTS {date_str}\n\n"
        tweet += f"No value bets qualified yesterday\n"
        tweet += f"Winner picks: {winners_correct}/{winners_total} ({winners_pct}%)\n"

    tweet += "\nFull report: truline.app\n\n#NBA #SportsBetting"

    if len(tweet) > 280:
        tweet = tweet[:277] + "..."
    return tweet


async def generate_nba_pregame_tweet(session: AsyncSession, market_id: str) -> str | None:
    """Single NBA pregame pick tweet for a given market."""
    result = await session.execute(
        text("""
            SELECT
                g.home_team_id, g.away_team_id, g.tip_time_utc,
                m.market_type, m.outcome_label, m.line, m.odds_decimal,
                mp.p_true, mp.p_market, mp.raw_edge,
                vs.algo_a_value_score
            FROM value_scores vs
            JOIN markets m ON m.market_id = vs.market_id
            JOIN games g ON g.game_id = m.game_id
            JOIN model_predictions mp ON mp.prediction_id = vs.prediction_id
            WHERE m.market_id = :mid
              AND vs.active_algorithm = 'A'
            ORDER BY vs.calc_time DESC
            LIMIT 1
        """),
        {"mid": market_id},
    )
    row = result.fetchone()
    if not row:
        return None

    home, away = row.home_team_id, row.away_team_id
    odds_dec = float(row.odds_decimal) if row.odds_decimal else 0.0
    line = float(row.line) if row.line is not None else None
    pick_team, pick_label = _nba_bet_label(row.market_type, row.outcome_label, line, home, away)
    odds_str = _fmt_american(odds_dec)
    edge_pct = float(row.raw_edge) * 100
    score = int(row.algo_a_value_score or 0)

    game_time = ""
    if row.tip_time_utc:
        et = row.tip_time_utc - timedelta(hours=4)
        try:
            game_time = et.strftime("%-I:%M %p ET")
        except Exception:
            game_time = et.strftime("%I:%M %p ET").lstrip("0")

    away_name = NBA_TEAM_NAMES.get(away, away)
    home_name = NBA_TEAM_NAMES.get(home, home)
    away_handle = NBA_TEAM_HANDLES.get(away, away)
    home_handle = NBA_TEAM_HANDLES.get(home, home)

    lines = [f"🏀 {away_name} @ {home_name}"]
    if game_time:
        lines.append(f"🕐 {game_time}")
    lines.append("")
    lines.append(f"Model Pick: {pick_label}")
    lines.append(f"Odds: {odds_str}")
    lines.append("")
    lines.append(f"Value Score: {score}/100")
    lines.append(f"Edge: +{edge_pct:.1f}%")
    lines.append("")
    lines.append(f"{away_handle} vs {home_handle}")
    lines.append("")
    lines.append("truline.app")
    lines.append("")
    lines.append("#NBA #NBAPlayoffs")

    tweet = "\n".join(lines)
    if len(tweet) > 280:
        tweet = tweet[:277] + "..."
    return tweet


def generate_final_recap_tweet(game: MLBGame) -> str | None:
    """Generate a final recap tweet for a single game."""
    if game.status != "final":
        return None
    if game.home_score is None or game.away_score is None:
        return None

    away_name = TEAM_NAMES.get(game.away_team, game.away_team)
    home_name = TEAM_NAMES.get(game.home_team, game.home_team)

    home_1st = game.home_first_inning_runs
    away_1st = game.away_first_inning_runs
    first_line = ""
    if home_1st is not None and away_1st is not None:
        tag = "NRFI" if (home_1st + away_1st) == 0 else "YRFI"
        first_line = f"1st Inning: {away_1st}-{home_1st} ({tag})"

    total_runs = (game.home_score or 0) + (game.away_score or 0)

    if game.home_score > game.away_score:
        winner = game.home_team
    elif game.away_score > game.home_score:
        winner = game.away_team
    else:
        winner = None
    winner_handle = TEAM_HANDLES.get(winner, winner) if winner else None

    lines = [
        "FINAL",
        "",
        f"{away_name} {game.away_score}, {home_name} {game.home_score}",
        "",
    ]
    if first_line:
        lines.append(first_line)
    lines.append(f"Total: {total_runs} runs")
    lines.append("")
    if winner_handle:
        lines.append(f"{winner_handle} wins")
        lines.append("")
    lines.append("#MLB")

    tweet = "\n".join(lines)
    if len(tweet) > 280:
        tweet = tweet[:277] + "..."
    return tweet


def generate_celebration_tweet(
    sport: str,
    winner_team: str,
    winner_name: str,
    loser_team: str,
    loser_name: str,
    odds_american: int,
    profit_units: float,
    score_text: str | None = None,
) -> str:
    """Build a tweet for a winning underdog moneyline pick.

    Tone: confident, specific, no emoji prefixes. Mentions the winning team
    handle for tagging.
    """
    if sport == "nba":
        winner_handle = NBA_TEAM_HANDLES.get(winner_team, f"@{winner_team}")
        loser_handle = NBA_TEAM_HANDLES.get(loser_team, f"@{loser_team}")
        league_tag = "#NBA"
    else:
        winner_handle = TEAM_HANDLES.get(winner_team, f"@{winner_team}")
        loser_handle = TEAM_HANDLES.get(loser_team, f"@{loser_team}")
        league_tag = "#MLB"

    odds_str = f"+{odds_american}" if odds_american > 0 else str(odds_american)

    parts = [f"{winner_name} hit at {odds_str}.", ""]
    parts.append(f"Model called the {winner_name} ML over the {loser_name}.")
    if score_text:
        parts.append(score_text)
    parts.append("")
    parts.append(f"+{profit_units:.2f}u on a unit bet.")
    parts.append("")
    parts.append(f"{winner_handle} over {loser_handle}")
    parts.append("")
    parts.append(f"{league_tag} #SportsBetting")

    tweet = "\n".join(parts)
    if len(tweet) > 280:
        # Trim by dropping the score + analytical line
        parts = [
            f"{winner_name} hit at {odds_str}.",
            "",
            f"+{profit_units:.2f}u on a unit bet.",
            "",
            f"{winner_handle} over {loser_handle}",
            "",
            f"{league_tag} #SportsBetting",
        ]
        tweet = "\n".join(parts)
    if len(tweet) > 280:
        tweet = tweet[:277] + "..."
    return tweet
