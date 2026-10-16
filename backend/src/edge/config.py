# backend/src/edge/config.py
"""TruLine Edge constants. Values are the spec's (§5, §6); do not tune here."""
from dataclasses import dataclass

SHARP_BOOKS = frozenset({"pinnacle", "novig", "prophetx"})
DFS_BOOKS = frozenset({"prizepicks", "underdog"})
BOOKS = (
    "pinnacle", "novig", "prophetx",
    "draftkings", "fanduel", "betmgm", "williamhill_us", "espnbet", "fanatics",
    "betrivers", "hardrockbet", "betonlineag", "bovada", "lowvig",
    "prizepicks", "underdog",
)
BOOK_WEIGHTS = {"pinnacle": 3.0, "novig": 2.0, "prophetx": 2.0, "draftkings": 1.5, "fanduel": 1.5}

LINE_MARKETS = ("h2h", "spreads", "totals")
NBA_PROP_MARKETS = (
    "player_points", "player_rebounds", "player_assists", "player_threes",
    "player_points_rebounds_assists", "player_points_rebounds",
    "player_points_assists", "player_rebounds_assists",
)
# Anytime TD is excluded: most books quote only "Yes", so there is no
# two-sided market to devig (2026-10-16 sample: 319 Yes vs 55 No).
NFL_PROP_MARKETS = (
    "player_pass_yds", "player_pass_tds", "player_pass_completions", "player_pass_attempts",
    "player_pass_interceptions", "player_rush_yds", "player_rush_attempts",
    "player_receptions", "player_reception_yds", "player_rush_reception_yds",
)


@dataclass(frozen=True)
class SportConfig:
    key: str                         # Odds API sport key
    slug: str                        # API value and state-key prefix
    prop_markets: tuple[str, ...]
    lines_window_hours: int          # poll game lines from this far before kickoff
    has_players: bool                # player matching, pages and prop grading


NBA = SportConfig("basketball_nba", "nba", NBA_PROP_MARKETS, 36, True)
NFL = SportConfig("americanfootball_nfl", "nfl", NFL_PROP_MARKETS, 7 * 24, False)
SPORTS = {"nba": NBA, "nfl": NFL}

# Legacy aliases (NBA), kept so existing imports keep working.
SPORT_KEY = NBA.key
PROP_MARKETS = NBA_PROP_MARKETS


def enabled_sports(csv: str) -> list[SportConfig]:
    import logging
    out = []
    for slug in (s.strip() for s in csv.split(",")):
        if not slug:
            continue
        if slug in SPORTS:
            out.append(SPORTS[slug])
        else:
            logging.getLogger("edge.config").warning("EDGE_SPORTS: unknown sport %r ignored", slug)
    return out


def sport_by_key(key: str) -> SportConfig | None:
    return next((s for s in SPORTS.values() if s.key == key), None)


def state_key(sport: SportConfig, name: str) -> str:
    """NBA keeps its pre-multi-sport key names so the live worker's state carries over."""
    return name if sport.slug == "nba" else f"{sport.slug}:{name}"


MIN_BOOKS = 3
DEVIG_METHOD = "power"
# The close is read from edge_quotes, which freezes at tip because polling
# stops there. A key whose last sighting is older than this window was pulled
# before tip and is not part of the close. 30 covers the 2-minute final-hour
# cadence even at guard factor 8 (16 minutes).
CLOSING_WINDOW_MINUTES = 30
