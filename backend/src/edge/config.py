# backend/src/edge/config.py
"""TruLine Edge constants. Values are the spec's (§5, §6); do not tune here."""
SPORT_KEY = "basketball_nba"

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
PROP_MARKETS = (
    "player_points", "player_rebounds", "player_assists", "player_threes",
    "player_points_rebounds_assists", "player_points_rebounds",
    "player_points_assists", "player_rebounds_assists",
)
MIN_BOOKS = 3
DEVIG_METHOD = "power"
# The close is read from edge_quotes, which freezes at tip because polling
# stops there. A key whose last sighting is older than this window was pulled
# before tip and is not part of the close. 30 covers the 2-minute final-hour
# cadence even at guard factor 8 (16 minutes).
CLOSING_WINDOW_MINUTES = 30
