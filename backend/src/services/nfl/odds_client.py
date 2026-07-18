"""The Odds API client for NFL + pure parser to nfl_markets rows."""

import httpx
import structlog

from src.services.data.odds_api import OddsAPIClient

logger = structlog.get_logger()


# NFL team full name -> nflverse abbreviation (matches constants.NFL_DIVISIONS keys).
NFL_TEAM_NAME_TO_ABBR: dict[str, str] = {
    "Buffalo Bills": "BUF",
    "Miami Dolphins": "MIA",
    "New England Patriots": "NE",
    "New York Jets": "NYJ",
    "Baltimore Ravens": "BAL",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Pittsburgh Steelers": "PIT",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Tennessee Titans": "TEN",
    "Denver Broncos": "DEN",
    "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV",
    "Los Angeles Chargers": "LAC",
    "Dallas Cowboys": "DAL",
    "New York Giants": "NYG",
    "Philadelphia Eagles": "PHI",
    "Washington Commanders": "WAS",
    "Chicago Bears": "CHI",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Minnesota Vikings": "MIN",
    "Atlanta Falcons": "ATL",
    "Carolina Panthers": "CAR",
    "New Orleans Saints": "NO",
    "Tampa Bay Buccaneers": "TB",
    "Arizona Cardinals": "ARI",
    "Los Angeles Rams": "LA",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
}


class NFLOddsClient(OddsAPIClient):
    """Odds API client configured for NFL."""

    SPORT = "americanfootball_nfl"

    async def get_nfl_odds(
        self,
        markets: list[str] = ["h2h", "spreads", "totals"],
        bookmakers: list[str] | None = None,
    ) -> list[dict]:
        """Fetch current NFL odds."""
        params = {
            "apiKey": self.api_key,
            "regions": "us",
            "markets": ",".join(markets),
            "oddsFormat": "decimal",
        }

        if bookmakers:
            params["bookmakers"] = ",".join(bookmakers)

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.BASE_URL}/sports/{self.SPORT}/odds",
                params=params,
                timeout=30.0,
            )
            response.raise_for_status()

            self.requests_remaining = int(
                response.headers.get("x-requests-remaining", 0)
            )
            self.requests_used = int(response.headers.get("x-requests-used", 0))

            logger.info(
                "Fetched NFL odds",
                requests_remaining=self.requests_remaining,
            )

            return response.json()


def parse_nfl_odds_to_markets(
    odds_events: list[dict],
    team_name_to_abbr: dict[str, str],
) -> list[dict]:
    """Pure: map The Odds API v4 event JSON to NFLMarket kwargs dicts.

    One event yields up to 3 rows (spread, moneyline, total), using the
    first bookmaker in the event as the consensus book. `game_id` is left
    for the caller to resolve (matched to nfl_games by teams+date);
    `home_team_abbr`/`away_team_abbr`/`commence_date` (ISO `YYYY-MM-DD`,
    the UTC date portion of the event's `commence_time`) are included for
    that lookup — see `season_update.match_event_to_game`.

    Convention: our stored `line` for spreads is HOME-FAVORED POSITIVE,
    while The Odds API's home-team `point` is NEGATIVE when the home team
    is favored. So `line = -home_point` (KC -3.5 -> line +3.5).
    """
    rows: list[dict] = []

    for event in odds_events:
        home_team = event.get("home_team")
        away_team = event.get("away_team")
        home_abbr = team_name_to_abbr.get(home_team)
        away_abbr = team_name_to_abbr.get(away_team)

        if not home_abbr or not away_abbr:
            logger.warning(
                "Unknown NFL team in odds",
                home=home_team,
                away=away_team,
            )
            continue

        bookmakers = event.get("bookmakers", [])
        if not bookmakers:
            continue

        bookmaker = bookmakers[0]
        book = bookmaker.get("key")

        # commence_time is UTC ISO-8601, e.g. "2026-09-13T17:00:00Z" — take
        # the date portion so match_event_to_game (season_update.py) can key
        # on (home_team_abbr, away_team_abbr, commence_date).
        commence_time = event.get("commence_time") or ""
        commence_date = commence_time[:10] if commence_time else None

        base = {
            "home_team_abbr": home_abbr,
            "away_team_abbr": away_abbr,
            "book": book,
            "commence_date": commence_date,
        }

        for market in bookmaker.get("markets", []):
            market_key = market.get("key")
            outcomes = market.get("outcomes", [])

            if market_key == "h2h":
                home_odds = None
                away_odds = None
                for outcome in outcomes:
                    team_abbr = team_name_to_abbr.get(outcome.get("name"))
                    if team_abbr == home_abbr:
                        home_odds = outcome.get("price")
                    elif team_abbr == away_abbr:
                        away_odds = outcome.get("price")

                rows.append({
                    **base,
                    "market_type": "moneyline",
                    "line": None,
                    "home_odds": home_odds,
                    "away_odds": away_odds,
                    "over_odds": None,
                    "under_odds": None,
                })

            elif market_key == "spreads":
                line = None
                home_odds = None
                away_odds = None
                for outcome in outcomes:
                    team_abbr = team_name_to_abbr.get(outcome.get("name"))
                    if team_abbr == home_abbr:
                        home_point = outcome.get("point")
                        line = -home_point if home_point is not None else None
                        home_odds = outcome.get("price")
                    elif team_abbr == away_abbr:
                        away_odds = outcome.get("price")

                rows.append({
                    **base,
                    "market_type": "spread",
                    "line": line,
                    "home_odds": home_odds,
                    "away_odds": away_odds,
                    "over_odds": None,
                    "under_odds": None,
                })

            elif market_key == "totals":
                line = None
                over_odds = None
                under_odds = None
                for outcome in outcomes:
                    name = outcome.get("name")
                    if name == "Over":
                        line = outcome.get("point")
                        over_odds = outcome.get("price")
                    elif name == "Under":
                        under_odds = outcome.get("price")

                rows.append({
                    **base,
                    "market_type": "total",
                    "line": line,
                    "home_odds": None,
                    "away_odds": None,
                    "over_odds": over_odds,
                    "under_odds": under_odds,
                })

    return rows
