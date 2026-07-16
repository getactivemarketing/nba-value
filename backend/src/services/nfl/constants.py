"""Static NFL reference data and pure schedule derivations."""

# nflverse team abbreviations -> division.
NFL_DIVISIONS: dict[str, str] = {
    "BUF": "AFC East", "MIA": "AFC East", "NE": "AFC East", "NYJ": "AFC East",
    "BAL": "AFC North", "CIN": "AFC North", "CLE": "AFC North", "PIT": "AFC North",
    "HOU": "AFC South", "IND": "AFC South", "JAX": "AFC South", "TEN": "AFC South",
    "DEN": "AFC West", "KC": "AFC West", "LV": "AFC West", "LAC": "AFC West",
    "DAL": "NFC East", "NYG": "NFC East", "PHI": "NFC East", "WAS": "NFC East",
    "CHI": "NFC North", "DET": "NFC North", "GB": "NFC North", "MIN": "NFC North",
    "ATL": "NFC South", "CAR": "NFC South", "NO": "NFC South", "TB": "NFC South",
    "ARI": "NFC West", "LA": "NFC West", "SF": "NFC West", "SEA": "NFC West",
}

# Historical / variant nflverse abbreviations -> current franchise code.
CANONICAL_TEAM = {
    "OAK": "LV", "SD": "LAC", "STL": "LA",
    "LAR": "LA", "WSH": "WAS", "JAC": "JAX", "ARZ": "ARI",
}


def normalize_team(abbr):
    """Map historical/variant team abbreviations to the current franchise code."""
    return CANONICAL_TEAM.get(abbr, abbr) if isinstance(abbr, str) else abbr


def is_divisional(home: str, away: str) -> bool:
    """True when both teams share a division."""
    h, a = NFL_DIVISIONS.get(home), NFL_DIVISIONS.get(away)
    return h is not None and h == a


def primetime_slot(weekday: str, gametime: str) -> str | None:
    """Classify a nationally televised primetime window.

    weekday: nflverse 'weekday' (e.g. 'Sunday'); gametime: 'HH:MM' 24h ET.
    Sunday counts as primetime only at/after 19:00 (SNF).
    """
    if not gametime:
        return None
    try:
        hour = int(gametime.split(":")[0])
    except (ValueError, IndexError, AttributeError):
        return None
    if weekday == "Thursday" and hour >= 19:
        return "TNF"
    if weekday == "Monday" and hour >= 19:
        return "MNF"
    if weekday == "Sunday" and hour >= 19:
        return "SNF"
    return None


def is_primetime(weekday: str, gametime: str) -> bool:
    return primetime_slot(weekday, gametime) is not None
