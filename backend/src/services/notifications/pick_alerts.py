"""Formats frozen best_bet picks as founder SMS alerts."""

from datetime import datetime, timedelta

from src.services.social.content import _fmt_american


def _last_name(full_name: str | None) -> str | None:
    if not full_name or not full_name.strip():
        return None
    return full_name.strip().split()[-1]


def format_pick_alert(
    *,
    away_team: str,
    home_team: str,
    bet_type: str,
    team: str | None,
    line: float | None,
    odds_decimal: float | None,
    value_score: int,
    edge: float | None,
    game_time: datetime | None,
    away_starter: str | None = None,
    home_starter: str | None = None,
) -> str:
    if bet_type == "moneyline":
        label = f"{team} ML"
    elif bet_type == "runline":
        if line is None:
            label = f"{team} RL"
        else:
            sign = "+" if line > 0 else ""
            label = f"{team} {sign}{line:g}"
    else:  # totals — not in best_bet while the re-entry gate is closed
        label = f"O/U {line:g}" if line is not None else "O/U"

    if team is None:
        opponent = f"{away_team} @ {home_team}"
    elif team == home_team:
        opponent = f"vs {away_team}"
    else:
        opponent = f"@ {home_team}"

    odds_str = ""
    if odds_decimal:
        am = _fmt_american(float(odds_decimal))
        if am != "-":
            odds_str = f" ({am})"

    time_str = ""
    if game_time:
        # EDT display, same convention as content.py picks thread
        et = game_time - timedelta(hours=4)
        time_str = ", " + et.strftime("%I:%M %p").lstrip("0") + " ET"

    parts = [f"Score {int(value_score)}"]
    if edge is not None:
        parts.append(f"Edge {round(float(edge) * 100)}%")
    away_p, home_p = _last_name(away_starter), _last_name(home_starter)
    if away_p and home_p:
        parts.append(f"{away_p} vs {home_p}")

    return f"TruLine pick: {label}{odds_str} {opponent}{time_str}\n" + " | ".join(parts)
