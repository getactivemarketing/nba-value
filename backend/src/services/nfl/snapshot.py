"""Build and grade nfl_prediction_snapshots (flat $100 units).

Grading conventions:
- Spread/total are -110 flat bets: win = +90.9, loss = -100, push = 0.
- Moneyline pays actual odds: win = (odds_decimal - 1) * 100, loss = -100.
- A moneyline tie (actual_margin == 0) is graded as a push (0 profit), not a loss.
  NFL ties are astronomically rare (regular season only, after one OT period with
  no score); treating it as a loss would be an unearned penalty on the pick, and
  sportsbooks refund moneyline bets on a tie, so "push" matches real-world settlement.
"""
from datetime import datetime, timezone

_WIN_110 = 100 * (100 / 110)  # 90.909...


def build_snapshot(game: dict, scored: dict) -> dict:
    bs, bm, bt, bb = scored["best_spread"], scored["best_ml"], scored["best_total"], scored["best_bet"]
    row = {
        "game_id": game["game_id"],
        "snapshot_time": game.get("snapshot_time") or datetime.now(timezone.utc),
        "home_team": game["home_team"], "away_team": game["away_team"],
        "kickoff_utc": game.get("kickoff_utc"), "game_date": game.get("game_date"),
        "predicted_margin": scored["predicted_margin"], "predicted_total": scored["predicted_total"],

        "best_spread_team": bs.team if bs else None,
        "best_spread_line": bs.line if bs else None,
        "best_spread_odds": bs.odds_decimal if bs else None,
        "best_spread_value_score": bs.value_score if bs else None,
        "best_spread_edge": bs.raw_edge if bs else None,

        "best_ml_team": bm.team if bm else None,
        "best_ml_odds": bm.odds_decimal if bm else None,
        "best_ml_value_score": bm.value_score if bm else None,
        "best_ml_edge": bm.raw_edge if bm else None,

        "best_total_direction": bt.bet_type if bt else None,
        "best_total_line": bt.line if bt else None,
        "best_total_odds": bt.odds_decimal if bt else None,
        "best_total_value_score": bt.value_score if bt else None,
        "best_total_edge": bt.raw_edge if bt else None,

        "best_bet_type": bb.market_type if bb else None,
        "best_bet_team": bb.team if bb else None,
        "best_bet_line": bb.line if bb else None,
        "best_bet_odds": bb.odds_decimal if bb else None,
        "best_bet_value_score": bb.value_score if bb else None,
        "best_bet_edge": bb.raw_edge if bb else None,
    }
    return row


def _grade_total(direction, line, actual_total):
    if direction is None or line is None:
        return None, None
    if actual_total == line:
        return "push", 0.0
    over_hit = actual_total > line
    won = over_hit if direction == "over" else not over_hit
    return ("win", _WIN_110) if won else ("loss", -100.0)


def _grade_spread(team, line, actual_margin):
    """team is "home"/"away"; line is that picked side's own spread line.
    Home covers iff actual_margin > line; away covers iff actual_margin < line;
    push iff actual_margin == line (exact-line equality)."""
    if team is None or line is None:
        return None, None
    if actual_margin == line:
        return "push", 0.0
    if team == "home":
        covered = actual_margin > line
    else:
        covered = actual_margin < line
    return ("win", _WIN_110) if covered else ("loss", -100.0)


def _grade_moneyline(team, odds, actual_margin):
    if team is None or odds is None:
        return None, None
    if actual_margin == 0:
        return "push", 0.0  # documented decision: NFL tie -> push, not loss
    home_won = actual_margin > 0
    won = home_won if team == "home" else not home_won
    return ("win", (odds - 1) * 100) if won else ("loss", -100.0)


def grade_snapshot(snap: dict, home_score, away_score, spread_line, total_line) -> dict:
    actual_margin = home_score - away_score
    actual_total = home_score + away_score
    out = {
        "actual_margin": actual_margin, "actual_total": actual_total,
        "home_score": home_score, "away_score": away_score,
    }

    total_result, total_profit = _grade_total(
        snap.get("best_total_direction"), snap.get("best_total_line"), actual_total)
    out["best_total_result"], out["best_total_profit"] = total_result, total_profit

    spread_result, spread_profit = _grade_spread(
        snap.get("best_spread_team"), snap.get("best_spread_line"), actual_margin)
    out["best_spread_result"], out["best_spread_profit"] = spread_result, spread_profit

    ml_result, ml_profit = _grade_moneyline(
        snap.get("best_ml_team"), snap.get("best_ml_odds"), actual_margin)
    out["best_ml_result"], out["best_ml_profit"] = ml_result, ml_profit

    best_bet_type = snap.get("best_bet_type")
    mirror = {"total": (total_result, total_profit),
              "spread": (spread_result, spread_profit),
              "moneyline": (ml_result, ml_profit)}.get(best_bet_type)
    if mirror:
        out["best_bet_result"], out["best_bet_profit"] = mirror
    else:
        out["best_bet_result"], out["best_bet_profit"] = None, None

    return out
