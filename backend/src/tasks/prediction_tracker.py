"""
Prediction Tracker - Snapshots predictions before games and grades them after.

This module handles:
1. Saving predictions to prediction_snapshots before tip-off
2. Grading predictions after games complete
3. Tracking model performance over time
"""

import asyncio
import json
from datetime import datetime, timezone, timedelta
from decimal import Decimal

import psycopg2
import structlog

from src.services.injuries import get_all_team_injury_reports

logger = structlog.get_logger()

DB_URL = 'postgresql://postgres:wzYHkiAOkykxiPitXKBIqPJxvifFtDPI@maglev.proxy.rlwy.net:46068/railway'

# Minimum value score to consider a bet worth tracking
# Below this threshold, we don't record a "best bet" - it's a pass
# Set to 65 after analysis showed low-value bets (40-60) had poor win rates
MIN_VALUE_THRESHOLD = 65

# Higher threshold for weak teams on the road (require more edge)
MIN_VALUE_THRESHOLD_WEAK_ROAD = 72

# Import suppress_totals flag from config (centralized setting)
# Totals model has 41% win rate, suppressed until proper model is trained
try:
    from src.config import settings
    SUPPRESS_TOTALS = settings.suppress_totals
except ImportError:
    SUPPRESS_TOTALS = True  # Default to suppressed if config unavailable


# ============================================================================
# TEAM QUALITY & RISK ASSESSMENT
# Added to fix: 98.9% away bias, blowout losses, picking bad road teams
# ============================================================================

def get_team_quality(cur, team_id: str) -> dict:
    """
    Assess team quality based on record, net rating, and recent form.

    Returns:
        dict with quality metrics and tier classification
    """
    cur.execute('''
        SELECT
            wins, losses, net_rtg_10, wins_l10, losses_l10,
            home_wins, home_losses, away_wins, away_losses
        FROM team_stats
        WHERE team_id = %s
        ORDER BY stat_date DESC
        LIMIT 1
    ''', (team_id,))

    row = cur.fetchone()
    if not row:
        return {"tier": "unknown", "win_pct": 0.5, "net_rtg": 0, "is_weak": False, "is_elite": False}

    wins, losses, net_rtg, wins_l10, losses_l10, home_w, home_l, away_w, away_l = row

    total_games = (wins or 0) + (losses or 0)
    win_pct = wins / total_games if total_games > 0 else 0.5

    l10_win_pct = wins_l10 / 10 if wins_l10 is not None else win_pct

    away_games = (away_w or 0) + (away_l or 0)
    away_win_pct = away_w / away_games if away_games > 0 else 0.5

    net_rtg = float(net_rtg) if net_rtg else 0

    # Classify team tier
    # Elite: Top teams (>60% win rate AND positive net rating)
    # Good: Above average (>50% win rate)
    # Average: Around .500
    # Weak: Below average (<45% win rate OR very negative net rating)
    # Bottom: Worst teams (<35% win rate)

    if win_pct >= 0.60 and net_rtg >= 3:
        tier = "elite"
    elif win_pct >= 0.55 or net_rtg >= 2:
        tier = "good"
    elif win_pct >= 0.45:
        tier = "average"
    elif win_pct >= 0.35:
        tier = "weak"
    else:
        tier = "bottom"

    return {
        "tier": tier,
        "win_pct": round(win_pct, 3),
        "away_win_pct": round(away_win_pct, 3),
        "net_rtg": net_rtg,
        "l10_win_pct": round(l10_win_pct, 3),
        "is_weak": tier in ("weak", "bottom"),
        "is_elite": tier == "elite",
        "is_bottom": tier == "bottom",
    }


def assess_blowout_risk(bet_team_quality: dict, opponent_quality: dict,
                        is_home_bet: bool, spread_line: float) -> dict:
    """
    Assess risk of a blowout loss.

    High blowout risk scenarios:
    - Bottom team on road vs elite team
    - Weak team on road getting <10 points vs good+ team
    - Team on cold streak (<30% L10) on road

    Returns:
        dict with risk level and recommendation
    """
    risk_score = 0
    risk_factors = []

    # Factor 1: Team quality mismatch
    tier_scores = {"elite": 5, "good": 4, "average": 3, "weak": 2, "bottom": 1, "unknown": 3}
    bet_tier = tier_scores.get(bet_team_quality.get("tier", "unknown"), 3)
    opp_tier = tier_scores.get(opponent_quality.get("tier", "unknown"), 3)

    tier_diff = opp_tier - bet_tier

    if tier_diff >= 3:  # e.g., bottom team vs elite
        risk_score += 40
        risk_factors.append(f"Large tier mismatch ({bet_team_quality['tier']} vs {opponent_quality['tier']})")
    elif tier_diff >= 2:  # e.g., weak team vs good
        risk_score += 25
        risk_factors.append(f"Tier mismatch ({bet_team_quality['tier']} vs {opponent_quality['tier']})")

    # Factor 2: Road underdog with small spread
    if not is_home_bet and spread_line and spread_line > 0:
        # Betting on road underdog
        if bet_team_quality.get("is_weak") and spread_line < 10:
            risk_score += 30
            risk_factors.append(f"Weak road dog getting only +{spread_line}")
        if bet_team_quality.get("is_bottom"):
            risk_score += 20
            risk_factors.append("Bottom-tier team on road")

    # Factor 3: Poor road record
    away_win_pct = bet_team_quality.get("away_win_pct", 0.5)
    if not is_home_bet and away_win_pct < 0.30:
        risk_score += 25
        risk_factors.append(f"Poor road record ({away_win_pct:.0%})")

    # Factor 4: Cold streak (L10)
    l10_win_pct = bet_team_quality.get("l10_win_pct", 0.5)
    if l10_win_pct < 0.30:
        risk_score += 20
        risk_factors.append(f"Cold streak ({l10_win_pct:.0%} L10)")

    # Factor 5: Opponent is elite at home
    if opponent_quality.get("is_elite") and is_home_bet == False:
        risk_score += 15
        risk_factors.append("Facing elite team at their home")

    # Determine risk level
    if risk_score >= 60:
        risk_level = "high"
    elif risk_score >= 35:
        risk_level = "medium"
    else:
        risk_level = "low"

    return {
        "risk_level": risk_level,
        "risk_score": risk_score,
        "risk_factors": risk_factors,
        "should_skip": risk_score >= 60,  # Skip high-risk bets
        "require_higher_threshold": risk_score >= 35,  # Require higher value score
    }


def should_consider_home_bet(home_quality: dict, away_quality: dict,
                              home_value_score: float, away_value_score: float) -> bool:
    """
    Check if we should favor the home bet to balance away bias.

    Returns True if home bet should be considered even if slightly lower score.
    """
    # If home team is clearly better quality and scores are close, prefer home
    if home_quality.get("tier") in ("elite", "good") and away_quality.get("is_weak"):
        # Home is much better - give home bet a boost consideration
        if home_value_score and away_value_score:
            # If home score is within 5 points of away, prefer home
            if home_value_score >= away_value_score - 5:
                return True

    # If away team is bottom tier on road, strongly prefer home
    if away_quality.get("is_bottom") and home_value_score and home_value_score >= MIN_VALUE_THRESHOLD:
        return True

    return False


def adjust_value_score_for_quality(base_score: float, bet_team_quality: dict,
                                    opponent_quality: dict, is_home_bet: bool) -> float:
    """
    Adjust value score based on team quality factors.

    Penalizes:
    - Weak teams on the road
    - Bottom teams against elite opponents

    Bonuses:
    - Elite teams at home
    - Good teams against weak opponents
    """
    adjustment = 0

    # Penalty for weak/bottom road teams
    if not is_home_bet:
        if bet_team_quality.get("is_bottom"):
            adjustment -= 15  # Big penalty for bottom road teams
        elif bet_team_quality.get("is_weak"):
            adjustment -= 8   # Moderate penalty for weak road teams

    # Penalty for facing elite teams on road
    if not is_home_bet and opponent_quality.get("is_elite"):
        adjustment -= 10

    # Bonus for elite/good home teams
    if is_home_bet:
        if bet_team_quality.get("is_elite"):
            adjustment += 5
        elif bet_team_quality.get("tier") == "good":
            adjustment += 3

    # Bonus for playing against weak/bottom teams
    if opponent_quality.get("is_bottom"):
        adjustment += 5
    elif opponent_quality.get("is_weak"):
        adjustment += 3

    adjusted_score = base_score + adjustment

    # Log significant adjustments
    if abs(adjustment) >= 5:
        logger.debug(
            "Value score adjusted for quality",
            base_score=base_score,
            adjustment=adjustment,
            adjusted_score=adjusted_score,
            bet_team_tier=bet_team_quality.get("tier"),
            opponent_tier=opponent_quality.get("tier"),
            is_home=is_home_bet,
        )

    return max(0, min(100, adjusted_score))  # Keep in 0-100 range


def snapshot_predictions(hours_ahead: float = 0.75, db_url: str = None) -> dict:
    """
    Snapshot predictions for games starting in the next N hours.

    Default is 0.75 hours (45 minutes) to capture predictions close to tip-off.
    Combined with a 15-minute schedule, this captures games ~30 min before tip.

    Args:
        hours_ahead: How many hours ahead to look for games (supports decimals)
        db_url: Database connection string

    Returns:
        Summary of snapshots created
    """
    conn = psycopg2.connect(db_url or DB_URL)
    conn.autocommit = True
    cur = conn.cursor()

    now = datetime.now(timezone.utc)
    cutoff = now + timedelta(hours=hours_ahead)

    # Find upcoming games that haven't been snapshotted yet
    cur.execute('''
        SELECT DISTINCT g.game_id, g.home_team_id, g.away_team_id, g.tip_time_utc
        FROM games g
        WHERE g.tip_time_utc > %s
        AND g.tip_time_utc < %s
        AND g.status = 'scheduled'
        AND NOT EXISTS (
            SELECT 1 FROM prediction_snapshots ps
            WHERE ps.game_id = g.game_id
        )
        ORDER BY g.tip_time_utc
    ''', (now, cutoff))

    games = cur.fetchall()

    if not games:
        logger.info("No new games to snapshot")
        cur.close()
        conn.close()
        return {"games_snapshotted": 0, "status": "no_new_games"}

    # Fetch injury reports for all teams
    try:
        injury_reports = asyncio.run(get_all_team_injury_reports())
        logger.info(f"Fetched injury reports for {len(injury_reports)} teams")
    except Exception as e:
        logger.warning(f"Failed to fetch injury reports: {e}")
        injury_reports = {}

    snapshots_created = 0

    for game_id, home_team, away_team, tip_time in games:
        # Get team stats for prediction factors
        home_stats = get_team_stats(cur, home_team)
        away_stats = get_team_stats(cur, away_team)

        # Get value scores for this game - both algorithms
        cur.execute('''
            SELECT
                m.market_type,
                m.outcome_label,
                m.line,
                m.odds_decimal,
                vs.algo_a_value_score,
                vs.algo_a_edge_score,
                vs.algo_a_confidence,
                vs.algo_b_value_score,
                vs.algo_b_combined_edge,
                vs.algo_b_confidence,
                vs.active_algorithm,
                mp.p_true,
                mp.p_market,
                mp.raw_edge
            FROM value_scores vs
            JOIN markets m ON vs.market_id = m.market_id
            JOIN model_predictions mp ON vs.prediction_id = mp.prediction_id
            WHERE m.game_id = %s
            ORDER BY vs.algo_b_value_score DESC
        ''', (game_id,))

        scores = cur.fetchall()

        if not scores:
            logger.info(f"No value scores for game {game_id}, skipping")
            continue

        # ================================================================
        # TEAM QUALITY ASSESSMENT (Fix for away bias and blowout losses)
        # ================================================================
        home_quality = get_team_quality(cur, home_team)
        away_quality = get_team_quality(cur, away_team)

        logger.info(
            f"Team quality: {home_team} ({home_quality['tier']}, {home_quality['win_pct']:.0%}) vs "
            f"{away_team} ({away_quality['tier']}, {away_quality['win_pct']:.0%})"
        )

        # Determine predicted winner from moneyline markets
        home_prob = 0.5
        away_prob = 0.5
        best_bet_a = None  # Best bet according to Algorithm A
        best_bet_b = None  # Best bet according to Algorithm B
        best_score_a = 0
        best_score_b = 0
        best_total = None  # Best total (over/under) bet
        best_total_score = 0
        active_algo = 'b'  # Default

        # Track all spread bets for comparison (to fix home/away balance)
        home_spread_bet = None
        away_spread_bet = None

        for row in scores:
            (mtype, outcome, line, odds,
             algo_a_score, algo_a_edge, algo_a_conf,
             algo_b_score, algo_b_edge, algo_b_conf,
             active_algorithm, p_true, p_market, raw_edge) = row

            is_home = 'home' in outcome.lower()
            active_algo = (active_algorithm or 'b').lower()

            # Get the bet team and opponent for quality assessment
            bet_team = home_team if is_home else away_team
            opponent = away_team if is_home else home_team
            bet_quality = home_quality if is_home else away_quality
            opp_quality = away_quality if is_home else home_quality

            # ================================================================
            # QUALITY-ADJUSTED VALUE SCORE
            # ================================================================
            raw_algo_a_score = float(algo_a_score) if algo_a_score else 0
            raw_algo_b_score = float(algo_b_score) if algo_b_score else 0

            # Adjust scores based on team quality
            adj_algo_a_score = adjust_value_score_for_quality(
                raw_algo_a_score, bet_quality, opp_quality, is_home
            ) if raw_algo_a_score > 0 else 0

            adj_algo_b_score = adjust_value_score_for_quality(
                raw_algo_b_score, bet_quality, opp_quality, is_home
            ) if raw_algo_b_score > 0 else 0

            # ================================================================
            # BLOWOUT RISK CHECK (for spread bets only)
            # ================================================================
            skip_bet = False
            required_threshold = MIN_VALUE_THRESHOLD

            if mtype == 'spread':
                blowout_risk = assess_blowout_risk(
                    bet_quality, opp_quality, is_home, float(line) if line else 0
                )

                if blowout_risk["should_skip"]:
                    logger.warning(
                        f"Skipping high blowout risk bet: {bet_team} {line}",
                        risk_score=blowout_risk["risk_score"],
                        factors=blowout_risk["risk_factors"],
                    )
                    skip_bet = True
                elif blowout_risk["require_higher_threshold"]:
                    required_threshold = MIN_VALUE_THRESHOLD_WEAK_ROAD
                    logger.info(
                        f"Elevated threshold for {bet_team}: {required_threshold}",
                        risk_factors=blowout_risk["risk_factors"],
                    )

                # Track home/away spread bets separately
                bet_data = {
                    "type": mtype,
                    "team": bet_team,
                    "line": float(line) if line else None,
                    "value_score": int(adj_algo_a_score),
                    "raw_score": int(raw_algo_a_score),
                    "edge_score": float(algo_a_edge) if algo_a_edge else 0,
                    "confidence": float(algo_a_conf) if algo_a_conf else 1.0,
                    "odds": float(odds) if odds else None,
                    "p_true": float(p_true) if p_true else 0,
                    "p_market": float(p_market) if p_market else 0,
                    "is_home": is_home,
                    "blowout_risk": blowout_risk["risk_level"],
                    "skip": skip_bet,
                    "threshold": required_threshold,
                }

                if is_home:
                    home_spread_bet = bet_data
                else:
                    away_spread_bet = bet_data

            if skip_bet:
                continue

            # Track best value bet for Algorithm A (only if meets threshold)
            if adj_algo_a_score > best_score_a and adj_algo_a_score >= required_threshold:
                best_score_a = adj_algo_a_score
                best_bet_a = {
                    "type": mtype,
                    "team": bet_team,
                    "line": float(line) if line else None,
                    "value_score": int(adj_algo_a_score),
                    "raw_score": int(raw_algo_a_score),
                    "edge_score": float(algo_a_edge) if algo_a_edge else 0,
                    "confidence": float(algo_a_conf) if algo_a_conf else 1.0,
                    "odds": float(odds) if odds else None,
                    "p_true": float(p_true) if p_true else 0,
                    "p_market": float(p_market) if p_market else 0,
                    "is_home": is_home,
                }

            # Track best value bet for Algorithm B (only if meets threshold)
            if adj_algo_b_score > best_score_b and adj_algo_b_score >= required_threshold:
                best_score_b = adj_algo_b_score
                best_bet_b = {
                    "type": mtype,
                    "team": bet_team,
                    "line": float(line) if line else None,
                    "value_score": int(adj_algo_b_score),
                    "raw_score": int(raw_algo_b_score),
                    "combined_edge": float(algo_b_edge) if algo_b_edge else 0,
                    "confidence": float(algo_b_conf) if algo_b_conf else 1.0,
                    "odds": float(odds) if odds else None,
                    "p_true": float(p_true) if p_true else 0,
                    "p_market": float(p_market) if p_market else 0,
                    "is_home": is_home,
                }

            # Track best total bet separately (only if meets threshold and not suppressed)
            if not SUPPRESS_TOTALS and mtype == 'total' and adj_algo_a_score > best_total_score and adj_algo_a_score >= MIN_VALUE_THRESHOLD:
                best_total_score = adj_algo_a_score
                best_total = {
                    "direction": outcome.replace('_', ''),  # "over" or "under"
                    "line": float(line) if line else None,
                    "value_score": int(adj_algo_a_score),
                    "edge": float(algo_a_edge) if algo_a_edge else 0,
                    "odds": float(odds) if odds else None,
                }

            # Get moneyline probabilities
            if mtype == 'moneyline':
                if is_home:
                    home_prob = float(p_true) if p_true else 0.5
                else:
                    away_prob = float(p_true) if p_true else 0.5

        # ================================================================
        # HOME/AWAY BALANCE CHECK (Fix for 98.9% away bias)
        # ================================================================
        # If we're about to pick an away bet, check if home should be considered
        best_bet = best_bet_a if active_algo == 'a' else best_bet_b

        if best_bet and not best_bet.get("is_home") and home_spread_bet and away_spread_bet:
            # We picked away - should we reconsider home?
            if should_consider_home_bet(
                home_quality, away_quality,
                home_spread_bet.get("value_score", 0),
                away_spread_bet.get("value_score", 0)
            ):
                # Check if home bet meets threshold and isn't high risk
                if (home_spread_bet.get("value_score", 0) >= MIN_VALUE_THRESHOLD
                    and not home_spread_bet.get("skip")
                    and home_spread_bet.get("blowout_risk") != "high"):

                    logger.info(
                        f"Switching to home bet due to quality mismatch: "
                        f"{home_team} ({home_spread_bet['value_score']}) over "
                        f"{away_team} ({away_spread_bet['value_score']})"
                    )
                    best_bet = {
                        "type": home_spread_bet["type"],
                        "team": home_spread_bet["team"],
                        "line": home_spread_bet["line"],
                        "value_score": home_spread_bet["value_score"],
                        "edge_score": home_spread_bet.get("edge_score", 0),
                        "confidence": home_spread_bet.get("confidence", 1.0),
                        "odds": home_spread_bet.get("odds"),
                        "p_true": home_spread_bet.get("p_true", 0),
                        "p_market": home_spread_bet.get("p_market", 0),
                        "is_home": True,
                    }

        # Log final bet selection
        if best_bet:
            logger.info(
                f"Selected bet: {best_bet['team']} {best_bet.get('line', '')} "
                f"(score: {best_bet['value_score']}, home: {best_bet.get('is_home', 'N/A')})"
            )

        # Determine predicted winner
        if home_prob >= away_prob:
            predicted_winner = home_team
            winner_prob = home_prob
        else:
            predicted_winner = away_team
            winner_prob = away_prob

        # Confidence level
        if winner_prob >= 0.65:
            confidence = 'high'
        elif winner_prob >= 0.55:
            confidence = 'medium'
        else:
            confidence = 'low'

        # Build explanation factors
        factors = build_factors(home_team, away_team, home_stats, away_stats, best_bet)

        # Get injury data for this game
        home_injury = injury_reports.get(home_team)
        away_injury = injury_reports.get(away_team)

        home_injury_score = home_injury.spread_injury_score if home_injury else 0
        away_injury_score = away_injury.spread_injury_score if away_injury else 0
        home_ppg_lost = home_injury.total_ppg_lost if home_injury else 0
        away_ppg_lost = away_injury.total_ppg_lost if away_injury else 0
        injury_edge = away_injury_score - home_injury_score  # Positive = home advantage

        # Add injury factor if significant
        if abs(injury_edge) >= 0.05:
            advantage_team = home_team if injury_edge > 0 else away_team
            factors.append(f"{advantage_team} injury advantage ({abs(injury_edge):.0%})")

        # Get line movement data
        line_movement = get_line_movement(cur, game_id)

        # Add line movement factor if significant
        if line_movement.get("spread_movement") and abs(line_movement["spread_movement"]) >= 0.5:
            direction = "toward home" if line_movement["spread_movement"] < 0 else "toward away"
            factors.append(f"Line moved {abs(line_movement['spread_movement'])} pts {direction}")

        # Calculate game_date from tip_time (use Eastern Time for NBA game dates)
        # tip_time is UTC, convert to ET for the game date
        eastern_offset = timedelta(hours=-5)
        tip_time_et = tip_time + eastern_offset if tip_time.tzinfo else tip_time.replace(tzinfo=timezone.utc) + eastern_offset
        game_date = tip_time_et.date()

        # Insert snapshot with both algorithm scores, total bet, and line movement
        cur.execute('''
            INSERT INTO prediction_snapshots (
                game_id, snapshot_time, home_team, away_team, tip_time, game_date,
                predicted_winner, winner_probability, winner_confidence,
                best_bet_type, best_bet_team, best_bet_line,
                best_bet_value_score, best_bet_edge, best_bet_odds,
                best_total_direction, best_total_line, best_total_value_score,
                best_total_edge, best_total_odds,
                factors,
                home_injury_score, away_injury_score, home_ppg_lost, away_ppg_lost, injury_edge,
                algo_a_value_score, algo_a_edge_score, algo_a_confidence,
                algo_b_value_score, algo_b_combined_edge, algo_b_confidence,
                active_algorithm,
                opening_spread, current_spread, spread_movement,
                opening_total, current_total, total_movement,
                line_movement_direction
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                      %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ''', (
            game_id, now, home_team, away_team, tip_time, game_date,
            predicted_winner, round(winner_prob * 100, 1), confidence,
            best_bet['type'] if best_bet else None,
            best_bet['team'] if best_bet else None,
            best_bet['line'] if best_bet else None,
            best_bet['value_score'] if best_bet else None,
            best_bet.get('edge') or best_bet.get('edge_score') or best_bet.get('combined_edge') if best_bet else None,
            best_bet['odds'] if best_bet else None,
            # Total bet fields
            best_total['direction'] if best_total else None,
            best_total['line'] if best_total else None,
            best_total['value_score'] if best_total else None,
            best_total['edge'] if best_total else None,
            best_total['odds'] if best_total else None,
            json.dumps(factors),
            round(home_injury_score, 2),
            round(away_injury_score, 2),
            round(home_ppg_lost, 1),
            round(away_ppg_lost, 1),
            round(injury_edge, 2),
            # Algorithm A
            best_bet_a['value_score'] if best_bet_a else None,
            best_bet_a['edge_score'] if best_bet_a else None,
            best_bet_a['confidence'] if best_bet_a else None,
            # Algorithm B
            best_bet_b['value_score'] if best_bet_b else None,
            best_bet_b['combined_edge'] if best_bet_b else None,
            best_bet_b['confidence'] if best_bet_b else None,
            active_algo,
            # Line movement
            line_movement.get("opening_spread"),
            line_movement.get("current_spread"),
            line_movement.get("spread_movement"),
            line_movement.get("opening_total"),
            line_movement.get("current_total"),
            line_movement.get("total_movement"),
            line_movement.get("direction"),
        ))

        snapshots_created += 1
        logger.info(f"Snapshotted {away_team} @ {home_team}: {predicted_winner} to win ({confidence})")

        if best_bet and best_bet['value_score'] >= MIN_VALUE_THRESHOLD:
            logger.info(f"  Best bet: {best_bet['team']} {best_bet['type']} {best_bet['line'] or ''} ({best_bet['value_score']}%)")
        else:
            logger.info(f"  No qualifying bet (threshold: {MIN_VALUE_THRESHOLD})")

        if line_movement.get("spread_movement") and abs(line_movement["spread_movement"]) >= 0.5:
            logger.info(f"  Line move: {line_movement['opening_spread']} -> {line_movement['current_spread']} ({line_movement['direction']})")

    cur.close()
    conn.close()

    return {
        "games_snapshotted": snapshots_created,
        "snapshot_time": now.isoformat(),
        "status": "completed"
    }


def get_line_movement(cur, game_id: str) -> dict:
    """
    Get opening and current lines for a game to track line movement.

    Returns dict with opening/current spread and total, plus movement.
    Line movement helps identify sharp money:
    - Spread moving toward a team often indicates sharp action
    - Total movement shows where money is going on O/U
    """
    result = {
        "opening_spread": None,
        "current_spread": None,
        "spread_movement": None,
        "opening_total": None,
        "current_total": None,
        "total_movement": None,
        "direction": None,
    }

    # Get opening lines (earliest snapshot for this game)
    cur.execute('''
        SELECT home_spread, total_line, snapshot_time
        FROM odds_snapshots
        WHERE game_id = %s
        AND home_spread IS NOT NULL
        ORDER BY snapshot_time ASC
        LIMIT 1
    ''', (game_id,))

    opening = cur.fetchone()
    if opening:
        result["opening_spread"] = float(opening[0]) if opening[0] else None
        result["opening_total"] = float(opening[1]) if opening[1] else None

    # Get current lines (most recent snapshot)
    cur.execute('''
        SELECT home_spread, total_line, snapshot_time
        FROM odds_snapshots
        WHERE game_id = %s
        AND home_spread IS NOT NULL
        ORDER BY snapshot_time DESC
        LIMIT 1
    ''', (game_id,))

    current = cur.fetchone()
    if current:
        result["current_spread"] = float(current[0]) if current[0] else None
        result["current_total"] = float(current[1]) if current[1] else None

    # Fallback: If odds_snapshots is empty, get current lines from markets table
    if result["current_spread"] is None or result["current_total"] is None:
        cur.execute('''
            SELECT
                (SELECT line FROM markets WHERE game_id = %s AND market_type = 'spread'
                 AND outcome_label ILIKE '%%home%%' LIMIT 1) as home_spread,
                (SELECT line FROM markets WHERE game_id = %s AND market_type = 'total'
                 AND outcome_label ILIKE '%%over%%' LIMIT 1) as total_line
        ''', (game_id, game_id))
        fallback = cur.fetchone()
        if fallback:
            if result["current_spread"] is None and fallback[0]:
                result["current_spread"] = float(fallback[0])
            if result["current_total"] is None and fallback[1]:
                result["current_total"] = float(fallback[1])
            # Use as opening lines too if not available
            if result["opening_spread"] is None:
                result["opening_spread"] = result["current_spread"]
            if result["opening_total"] is None:
                result["opening_total"] = result["current_total"]

    # Calculate movement
    if result["opening_spread"] is not None and result["current_spread"] is not None:
        # Movement: current - opening
        # Positive = line moved toward away team (home got more points)
        # Negative = line moved toward home team (home gave more points)
        result["spread_movement"] = round(result["current_spread"] - result["opening_spread"], 1)

        # Determine direction (half point or more is significant)
        if result["spread_movement"] <= -0.5:
            result["direction"] = "sharp_home"  # Money on home
        elif result["spread_movement"] >= 0.5:
            result["direction"] = "sharp_away"  # Money on away

    if result["opening_total"] is not None and result["current_total"] is not None:
        result["total_movement"] = round(result["current_total"] - result["opening_total"], 1)

        # Combine with spread direction if significant total movement
        if result["total_movement"] <= -0.5:
            if result["direction"]:
                result["direction"] += "_under"
            else:
                result["direction"] = "steam_under"
        elif result["total_movement"] >= 0.5:
            if result["direction"]:
                result["direction"] += "_over"
            else:
                result["direction"] = "steam_over"

    return result


def get_team_stats(cur, team_id: str) -> dict:
    """Get latest team stats for building prediction factors."""
    cur.execute('''
        SELECT
            wins, losses, net_rtg_10, days_rest, is_back_to_back,
            wins_l10, losses_l10, home_wins, home_losses, away_wins, away_losses,
            ats_wins_l10, ats_losses_l10
        FROM team_stats
        WHERE team_id = %s
        ORDER BY stat_date DESC
        LIMIT 1
    ''', (team_id,))

    row = cur.fetchone()
    if not row:
        return {}

    return {
        "wins": row[0] or 0,
        "losses": row[1] or 0,
        "net_rtg_l10": float(row[2]) if row[2] else None,
        "days_rest": row[3],
        "is_b2b": row[4] or False,
        "wins_l10": row[5] or 0,
        "losses_l10": row[6] or 0,
        "home_wins": row[7] or 0,
        "home_losses": row[8] or 0,
        "away_wins": row[9] or 0,
        "away_losses": row[10] or 0,
        "ats_wins_l10": row[11] or 0,
        "ats_losses_l10": row[12] or 0,
    }


def build_factors(home_team: str, away_team: str, home_stats: dict, away_stats: dict, best_bet: dict | None) -> list[str]:
    """Build explanation factors for the prediction."""
    factors = []

    # 1. Net rating comparison
    home_net = home_stats.get('net_rtg_l10')
    away_net = away_stats.get('net_rtg_l10')
    if home_net is not None and away_net is not None:
        diff = home_net - away_net
        if abs(diff) >= 1.0:
            better = home_team if diff > 0 else away_team
            factors.append(f"{better} +{abs(diff):.1f} Net Rating (L10)")

    # 2. Rest advantage
    home_b2b = home_stats.get('is_b2b', False)
    away_b2b = away_stats.get('is_b2b', False)
    home_rest = home_stats.get('days_rest') or 0
    away_rest = away_stats.get('days_rest') or 0

    if home_b2b and not away_b2b:
        factors.append(f"{away_team} rest advantage (vs B2B)")
    elif away_b2b and not home_b2b:
        factors.append(f"{home_team} rest advantage (vs B2B)")
    elif abs(home_rest - away_rest) >= 2:
        better = home_team if home_rest > away_rest else away_team
        factors.append(f"{better} +{abs(home_rest - away_rest)} days rest")

    # 3. Model edge on best bet
    if best_bet and best_bet.get('edge', 0) > 0:
        p_true = best_bet.get('p_true', 50)
        p_market = best_bet.get('p_market', 50)
        edge = best_bet.get('edge', 0)
        factors.append(f"Model: {p_true:.0f}% vs Market: {p_market:.0f}% (+{edge:.1f}% edge)")

    # 4. L10 record comparison
    home_l10_wins = home_stats.get('wins_l10', 0)
    away_l10_wins = away_stats.get('wins_l10', 0)
    if abs(home_l10_wins - away_l10_wins) >= 3:
        home_l10 = f"{home_stats.get('wins_l10', 0)}-{home_stats.get('losses_l10', 0)}"
        away_l10 = f"{away_stats.get('wins_l10', 0)}-{away_stats.get('losses_l10', 0)}"
        better = home_team if home_l10_wins > away_l10_wins else away_team
        better_record = home_l10 if home_l10_wins > away_l10_wins else away_l10
        factors.append(f"{better} is {better_record} in L10")

    return factors[:4]


def grade_bet(bet_type: str, bet_team: str, home_team: str, actual_winner: str,
               spread_result: str, total_result: str, bet_odds: float = None) -> tuple:
    """Grade a single bet and return (result, profit)."""
    bet_result = None
    bet_profit = None

    if not bet_type or not bet_team:
        return None, None

    if bet_type == 'spread':
        is_home_bet = (bet_team == home_team)
        if is_home_bet:
            bet_result = 'win' if spread_result == 'home_cover' else ('push' if spread_result == 'push' else 'loss')
        else:
            bet_result = 'win' if spread_result == 'away_cover' else ('push' if spread_result == 'push' else 'loss')

    elif bet_type == 'moneyline':
        bet_result = 'win' if bet_team == actual_winner else 'loss'

    elif bet_type == 'total':
        if total_result:
            bet_result = 'push' if total_result == 'push' else total_result

    # Calculate profit (assuming $100 bet at -110 for spreads/totals)
    if bet_result == 'win':
        if bet_odds:
            bet_profit = 100 * (float(bet_odds) - 1)
        else:
            bet_profit = 90.91  # -110 payout
    elif bet_result == 'loss':
        bet_profit = -100
    else:
        bet_profit = 0  # Push

    return bet_result, bet_profit


def grade_predictions(db_url: str = None) -> dict:
    """
    Grade predictions for completed games.

    Looks for games that have:
    - A prediction snapshot
    - Final scores in game_results (or games table)
    - Not yet graded

    IMPORTANT: Uses the best_bet_line and best_total_line captured at snapshot time
    for grading, NOT the closing_spread/total from game_results (which may be incorrect).

    Grades both Algorithm A and Algorithm B picks for comparison.

    Returns:
        Summary of grading results including per-algorithm performance
    """
    conn = psycopg2.connect(db_url or DB_URL)
    conn.autocommit = True
    cur = conn.cursor()

    # Find ungraded predictions with completed games
    # Use snapshot's current_spread/current_total for grading (captured pre-game), not game_results closing lines
    cur.execute('''
        SELECT
            ps.id, ps.game_id, ps.predicted_winner, ps.winner_probability,
            ps.best_bet_type, ps.best_bet_team, ps.best_bet_line,
            ps.best_bet_value_score, ps.best_bet_odds,
            gr.actual_winner, gr.home_score, gr.away_score,
            ps.current_spread as snapshot_spread, ps.current_total as snapshot_total,
            NULL as spread_result, NULL as total_result,
            ps.home_team, ps.away_team,
            ps.algo_a_value_score, ps.algo_b_value_score, ps.active_algorithm,
            ps.best_total_direction, ps.best_total_line, ps.best_total_odds
        FROM prediction_snapshots ps
        JOIN game_results gr ON ps.game_id = gr.game_id
        WHERE ps.winner_correct IS NULL
        AND gr.actual_winner IS NOT NULL
    ''')

    predictions = cur.fetchall()

    if not predictions:
        cur.close()
        conn.close()
        return {"predictions_graded": 0, "status": "no_predictions_to_grade"}

    graded = 0
    wins = 0
    losses = 0
    pushes = 0
    total_profit = 0

    # Per-algorithm tracking
    algo_a_stats = {"wins": 0, "losses": 0, "pushes": 0, "profit": 0}
    algo_b_stats = {"wins": 0, "losses": 0, "pushes": 0, "profit": 0}

    for row in predictions:
        (pred_id, game_id, predicted_winner, winner_prob,
         bet_type, bet_team, bet_line, bet_value, bet_odds,
         actual_winner, home_score, away_score,
         snapshot_spread, snapshot_total, _spread_result, _total_result,
         home_team, away_team,
         algo_a_value, algo_b_value, active_algo,
         total_direction, total_line, total_odds) = row

        # Derive home_spread from best_bet_line when bet_type is 'spread'
        # best_bet_line is from the perspective of bet_team:
        #   - If bet_team is AWAY, bet_line is positive (away getting points), so home_spread = -bet_line
        #   - If bet_team is HOME, bet_line is negative (home giving points), so home_spread = bet_line
        # Fall back to current_spread only if best_bet is not a spread bet
        if bet_type == 'spread' and bet_line is not None:
            if bet_team == away_team:
                closing_spread = -float(bet_line)  # Convert away spread to home spread
            else:
                closing_spread = float(bet_line)  # Already home spread
        else:
            closing_spread = float(snapshot_spread) if snapshot_spread is not None else None

        # For totals, prefer best_total_line over current_total
        if total_line is not None:
            closing_total = float(total_line)
        elif snapshot_total is not None:
            closing_total = float(snapshot_total)
        else:
            closing_total = None

        # Calculate spread_result ourselves using the derived home spread
        spread_result = None
        if closing_spread is not None and home_score is not None and away_score is not None:
            home_adjusted = home_score + closing_spread
            if home_adjusted > away_score:
                spread_result = 'home_cover'
            elif home_adjusted < away_score:
                spread_result = 'away_cover'
            else:
                spread_result = 'push'

        # Calculate total_result ourselves using the closing total
        total_result = None
        if closing_total is not None and home_score is not None and away_score is not None:
            actual_total = home_score + away_score
            if actual_total > float(closing_total):
                total_result = 'over'
            elif actual_total < float(closing_total):
                total_result = 'under'
            else:
                total_result = 'push'

        # Grade winner prediction
        winner_correct = (predicted_winner == actual_winner)

        # Grade total bet if present
        total_bet_result = None
        total_bet_profit = None
        if total_direction and total_result:
            if total_result == 'push':
                total_bet_result = 'push'
                total_bet_profit = 0
            elif total_direction == total_result:  # 'over' == 'over' or 'under' == 'under'
                total_bet_result = 'win'
                total_bet_profit = 90.91 if total_odds is None else 100 * (float(total_odds) - 1)
            else:
                total_bet_result = 'loss'
                total_bet_profit = -100

        # Grade the primary best bet (from active algorithm)
        bet_result, bet_profit = grade_bet(
            bet_type, bet_team, home_team, actual_winner,
            spread_result, total_result, bet_odds
        )

        # For now, use same bet for both algos (they pick same bet, just different scores)
        # In future, we could track separate best bets per algorithm
        algo_a_result = bet_result
        algo_b_result = bet_result
        algo_a_profit = bet_profit
        algo_b_profit = bet_profit

        # Track per-algorithm stats
        if algo_a_result == 'win':
            algo_a_stats["wins"] += 1
            algo_a_stats["profit"] += algo_a_profit or 0
        elif algo_a_result == 'loss':
            algo_a_stats["losses"] += 1
            algo_a_stats["profit"] += algo_a_profit or 0
        else:
            algo_a_stats["pushes"] += 1

        if algo_b_result == 'win':
            algo_b_stats["wins"] += 1
            algo_b_stats["profit"] += algo_b_profit or 0
        elif algo_b_result == 'loss':
            algo_b_stats["losses"] += 1
            algo_b_stats["profit"] += algo_b_profit or 0
        else:
            algo_b_stats["pushes"] += 1

        # Update the snapshot with both algorithm results and total bet
        cur.execute('''
            UPDATE prediction_snapshots SET
                actual_winner = %s,
                home_score = %s,
                away_score = %s,
                closing_spread = %s,
                closing_total = %s,
                winner_correct = %s,
                best_bet_result = %s,
                best_bet_profit = %s,
                best_total_result = %s,
                best_total_profit = %s,
                algo_a_bet_result = %s,
                algo_b_bet_result = %s,
                algo_a_profit = %s,
                algo_b_profit = %s
            WHERE id = %s
        ''', (
            actual_winner, home_score, away_score,
            closing_spread, closing_total,
            winner_correct, bet_result, bet_profit,
            total_bet_result, total_bet_profit,
            algo_a_result, algo_b_result,
            algo_a_profit, algo_b_profit,
            pred_id
        ))

        graded += 1
        if bet_result == 'win':
            wins += 1
            total_profit += (bet_profit or 0)
        elif bet_result == 'loss':
            losses += 1
            total_profit += (bet_profit or 0)
        else:
            pushes += 1

        result_str = 'WIN' if winner_correct else 'LOSS'
        bet_str = f", Best Bet: {bet_result.upper()}" if bet_result else ""
        logger.info(f"Graded {game_id}: Winner {result_str}{bet_str}")

    cur.close()
    conn.close()

    return {
        "predictions_graded": graded,
        "best_bet_wins": wins,
        "best_bet_losses": losses,
        "best_bet_pushes": pushes,
        "total_profit": round(total_profit, 2),
        "algo_a_performance": algo_a_stats,
        "algo_b_performance": algo_b_stats,
        "status": "completed"
    }


def get_performance_summary(days: int = 7, db_url: str = None) -> dict:
    """
    Get performance summary for the model over recent days.

    Compares Algorithm A and Algorithm B performance.

    Args:
        days: Number of days to analyze
        db_url: Database connection string

    Returns:
        Performance metrics including per-algorithm comparison
    """
    conn = psycopg2.connect(db_url or DB_URL)
    cur = conn.cursor()

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    # Get all graded predictions with algorithm data
    cur.execute('''
        SELECT
            predicted_winner, actual_winner, winner_correct,
            best_bet_type, best_bet_team, best_bet_value_score,
            best_bet_result, best_bet_profit,
            snapshot_time,
            algo_a_value_score, algo_b_value_score,
            algo_a_bet_result, algo_b_bet_result,
            algo_a_profit, algo_b_profit
        FROM prediction_snapshots
        WHERE snapshot_time >= %s
        AND winner_correct IS NOT NULL
    ''', (cutoff,))

    predictions = cur.fetchall()
    cur.close()
    conn.close()

    if not predictions:
        return {
            "days_analyzed": days,
            "total_predictions": 0,
            "status": "no_data"
        }

    # Calculate winner accuracy
    total = len(predictions)
    winner_correct = sum(1 for p in predictions if p[2])

    # Calculate best bet performance
    best_bets = [p for p in predictions if p[6]]  # Has a bet result
    bet_wins = sum(1 for p in best_bets if p[6] == 'win')
    bet_losses = sum(1 for p in best_bets if p[6] == 'loss')
    total_profit = sum(p[7] or 0 for p in best_bets)

    # Algorithm A performance (columns: 9=algo_a_value, 11=algo_a_result, 13=algo_a_profit)
    algo_a_bets = [p for p in predictions if p[11]]  # Has algo_a result
    algo_a_wins = sum(1 for p in algo_a_bets if p[11] == 'win')
    algo_a_losses = sum(1 for p in algo_a_bets if p[11] == 'loss')
    algo_a_profit = sum(p[13] or 0 for p in algo_a_bets)

    # Algorithm B performance (columns: 10=algo_b_value, 12=algo_b_result, 14=algo_b_profit)
    algo_b_bets = [p for p in predictions if p[12]]  # Has algo_b result
    algo_b_wins = sum(1 for p in algo_b_bets if p[12] == 'win')
    algo_b_losses = sum(1 for p in algo_b_bets if p[12] == 'loss')
    algo_b_profit = sum(p[14] or 0 for p in algo_b_bets)

    # Performance by value score bucket (using best_bet_value_score)
    buckets = {}
    for p in best_bets:
        value_score = p[5] or 0
        if value_score >= 70:
            bucket = "70+"
        elif value_score >= 60:
            bucket = "60-69"
        elif value_score >= 50:
            bucket = "50-59"
        else:
            bucket = "<50"

        if bucket not in buckets:
            buckets[bucket] = {"wins": 0, "losses": 0, "profit": 0}

        if p[6] == 'win':
            buckets[bucket]["wins"] += 1
        elif p[6] == 'loss':
            buckets[bucket]["losses"] += 1
        buckets[bucket]["profit"] += (p[7] or 0)

    def calc_win_rate(wins, losses):
        return round(wins / (wins + losses) * 100, 1) if (wins + losses) > 0 else 0

    def calc_roi(profit, bets):
        return round(profit / (len(bets) * 100) * 100, 1) if bets else 0

    return {
        "days_analyzed": days,
        "total_predictions": total,
        "winner_accuracy": {
            "correct": winner_correct,
            "total": total,
            "percentage": round(winner_correct / total * 100, 1) if total > 0 else 0
        },
        "best_bet_performance": {
            "wins": bet_wins,
            "losses": bet_losses,
            "win_rate": calc_win_rate(bet_wins, bet_losses),
            "total_profit": round(total_profit, 2),
            "roi": calc_roi(total_profit, best_bets)
        },
        "algorithm_comparison": {
            "algo_a": {
                "wins": algo_a_wins,
                "losses": algo_a_losses,
                "win_rate": calc_win_rate(algo_a_wins, algo_a_losses),
                "profit": round(algo_a_profit, 2),
                "roi": calc_roi(algo_a_profit, algo_a_bets)
            },
            "algo_b": {
                "wins": algo_b_wins,
                "losses": algo_b_losses,
                "win_rate": calc_win_rate(algo_b_wins, algo_b_losses),
                "profit": round(algo_b_profit, 2),
                "roi": calc_roi(algo_b_profit, algo_b_bets)
            }
        },
        "by_value_bucket": buckets,
        "status": "success"
    }


def analyze_line_movement_performance(days: int = 30, db_url: str = None) -> dict:
    """
    Analyze how predictions perform based on line movement.

    Tracks:
    - Performance when betting WITH sharp money (following line movement)
    - Performance when betting AGAINST sharp money (fading line movement)
    - Performance by direction (sharp_home, sharp_away, etc.)

    Returns:
        Analysis of line movement correlation with results
    """
    conn = psycopg2.connect(db_url or DB_URL)
    cur = conn.cursor()

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    cur.execute('''
        SELECT
            best_bet_team, home_team, away_team,
            spread_movement, line_movement_direction,
            best_bet_result, best_bet_profit
        FROM prediction_snapshots
        WHERE snapshot_time >= %s
        AND best_bet_result IS NOT NULL
        AND spread_movement IS NOT NULL
    ''', (cutoff,))

    results = cur.fetchall()
    cur.close()
    conn.close()

    if not results:
        return {"status": "no_data", "days_analyzed": days}

    # Track performance by movement direction
    with_sharp = {"wins": 0, "losses": 0, "profit": 0}  # Betting same direction as line move
    against_sharp = {"wins": 0, "losses": 0, "profit": 0}  # Betting against line move
    by_direction = {}

    for row in results:
        bet_team, home, away, spread_move, direction, result, profit = row

        if not spread_move or abs(float(spread_move)) < 0.5:
            continue  # Ignore small movements

        # Determine if we bet WITH or AGAINST the line movement
        is_home_bet = (bet_team == home)
        line_moved_to_home = (float(spread_move) < 0)  # Negative = toward home

        if is_home_bet == line_moved_to_home:
            # We bet WITH the sharp money
            bucket = with_sharp
        else:
            # We bet AGAINST the sharp money
            bucket = against_sharp

        if result == 'win':
            bucket["wins"] += 1
            bucket["profit"] += float(profit or 0)
        elif result == 'loss':
            bucket["losses"] += 1
            bucket["profit"] += float(profit or 0)

        # Track by direction
        if direction:
            if direction not in by_direction:
                by_direction[direction] = {"wins": 0, "losses": 0, "profit": 0, "count": 0}
            by_direction[direction]["count"] += 1
            if result == 'win':
                by_direction[direction]["wins"] += 1
                by_direction[direction]["profit"] += float(profit or 0)
            elif result == 'loss':
                by_direction[direction]["losses"] += 1
                by_direction[direction]["profit"] += float(profit or 0)

    def calc_stats(bucket):
        total = bucket["wins"] + bucket["losses"]
        return {
            "wins": bucket["wins"],
            "losses": bucket["losses"],
            "win_rate": round(bucket["wins"] / total * 100, 1) if total > 0 else 0,
            "profit": round(bucket["profit"], 2),
            "roi": round(bucket["profit"] / (total * 100) * 100, 1) if total > 0 else 0,
        }

    return {
        "days_analyzed": days,
        "with_sharp_money": calc_stats(with_sharp),
        "against_sharp_money": calc_stats(against_sharp),
        "by_direction": {k: calc_stats(v) for k, v in by_direction.items()},
        "status": "success"
    }


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python prediction_tracker.py [snapshot|grade|summary]")
        sys.exit(1)

    command = sys.argv[1]

    if command == 'snapshot':
        hours = int(sys.argv[2]) if len(sys.argv) > 2 else 6
        result = snapshot_predictions(hours_ahead=hours)
        print(f"\nResult: {result}")

    elif command == 'grade':
        result = grade_predictions()
        print(f"\nResult: {result}")

    elif command == 'summary':
        days = int(sys.argv[2]) if len(sys.argv) > 2 else 7
        result = get_performance_summary(days=days)
        print(f"\nPerformance Summary ({result['days_analyzed']} days):")
        print(f"  Total Predictions: {result['total_predictions']}")
        if result['total_predictions'] > 0:
            wa = result['winner_accuracy']
            print(f"  Winner Accuracy: {wa['correct']}/{wa['total']} ({wa['percentage']}%)")
            bp = result['best_bet_performance']
            print(f"  Best Bet Record: {bp['wins']}-{bp['losses']} ({bp['win_rate']}%)")
            print(f"  Total Profit: ${bp['total_profit']:.2f}")
            print(f"  ROI: {bp['roi']}%")
            print(f"\n  By Value Bucket:")
            for bucket, stats in result.get('by_value_bucket', {}).items():
                print(f"    {bucket}: {stats['wins']}-{stats['losses']}, ${stats['profit']:.2f}")

    elif command == 'line-movement':
        days = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        result = analyze_line_movement_performance(days=days)
        print(f"\nLine Movement Analysis ({result['days_analyzed']} days):")

        if result['status'] == 'no_data':
            print("  No data available with line movement tracking")
        else:
            ws = result['with_sharp_money']
            print(f"\n  WITH Sharp Money (following line moves):")
            print(f"    Record: {ws['wins']}-{ws['losses']} ({ws['win_rate']}%)")
            print(f"    Profit: ${ws['profit']:.2f} (ROI: {ws['roi']}%)")

            ag = result['against_sharp_money']
            print(f"\n  AGAINST Sharp Money (fading line moves):")
            print(f"    Record: {ag['wins']}-{ag['losses']} ({ag['win_rate']}%)")
            print(f"    Profit: ${ag['profit']:.2f} (ROI: {ag['roi']}%)")

            if result.get('by_direction'):
                print(f"\n  By Direction:")
                for direction, stats in result['by_direction'].items():
                    print(f"    {direction}: {stats['wins']}-{stats['losses']} ({stats['win_rate']}%) ${stats['profit']:.2f}")

    else:
        print(f"Unknown command: {command}")
        print("Available commands: snapshot, grade, summary, line-movement")
        sys.exit(1)
