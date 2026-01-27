"""
Feature engineering utilities for totals model v3.

This module provides functions to build feature vectors for both:
1. Training (from historical game_results + team_stats)
2. Live prediction (from current game + team_stats + injuries)

Usage:
    from src.services.ml.totals_features import build_prediction_features

    features = await build_prediction_features(
        game_id='ABC123',
        home_team_id='LAL',
        away_team_id='GSW',
        game_date=date.today(),
    )
"""

import numpy as np
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple
import psycopg2
import structlog

from src.services.injuries import (
    get_all_team_injury_reports,
    ABBREV_TO_TEAM_ID,
    TEAM_ID_TO_ABBREV,
)

logger = structlog.get_logger()

DB_URL = 'postgresql://postgres:wzYHkiAOkykxiPitXKBIqPJxvifFtDPI@maglev.proxy.rlwy.net:46068/railway'


# Feature names (must match training order)
FEATURE_NAMES = [
    # Basic stats (8 features)
    'home_pace', 'away_pace', 'home_ortg', 'away_ortg',
    'home_drtg', 'away_drtg', 'home_ppg', 'away_ppg',
    # Pace interactions (4 features)
    'avg_pace', 'weighted_pace', 'pace_clash', 'pace_variance',
    # Rest/fatigue (5 features)
    'rest_advantage', 'total_fatigue', 'fatigue_asymmetry',
    'home_well_rested', 'away_well_rested',
    # Injury impact (5 features)
    'home_injury_score', 'away_injury_score', 'injury_asymmetry',
    'total_injury', 'injury_edge',
    # Scoring variance (2 features)
    'home_scoring_std', 'away_scoring_std',
    # Home/away splits (4 features)
    'home_home_win_pct', 'away_away_win_pct',
    'home_ortg_home_boost', 'away_ortg_away_penalty',
    # O/U tendencies (2 features)
    'home_ou_over_pct', 'away_ou_over_pct',
]


def calculate_pace_interaction(home_pace: float, away_pace: float) -> Dict[str, float]:
    """Calculate pace interaction features."""
    avg_pace = (home_pace + away_pace) / 2

    faster_pace = max(home_pace, away_pace)
    slower_pace = min(home_pace, away_pace)
    weighted_pace = faster_pace * 0.60 + slower_pace * 0.40

    pace_clash = abs(home_pace - away_pace)
    pace_variance = pace_clash / avg_pace if avg_pace > 0 else 0

    return {
        'avg_pace': avg_pace,
        'weighted_pace': weighted_pace,
        'pace_clash': pace_clash,
        'pace_variance': pace_variance,
    }


def calculate_rest_asymmetry(home_rest: int, away_rest: int,
                              home_b2b: bool, away_b2b: bool) -> Dict[str, float]:
    """Calculate rest/fatigue asymmetry features."""
    rest_advantage = home_rest - away_rest

    home_fatigue = 1.0 if home_b2b else 0.0
    away_fatigue = 1.0 if away_b2b else 0.0
    total_fatigue = home_fatigue + away_fatigue

    fatigue_asymmetry = abs(home_fatigue - away_fatigue)

    home_well_rested = 1 if home_rest >= 2 and not home_b2b else 0
    away_well_rested = 1 if away_rest >= 2 and not away_b2b else 0

    return {
        'rest_advantage': rest_advantage,
        'total_fatigue': total_fatigue,
        'fatigue_asymmetry': fatigue_asymmetry,
        'home_well_rested': home_well_rested,
        'away_well_rested': away_well_rested,
    }


def calculate_injury_impact(home_team_id: str, away_team_id: str,
                             injury_reports: dict) -> Dict[str, float]:
    """Calculate injury impact features for totals."""
    # Convert team IDs to abbreviations for injury lookup
    if home_team_id.isdigit():
        home_abbrev = TEAM_ID_TO_ABBREV.get(int(home_team_id), home_team_id)
    else:
        home_abbrev = home_team_id

    if away_team_id.isdigit():
        away_abbrev = TEAM_ID_TO_ABBREV.get(int(away_team_id), away_team_id)
    else:
        away_abbrev = away_team_id

    home_report = injury_reports.get(home_abbrev)
    away_report = injury_reports.get(away_abbrev)

    home_injury = home_report.totals_injury_score if home_report else 0.0
    away_injury = away_report.totals_injury_score if away_report else 0.0

    injury_asymmetry = abs(home_injury - away_injury)
    total_injury = home_injury + away_injury
    injury_edge = away_injury - home_injury

    return {
        'home_injury_score': home_injury,
        'away_injury_score': away_injury,
        'injury_asymmetry': injury_asymmetry,
        'total_injury': total_injury,
        'injury_edge': injury_edge,
    }


async def build_prediction_features(
    home_team_id: str,
    away_team_id: str,
    game_date: date,
    db_url: str = None,
) -> Optional[np.ndarray]:
    """
    Build feature vector for a single game prediction.

    Args:
        home_team_id: Home team ID (string like 'LAL' or numeric like '14')
        away_team_id: Away team ID
        game_date: Date of the game
        db_url: Database connection string

    Returns:
        Feature vector (35 features) or None if insufficient data
    """
    conn = psycopg2.connect(db_url or DB_URL)
    cur = conn.cursor()

    # Fetch team stats for both teams (most recent before game date)
    cur.execute('''
        SELECT
            -- Home team stats
            ts_h.pace_10, ts_h.pace_season,
            ts_h.ortg_10, ts_h.drtg_10, ts_h.ppg_10, ts_h.opp_ppg_10,
            ts_h.days_rest, ts_h.is_back_to_back,
            CASE WHEN (ts_h.ou_overs_l10 + ts_h.ou_unders_l10) > 0
                 THEN ts_h.ou_overs_l10::float / (ts_h.ou_overs_l10 + ts_h.ou_unders_l10)
                 ELSE 0.5 END as home_ou_over_pct,
            ts_h.home_wins, ts_h.home_losses,
            ts_h.away_wins, ts_h.away_losses,
            -- Away team stats
            ts_a.pace_10, ts_a.pace_season,
            ts_a.ortg_10, ts_a.drtg_10, ts_a.ppg_10, ts_a.opp_ppg_10,
            ts_a.days_rest, ts_a.is_back_to_back,
            CASE WHEN (ts_a.ou_overs_l10 + ts_a.ou_unders_l10) > 0
                 THEN ts_a.ou_overs_l10::float / (ts_a.ou_overs_l10 + ts_a.ou_unders_l10)
                 ELSE 0.5 END as away_ou_over_pct,
            ts_a.home_wins, ts_a.home_losses,
            ts_a.away_wins, ts_a.away_losses
        FROM team_stats ts_h
        CROSS JOIN team_stats ts_a
        WHERE ts_h.team_id = %s
        AND ts_a.team_id = %s
        AND ts_h.stat_date = (
            SELECT MAX(stat_date) FROM team_stats
            WHERE team_id = %s AND stat_date < %s
        )
        AND ts_a.stat_date = (
            SELECT MAX(stat_date) FROM team_stats
            WHERE team_id = %s AND stat_date < %s
        )
    ''', (home_team_id, away_team_id, home_team_id, game_date, away_team_id, game_date))

    row = cur.fetchone()

    if not row:
        logger.warning(f"No team stats found for {home_team_id} vs {away_team_id} on {game_date}")
        cur.close()
        conn.close()
        return None

    # Fetch recent scoring history for variance calculation
    cur.execute('''
        SELECT
            gr.home_score
        FROM game_results gr
        WHERE gr.home_team_id = %s
        AND gr.game_date < %s
        AND gr.home_score IS NOT NULL
        ORDER BY gr.game_date DESC
        LIMIT 5
    ''', (home_team_id, game_date))

    home_recent = [r[0] for r in cur.fetchall()]

    cur.execute('''
        SELECT
            gr.away_score
        FROM game_results gr
        WHERE gr.away_team_id = %s
        AND gr.game_date < %s
        AND gr.away_score IS NOT NULL
        ORDER BY gr.game_date DESC
        LIMIT 5
    ''', (away_team_id, game_date))

    away_recent = [r[0] for r in cur.fetchall()]

    cur.close()
    conn.close()

    # Fetch injury reports
    injury_reports = await get_all_team_injury_reports()

    # Extract features from database row
    home_pace = float(row[0]) if row[0] else 100.0
    home_pace_season = float(row[1]) if row[1] else home_pace
    home_ortg = float(row[2]) if row[2] else 110.0
    home_drtg = float(row[3]) if row[3] else 110.0
    home_ppg = float(row[4]) if row[4] else 110.0
    home_opp_ppg = float(row[5]) if row[5] else 110.0
    home_rest = int(row[6]) if row[6] else 1
    home_b2b = bool(row[7]) if row[7] is not None else False
    home_ou_pct = float(row[8]) if row[8] else 0.5

    home_home_wins = int(row[9]) if row[9] else 0
    home_home_losses = int(row[10]) if row[10] else 0
    home_away_wins = int(row[11]) if row[11] else 0
    home_away_losses = int(row[12]) if row[12] else 0

    away_pace = float(row[13]) if row[13] else 100.0
    away_pace_season = float(row[14]) if row[14] else away_pace
    away_ortg = float(row[15]) if row[15] else 110.0
    away_drtg = float(row[16]) if row[16] else 110.0
    away_ppg = float(row[17]) if row[17] else 110.0
    away_opp_ppg = float(row[18]) if row[18] else 110.0
    away_rest = int(row[19]) if row[19] else 1
    away_b2b = bool(row[20]) if row[20] is not None else False
    away_ou_pct = float(row[21]) if row[21] else 0.5

    away_home_wins = int(row[22]) if row[22] else 0
    away_home_losses = int(row[23]) if row[23] else 0
    away_away_wins = int(row[24]) if row[24] else 0
    away_away_losses = int(row[25]) if row[25] else 0

    # Calculate derived features
    pace_features = calculate_pace_interaction(home_pace, away_pace)
    rest_features = calculate_rest_asymmetry(home_rest, away_rest, home_b2b, away_b2b)
    injury_features = calculate_injury_impact(home_team_id, away_team_id, injury_reports)

    # Scoring variance
    home_scoring_std = np.std(home_recent) if len(home_recent) >= 3 else 10.0
    away_scoring_std = np.std(away_recent) if len(away_recent) >= 3 else 10.0

    # Home/away splits
    home_home_games = home_home_wins + home_home_losses
    home_home_win_pct = home_home_wins / home_home_games if home_home_games > 0 else 0.5

    away_away_games = away_away_wins + away_away_losses
    away_away_win_pct = away_away_wins / away_away_games if away_away_games > 0 else 0.5

    home_ortg_home_boost = 1 if home_home_win_pct > 0.55 else 0
    away_ortg_away_penalty = 1 if away_away_win_pct < 0.45 else 0

    # Build feature vector in exact order (must match training)
    features = [
        # Basic (8)
        home_pace, away_pace, home_ortg, away_ortg,
        home_drtg, away_drtg, home_ppg, away_ppg,
        # Pace interactions (4)
        pace_features['avg_pace'], pace_features['weighted_pace'],
        pace_features['pace_clash'], pace_features['pace_variance'],
        # Rest/fatigue (5)
        rest_features['rest_advantage'], rest_features['total_fatigue'],
        rest_features['fatigue_asymmetry'], rest_features['home_well_rested'],
        rest_features['away_well_rested'],
        # Injury (5)
        injury_features['home_injury_score'], injury_features['away_injury_score'],
        injury_features['injury_asymmetry'], injury_features['total_injury'],
        injury_features['injury_edge'],
        # Variance (2)
        home_scoring_std, away_scoring_std,
        # Home/away (4)
        home_home_win_pct, away_away_win_pct,
        home_ortg_home_boost, away_ortg_away_penalty,
        # O/U (2)
        home_ou_pct, away_ou_pct,
    ]

    return np.array(features)


async def build_bulk_prediction_features(
    game_list: List[Tuple[str, str, date]],
    db_url: str = None,
) -> Dict[Tuple[str, str, date], Optional[np.ndarray]]:
    """
    Build feature vectors for multiple games at once (more efficient).

    Args:
        game_list: List of (home_team_id, away_team_id, game_date) tuples
        db_url: Database connection string

    Returns:
        Dict mapping (home, away, date) to feature vector (or None if insufficient data)
    """
    # Fetch injury reports once
    injury_reports = await get_all_team_injury_reports()

    results = {}
    for home_id, away_id, game_date in game_list:
        features = await build_prediction_features(
            home_id, away_id, game_date, db_url
        )
        results[(home_id, away_id, game_date)] = features

    return results


def validate_features(features: np.ndarray) -> bool:
    """
    Validate feature vector has correct shape and no invalid values.

    Args:
        features: Feature vector to validate

    Returns:
        True if valid, False otherwise
    """
    if features is None:
        return False

    if len(features) != len(FEATURE_NAMES):
        logger.error(f"Feature vector has {len(features)} features, expected {len(FEATURE_NAMES)}")
        return False

    if np.any(np.isnan(features)):
        logger.error("Feature vector contains NaN values")
        return False

    if np.any(np.isinf(features)):
        logger.error("Feature vector contains infinite values")
        return False

    return True


def get_feature_dict(features: np.ndarray) -> Dict[str, float]:
    """
    Convert feature vector to labeled dictionary for debugging.

    Args:
        features: Feature vector

    Returns:
        Dict mapping feature name to value
    """
    if len(features) != len(FEATURE_NAMES):
        raise ValueError(f"Feature vector has {len(features)} features, expected {len(FEATURE_NAMES)}")

    return dict(zip(FEATURE_NAMES, features))
