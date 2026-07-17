"""NFL per-bet value. Machinery mirrors MLBValueCalculator verbatim; the
qualification gate is NFL-calibrated (a floor AND a ceiling, plus a blowup cap)
because NFL totals profit lives in a mid edge band, not above a floor. Thresholds
come from config (tuned in the Phase-3 backtest)."""
import math
from dataclasses import dataclass

from src.config import settings

EDGE_SCALE_FACTOR = 4.0
MARKET_REGRESSION_WEIGHT = 0.50
DISPLAY_TANH_SCALE = 20.0
NFL_MAX_EDGE_PCT = 80.0  # blowup cap (same as MLB)


@dataclass
class NFLValueResult:
    market_type: str
    bet_type: str
    team: str | None
    line: float | None
    model_prob: float
    market_prob: float
    raw_edge: float
    edge_pct: float
    value_score: float
    confidence: str
    odds_decimal: float
    is_value_bet: bool
    sort_score: float = 0.0


class NFLValueCalculator:
    @classmethod
    def calculate_value(cls, market_type, bet_type, model_prob, market_prob,
                        odds_decimal, team=None, line=None, model_confidence=0.5):
        raw_edge = model_prob - market_prob
        edge_pct = (raw_edge / market_prob) * 100 if market_prob > 0 else 0.0
        confidence_multiplier = 0.8 + (model_confidence * 0.4)
        market_multiplier = 1.0
        if market_type == "moneyline":
            market_multiplier = 0.95
        elif market_type == "total":
            market_multiplier = 0.90

        gate_score = edge_pct * EDGE_SCALE_FACTOR * confidence_multiplier * market_multiplier
        if model_prob > 0.65 and raw_edge > 0.03:
            gate_score += 5
        gate_score = max(0, min(100, gate_score))

        sort_score = edge_pct * confidence_multiplier * market_multiplier

        blended_edge_pct = edge_pct * (1.0 - MARKET_REGRESSION_WEIGHT)
        value_score = 100.0 * math.tanh(blended_edge_pct / DISPLAY_TANH_SCALE) \
            * confidence_multiplier * market_multiplier
        adjusted_model_prob = market_prob + raw_edge * (1.0 - MARKET_REGRESSION_WEIGHT)
        if adjusted_model_prob > 0.65 and raw_edge > 0.03:
            value_score += 5
        value_score = max(0, min(100, value_score))

        if raw_edge >= 0.08 and model_confidence >= 0.6:
            confidence = "high"
        elif raw_edge >= 0.04 and model_confidence >= 0.4:
            confidence = "medium"
        else:
            confidence = "low"

        is_value = (
            gate_score >= settings.nfl_moderate_threshold
            and raw_edge >= settings.nfl_min_edge
            and raw_edge <= settings.nfl_max_edge   # NFL edge CEILING
            and edge_pct <= NFL_MAX_EDGE_PCT
        )
        return NFLValueResult(
            market_type=market_type, bet_type=bet_type, team=team, line=line,
            model_prob=model_prob, market_prob=market_prob, raw_edge=raw_edge,
            edge_pct=edge_pct, value_score=round(value_score, 1), confidence=confidence,
            odds_decimal=odds_decimal, is_value_bet=is_value, sort_score=round(sort_score, 2))

    @classmethod
    def find_best_value(cls, values):
        vb = [v for v in values if v.is_value_bet]
        return max(vb, key=lambda v: v.sort_score) if vb else None

    @classmethod
    def find_best_bet(cls, values, enabled_market_types):
        return cls.find_best_value([v for v in values if v.market_type in enabled_market_types])

    @staticmethod
    def devig_two_way(odds1, odds2):
        p1, p2 = 1 / odds1, 1 / odds2
        total = p1 + p2
        return p1 / total, p2 / total
