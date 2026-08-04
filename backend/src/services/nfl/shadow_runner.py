"""Enumerate market candidates and score every registered shadow model.

Design constraints, and why each exists:

EVERY CANDIDATE IS STORED. Not only the side a model likes, not only
qualifying edges, not only enabled markets. `mlb_markets` kept just the current
line, so MLB's historical candidate set is gone — no backtest there can ask
"what else was on the board", and no selection experiment is possible, ever.

EVERY MODEL SCORES EVERY CANDIDATE, on identical inputs. Comparing models on
separately-collected samples compares the samples as much as the models.

A MODEL THAT CANNOT SCORE STILL WRITES A ROW, carrying a structured status.
Omitting it makes "the model failed" indistinguishable from "the candidate
never existed" — precisely the ambiguity that let 111 Athletics games vanish
while the scheduler logged success.

IDEMPOTENT, YET EVERY MARKET UPDATE PRESERVED. These pull against each other,
so the identity of a row is its candidate CONTENT, never its timestamp:
re-scoring unchanged prices collides and is a no-op; a moved line or shaded
juice hashes differently and becomes a new row.

Pure functions; no I/O, no database.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any


class ShadowStatus:
    """Why a row does or does not carry a prediction.

    Statuses are values, not free text, so coverage can be aggregated and a
    systematic failure shows up as a count rather than a gap.
    """

    OK = "ok"
    MISSING_FEATURES = "missing_features"        # prior-week stats/lines absent
    MISSING_CALIBRATION = "missing_calibration"  # no resid_std -> no probability
    MODEL_DECLINED = "model_declined"            # model returned None by choice
    MODEL_ERROR = "model_error"                  # model raised


def enumerate_candidates(game_id: str, quote: dict) -> list[dict]:
    """Every bettable side this book is offering, as separate candidates.

    A market appears only when BOTH sides are priced: a one-sided quote cannot
    be devigged, so it is not a usable candidate and recording it would imply
    a comparison we cannot make.
    """
    book = quote.get("book")
    out: list[dict] = []

    def add(market_type: str, side: str, line, odds):
        out.append({
            "game_id": game_id, "book": book, "market_type": market_type,
            "side": side, "line": line, "odds_decimal": odds,
        })

    ml_h, ml_a = quote.get("ml_home"), quote.get("ml_away")
    if ml_h and ml_a:
        add("moneyline", "home", None, float(ml_h))
        add("moneyline", "away", None, float(ml_a))

    sl, sh, sa = quote.get("spread_line"), quote.get("spread_home_odds"), quote.get("spread_away_odds")
    if sl is not None and sh and sa:
        add("spread", "home", float(sl), float(sh))
        add("spread", "away", -float(sl), float(sa))   # away line mirrors home

    tl, ov, un = quote.get("total_line"), quote.get("over_odds"), quote.get("under_odds")
    if tl is not None and ov and un:
        add("total", "over", float(tl), float(ov))
        add("total", "under", float(tl), float(un))

    return out


def candidate_hash(candidate: dict) -> str:
    """Stable identity for one priced side at one book.

    Includes line AND price so a moved line or shaded juice produces a new row
    rather than colliding with the old one. Uses an explicit sentinel for a
    null line because Postgres treats NULLs as distinct in unique indexes,
    which would silently break idempotency for moneylines.
    """
    # Canonicalised before hashing: -3.50 and -3.5 are the same line, -0.0 and
    # 0.0 the same pick'em, 1.9100000001 and 1.91 the same price. Without this,
    # float noise silently splits one candidate into two rows and idempotency
    # stops working without any error.
    from src.services.nfl.settlement import canonical_line, canonical_price

    parts = [
        str(candidate.get("game_id")),
        str(candidate.get("book")),
        str(candidate.get("market_type")),
        str(candidate.get("side")),
        canonical_line(candidate.get("line")),
        canonical_price(candidate.get("odds_decimal")),
    ]
    return hashlib.sha1("|".join(parts).encode()).hexdigest()


def _side_probability(predicted_value: float, candidate: dict, resid_std: float) -> float:
    """P(this side wins) under Normal(predicted_value, resid_std).

    `predicted_value` is the model's margin (home - away) for spread/moneyline,
    or total points for totals. The line is already stored from this side's
    perspective, so the comparison is direct.
    """
    market, side, line = candidate["market_type"], candidate["side"], candidate.get("line")

    if market == "moneyline":
        threshold, want_over = 0.0, (side == "home")
    elif market == "spread":
        # Home line -3.5 means home must win by >3.5; the away row carries +3.5
        # and its own side wins when the margin stays BELOW that mirrored line.
        threshold = float(line) if side == "home" else -float(line)
        want_over = (side == "home")
    else:
        threshold, want_over = float(line), (side == "over")

    z = (predicted_value - threshold) / resid_std
    p_over = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    return p_over if want_over else 1.0 - p_over


def score_candidate(
    model: Any,
    candidate: dict,
    features: dict | None,
    resid_std: float | None,
) -> dict:
    """Score one candidate with one model. ALWAYS returns a row.

    Failure is reported as a status, never as a missing row — a gap is
    indistinguishable from a candidate that never existed, and a model that
    silently stops scoring should show up as a rising count rather than a
    quietly shrinking denominator.
    """
    row = {
        **candidate,
        "model_key": getattr(model, "key", "unknown"),
        "model_version": getattr(model, "version", None),
        "resid_std": resid_std,
        "predicted_value": None,
        "probability": None,
        "status": ShadowStatus.OK,
        "status_detail": None,
    }

    if features is None:
        row["status"] = ShadowStatus.MISSING_FEATURES
        return row

    try:
        value = model.predict_value(candidate, features)
    except Exception as exc:            # one model must not cost the others
        row["status"] = ShadowStatus.MODEL_ERROR
        row["status_detail"] = str(exc)[:200]
        return row

    if value is None:
        row["status"] = ShadowStatus.MODEL_DECLINED
        return row

    row["predicted_value"] = float(value)

    if not resid_std or resid_std <= 0:
        row["status"] = ShadowStatus.MISSING_CALIBRATION
        return row

    row["probability"] = _side_probability(float(value), candidate, float(resid_std))
    return row
