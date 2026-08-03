"""The registered shadow models. All score identical inputs, none can bet.

  market_only   the book's own number as the forecast — the baseline every
                model must beat before anything downstream is examined
  incumbent     nfl_mov_v1 / nfl_totals_v1, trained 2010-2023, random-split
                resid_std (understated 11.6% on MOV)
  challenger    rolling-origin shadow, trained 2010-2024, OOF resid_std
  v2            reserved

`market_only` is not a formality. On 1,537 out-of-fold games it beat BOTH
models on MAE (9.917 vs 10.942 MOV; 10.413 vs 11.214 totals). A model that
cannot out-predict the posted number has nothing to sell.

resid_std comes from each artifact and is recorded per row, because it IS the
calibration and the whole point of the rolling-origin refactor was that this
number had been measured wrong.
"""

from __future__ import annotations

import joblib
from pathlib import Path

import structlog

logger = structlog.get_logger()

MARKET_ONLY = "market_only"
INCUMBENT = "incumbent"
CHALLENGER = "challenger"


class MarketOnlyModel:
    """Forecasts exactly what the book posted. The bar, not a strawman."""

    key = MARKET_ONLY
    version = "baseline"
    resid_std_mov = None      # supplied by the caller from the challenger's OOF
    resid_std_total = None

    def predict_value(self, candidate: dict, features: dict) -> float | None:
        line = candidate.get("line")
        if candidate["market_type"] == "moneyline":
            # A moneyline carries no line; the market's implied margin is the
            # spread, which this candidate does not have. Declining is honest —
            # inventing 0.0 would score a coin flip as a forecast.
            return None
        if line is None:
            return None
        # Spread lines are stored from the candidate's own side; the market's
        # margin forecast is the HOME line, so mirror the away row back.
        if candidate["market_type"] == "spread":
            return float(line) if candidate["side"] == "home" else -float(line)
        return float(line)


class BundleModel:
    """A trained LightGBM bundle (incumbent or challenger)."""

    def __init__(self, key: str, mov_path: str, totals_path: str):
        self.key = key
        self._mov = self._load(mov_path)
        self._tot = self._load(totals_path)
        self.version = (self._mov or {}).get("version") or Path(mov_path).stem

    @staticmethod
    def _load(path: str) -> dict | None:
        p = Path(path)
        if not p.exists():
            logger.warning("nfl_shadow_bundle_missing", path=path)
            return None
        return joblib.load(p)

    def bundle_for(self, market_type: str) -> dict | None:
        return self._tot if market_type == "total" else self._mov

    def resid_std_for(self, market_type: str) -> float | None:
        b = self.bundle_for(market_type)
        return None if b is None else float(b.get("resid_std") or 0) or None

    def predict_value(self, candidate: dict, features: dict) -> float | None:
        bundle = self.bundle_for(candidate["market_type"])
        if bundle is None:
            return None
        cols = bundle["feature_cols"]
        missing = [c for c in cols if c not in features]
        if missing:
            # Surfaces as MODEL_DECLINED rather than a silent default-filled
            # prediction. MLB served league-average constants for 24 of 28
            # features for a season precisely because defaults looked normal.
            logger.debug("nfl_shadow_missing_cols", model=self.key, missing=missing[:5])
            return None
        row = [[float(features[c]) if features[c] is not None else 0.0 for c in cols]]
        return float(bundle["model"].predict(row)[0])


def build_registry(
    incumbent_mov: str = "models/nfl_mov_v1.joblib",
    incumbent_tot: str = "models/nfl_totals_v1.joblib",
    challenger_mov: str = "models/nfl_mov_shadow_2025.1.joblib",
    challenger_tot: str = "models/nfl_totals_shadow_2025.1.joblib",
) -> list:
    """Every model that scores. Order is display-only; all are peers."""
    return [
        MarketOnlyModel(),
        BundleModel(INCUMBENT, incumbent_mov, incumbent_tot),
        BundleModel(CHALLENGER, challenger_mov, challenger_tot),
    ]


def resid_std_for(model, market_type: str, fallback: float | None = None) -> float | None:
    """Residual scale this model should use for this market.

    market_only has no residual of its own — it is a point forecast, not a
    fitted model — so it borrows the challenger's OUT-OF-FOLD scale. That
    keeps the probability comparison about the forecasts rather than about two
    different uncertainty assumptions.
    """
    getter = getattr(model, "resid_std_for", None)
    if getter is not None:
        return getter(market_type)
    return fallback
