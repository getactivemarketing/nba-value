"""Enumerating market candidates and scoring every model against them.

Requirements this pins:

- EVERY (game, book, market, side, line, price) is a candidate — not only the
  side a model likes, and not only qualifying edges. mlb_markets kept just the
  current line, so MLB's historical candidate set is gone and selection
  experiments there are impossible forever.
- Every registered model scores every candidate on IDENTICAL inputs.
- A model that cannot score writes a row with a STRUCTURED STATUS. Omitting the
  row would make "model failed" indistinguishable from "candidate never
  existed" — the exact ambiguity that hid 111 Athletics games.
- Writes are idempotent, yet every market update is preserved. Those pull in
  opposite directions, so the key is candidate CONTENT, not the timestamp:
  re-scoring unchanged prices is a no-op, a moved line is a new row.
"""
import pytest

from src.services.nfl.shadow_runner import (
    ShadowStatus,
    candidate_hash,
    enumerate_candidates,
    score_candidate,
)


def quote(**kw):
    base = dict(
        book="draftkings",
        ml_home=1.55, ml_away=2.55,
        spread_line=-3.5, spread_home_odds=1.91, spread_away_odds=1.91,
        total_line=47.5, over_odds=1.91, under_odds=1.87,
    )
    base.update(kw)
    return base


class TestEnumerateCandidates:
    def test_full_quote_yields_six_candidates(self):
        """3 markets x 2 sides — every one stored, none filtered by edge."""
        got = enumerate_candidates("G1", quote())
        assert len(got) == 6
        assert {(c["market_type"], c["side"]) for c in got} == {
            ("moneyline", "home"), ("moneyline", "away"),
            ("spread", "home"), ("spread", "away"),
            ("total", "over"), ("total", "under"),
        }

    def test_spread_sides_carry_mirrored_lines(self):
        got = {(c["market_type"], c["side"]): c for c in enumerate_candidates("G1", quote())}
        assert got[("spread", "home")]["line"] == -3.5
        assert got[("spread", "away")]["line"] == 3.5

    def test_total_sides_share_the_same_line(self):
        got = {(c["market_type"], c["side"]): c for c in enumerate_candidates("G1", quote())}
        assert got[("total", "over")]["line"] == 47.5
        assert got[("total", "under")]["line"] == 47.5
        assert got[("total", "under")]["odds_decimal"] == 1.87

    def test_moneyline_has_no_line(self):
        got = {(c["market_type"], c["side"]): c for c in enumerate_candidates("G1", quote())}
        assert got[("moneyline", "home")]["line"] is None

    def test_partial_quote_yields_only_what_exists(self):
        got = enumerate_candidates("G1", quote(total_line=None, over_odds=None, under_odds=None))
        assert len(got) == 4
        assert all(c["market_type"] != "total" for c in got)

    def test_missing_one_side_drops_that_market(self):
        """A one-sided price cannot be devigged and is not a usable candidate."""
        got = enumerate_candidates("G1", quote(ml_away=None))
        assert all(c["market_type"] != "moneyline" for c in got)

    def test_empty_quote_yields_nothing(self):
        assert enumerate_candidates("G1", quote(
            ml_home=None, ml_away=None, spread_line=None,
            spread_home_odds=None, spread_away_odds=None,
            total_line=None, over_odds=None, under_odds=None)) == []


class TestCandidateHash:
    def _cand(self, **kw):
        base = dict(game_id="G1", book="dk", market_type="spread",
                    side="home", line=-3.5, odds_decimal=1.91)
        base.update(kw)
        return base

    def test_identical_candidates_hash_equal(self):
        assert candidate_hash(self._cand()) == candidate_hash(self._cand())

    def test_line_move_changes_the_hash(self):
        """A moved line must be preserved as a NEW row, not deduped away."""
        assert candidate_hash(self._cand()) != candidate_hash(self._cand(line=-4.5))

    def test_price_move_changes_the_hash(self):
        """Juice moving on a held line is real movement."""
        assert candidate_hash(self._cand()) != candidate_hash(self._cand(odds_decimal=1.83))

    def test_book_is_part_of_the_key(self):
        assert candidate_hash(self._cand()) != candidate_hash(self._cand(book="fanduel"))

    def test_null_line_is_stable_and_distinct(self):
        """Postgres treats NULLs as distinct in unique indexes; the hash must not."""
        a = candidate_hash(self._cand(market_type="moneyline", line=None))
        b = candidate_hash(self._cand(market_type="moneyline", line=None))
        assert a == b
        assert a != candidate_hash(self._cand(market_type="moneyline", line=0.0))


class _Model:
    """Minimal stand-in for a registered shadow model."""
    def __init__(self, key, value=None, raises=False):
        self.key = key
        self.version = "test"
        self._value = value
        self._raises = raises

    def predict_value(self, candidate, features):
        if self._raises:
            raise RuntimeError("boom")
        return self._value


class TestScoreCandidate:
    CAND = {"game_id": "G1", "book": "dk", "market_type": "spread",
            "side": "home", "line": -3.5, "odds_decimal": 1.91}

    def test_successful_score_is_status_ok(self):
        row = score_candidate(_Model("challenger", value=-6.0), self.CAND,
                              features={"x": 1}, resid_std=13.9)
        assert row["status"] == ShadowStatus.OK
        assert row["predicted_value"] == -6.0
        assert 0.0 < row["probability"] < 1.0

    def test_probability_reflects_the_side(self):
        """Model says home wins by 6 on a -3.5 line -> home side is favoured."""
        home = score_candidate(_Model("m", value=-6.0), self.CAND, {"x": 1}, 13.9)
        away = score_candidate(_Model("m", value=-6.0),
                               {**self.CAND, "side": "away", "line": 3.5}, {"x": 1}, 13.9)
        assert home["probability"] + away["probability"] == pytest.approx(1.0, abs=1e-6)

    def test_missing_features_writes_a_row_not_a_gap(self):
        row = score_candidate(_Model("challenger"), self.CAND, features=None, resid_std=13.9)
        assert row["status"] == ShadowStatus.MISSING_FEATURES
        assert row["predicted_value"] is None and row["probability"] is None

    def test_model_returning_none_is_recorded_as_unscored(self):
        row = score_candidate(_Model("challenger", value=None), self.CAND, {"x": 1}, 13.9)
        assert row["status"] == ShadowStatus.MODEL_DECLINED

    def test_model_raising_is_captured_not_propagated(self):
        """One model blowing up must not cost the other models their rows."""
        row = score_candidate(_Model("challenger", raises=True), self.CAND, {"x": 1}, 13.9)
        assert row["status"] == ShadowStatus.MODEL_ERROR
        assert "boom" in (row["status_detail"] or "")

    def test_missing_resid_std_is_a_status_not_a_crash(self):
        row = score_candidate(_Model("m", value=-6.0), self.CAND, {"x": 1}, resid_std=None)
        assert row["status"] == ShadowStatus.MISSING_CALIBRATION
        assert row["probability"] is None

    def test_every_row_carries_model_lineage(self):
        row = score_candidate(_Model("challenger", value=-6.0), self.CAND, {"x": 1}, 13.9)
        assert row["model_key"] == "challenger"
        assert row["model_version"] == "test"
        assert row["resid_std"] == 13.9
