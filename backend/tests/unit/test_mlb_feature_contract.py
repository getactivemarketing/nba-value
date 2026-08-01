"""One canonical map from model feature name -> value, shared by train and serve.

The vector builder hardcoded 28 values in fixed positional order while the
training builder emitted the dataclass's own field names. Twelve of the 28
serving names did not exist in the training output at all (home_era vs
home_team_era, home_avg vs home_batting_avg, home_starter_k9 vs
home_starter_k_rate, and so on). The mapping existed only as the ORDER of
expressions inside _build_model_feature_vector.

That is what made deploying a differently-shaped model dangerous: nothing
checked that the artifact's feature_cols matched what serving actually built,
so a mismatch would either raise on shape or — worse — silently feed the wrong
number into the wrong slot.
"""
import datetime

import pytest

from src.services.mlb.features import MLBGameFeatures
from src.services.mlb.scorer import MLBScorer, FeatureContractError


def features(**kw) -> MLBGameFeatures:
    f = MLBGameFeatures(
        game_id="g", game_date=datetime.date(2026, 5, 1),
        home_team="HOM", away_team="AWY",
    )
    for k, v in kw.items():
        setattr(f, k, v)
    return f


class TestModelFeatureDict:
    def test_covers_every_declared_model_feature(self):
        got = features().model_feature_dict()
        missing = [n for n in MLBScorer.MODEL_FEATURE_NAMES if n not in got]
        assert missing == [], f"unmapped model features: {missing}"

    def test_maps_renamed_fields_to_their_model_names(self):
        f = features(
            home_batting_avg=0.271, home_team_era=3.42, home_team_whip=1.11,
            home_starter_k_rate=10.5, home_starter_bb_rate=2.1,
        )
        got = f.model_feature_dict()
        assert got["home_avg"] == 0.271
        assert got["home_era"] == 3.42
        assert got["home_whip"] == 1.11
        assert got["home_starter_k9"] == 10.5
        assert got["home_starter_bb9"] == 2.1

    def test_team_era_diff_is_away_minus_home(self):
        got = features(home_team_era=3.0, away_team_era=5.0).model_feature_dict()
        assert got["team_era_diff"] == pytest.approx(2.0)

    def test_defaults_fill_missing_values(self):
        got = features().model_feature_dict()
        assert got["home_era"] == 4.00
        assert got["home_starter_whip"] == 1.25

    def test_new_context_features_are_exposed(self):
        f = features(temperature=88, is_dome=True, weather_factor=1.04,
                     home_last_10_win_pct=0.6, away_last_10_win_pct=0.4)
        got = f.model_feature_dict()
        assert got["temperature"] == 88
        assert got["is_dome"] == 1.0
        assert got["weather_factor"] == pytest.approx(1.04)
        assert got["home_last_10_win_pct"] == pytest.approx(0.6)

    def test_is_dome_is_numeric_not_boolean(self):
        """LightGBM needs a number; a bare bool would train as an object column."""
        got = features(is_dome=False).model_feature_dict()
        assert got["is_dome"] == 0.0
        assert isinstance(got["is_dome"], float)


class TestVectorBuiltByName:
    def test_vector_follows_the_artifact_column_order(self):
        f = features(home_team_era=3.0, away_team_era=5.0)
        scorer = object.__new__(MLBScorer)
        forward = scorer._build_model_feature_vector(f, ["home_era", "away_era"])
        reverse = scorer._build_model_feature_vector(f, ["away_era", "home_era"])
        assert list(forward[0]) == [3.0, 5.0]
        assert list(reverse[0]) == [5.0, 3.0]

    def test_unknown_feature_raises_instead_of_guessing(self):
        scorer = object.__new__(MLBScorer)
        with pytest.raises(FeatureContractError):
            scorer._build_model_feature_vector(features(), ["home_era", "not_a_feature"])

    def test_defaults_to_the_declared_model_features(self):
        scorer = object.__new__(MLBScorer)
        vec = scorer._build_model_feature_vector(features())
        assert vec.shape == (1, len(MLBScorer.MODEL_FEATURE_NAMES))
