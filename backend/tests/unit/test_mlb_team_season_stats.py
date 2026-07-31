"""Parsing team hitting/pitching stats out of the MLB Stats API payload.

These six fields — ops, avg, obp, slg, era, whip — had 0% coverage in
mlb_team_stats because update_team_stats only ever read the standings endpoint,
which carries runs and records but no rate stats. The model was trained on them
with real variance (home_era std 0.494, home_ops std 0.034) and served a
constant for every game in the league.

MLB returns these as STRINGS, sometimes with a leading dot (".233"), sometimes
absent, sometimes literally "-.--" or "*.**" for undefined rates. Parsing has
to survive all of that without poisoning a feature with a bogus number.
"""
import pytest

from src.services.mlb.mlb_api import parse_team_season_stats


def payload(hitting: dict | None = None, pitching: dict | None = None) -> dict:
    stats = []
    if hitting is not None:
        stats.append({"group": {"displayName": "hitting"}, "splits": [{"stat": hitting}]})
    if pitching is not None:
        stats.append({"group": {"displayName": "pitching"}, "splits": [{"stat": pitching}]})
    return {"stats": stats}


class TestParseTeamSeasonStats:
    def test_parses_a_full_payload(self):
        got = parse_team_season_stats(payload(
            hitting={"avg": ".233", "obp": ".311", "slg": ".413", "ops": ".724"},
            pitching={"era": "3.31", "whip": "1.18"},
        ))
        assert got == {
            "batting_avg": 0.233,
            "obp": 0.311,
            "slg": 0.413,
            "ops": 0.724,
            "era": 3.31,
            "team_whip": 1.18,
        }

    def test_leading_dot_values_parse(self):
        """MLB writes rate stats as '.233', not '0.233'."""
        got = parse_team_season_stats(payload(hitting={"avg": ".233"}))
        assert got["batting_avg"] == pytest.approx(0.233)

    def test_missing_group_yields_none_not_a_default(self):
        """A default would be indistinguishable from a real league-average team."""
        got = parse_team_season_stats(payload(hitting={"ops": ".724"}))
        assert got["ops"] == pytest.approx(0.724)
        assert got["era"] is None
        assert got["team_whip"] is None

    def test_undefined_rate_placeholders_are_none(self):
        """MLB emits '-.--' / '*.**' for undefined rates (e.g. 0 innings)."""
        got = parse_team_season_stats(payload(pitching={"era": "-.--", "whip": "*.**"}))
        assert got["era"] is None
        assert got["team_whip"] is None

    def test_infinity_era_is_none(self):
        assert parse_team_season_stats(payload(pitching={"era": "INF"}))["era"] is None

    def test_empty_payload_is_all_none(self):
        got = parse_team_season_stats({"stats": []})
        assert set(got.values()) == {None}

    def test_missing_splits_does_not_raise(self):
        got = parse_team_season_stats(
            {"stats": [{"group": {"displayName": "hitting"}, "splits": []}]}
        )
        assert got["ops"] is None

    def test_numeric_values_pass_through(self):
        """Defensive: the API occasionally returns numbers rather than strings."""
        got = parse_team_season_stats(payload(pitching={"era": 3.31, "whip": 1.18}))
        assert got["era"] == pytest.approx(3.31)
        assert got["team_whip"] == pytest.approx(1.18)

    def test_group_name_is_matched_case_insensitively(self):
        got = parse_team_season_stats(
            {"stats": [{"group": {"displayName": "Hitting"}, "splits": [{"stat": {"ops": ".700"}}]}]}
        )
        assert got["ops"] == pytest.approx(0.700)
