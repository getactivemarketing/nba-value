"""A starter's 'opponents scoreless in the 1st' record.

The stat is the OPPOSING side's first-inning runs: for a home start that is
`away_first_inning_runs`, for an away start `home_first_inning_runs`. Getting
this backwards would narrate a pitcher's own offence as his run prevention.
"""
from src.services.mlb.first_inning import (
    FirstInningSplit,
    StarterAppearance,
    starter_first_inning_split,
)


class TestStarterFirstInningSplit:
    def test_counts_scoreless_first_innings(self):
        apps = [
            StarterAppearance("2026-07-01", opponent_first_inning_runs=0),
            StarterAppearance("2026-07-06", opponent_first_inning_runs=2),
            StarterAppearance("2026-07-11", opponent_first_inning_runs=0),
        ]
        assert starter_first_inning_split(apps, as_of="2026-08-08") == FirstInningSplit(2, 3)

    def test_excludes_games_on_or_after_as_of(self):
        apps = [
            StarterAppearance("2026-07-01", opponent_first_inning_runs=0),
            StarterAppearance("2026-08-08", opponent_first_inning_runs=0),
        ]
        assert starter_first_inning_split(apps, as_of="2026-08-08") == FirstInningSplit(1, 1)

    def test_appearances_with_unknown_runs_are_dropped_not_counted_scoreless(self):
        """None means 'not recorded'. Counting it as zero invents a stat."""
        apps = [
            StarterAppearance("2026-07-01", opponent_first_inning_runs=0),
            StarterAppearance("2026-07-06", opponent_first_inning_runs=None),
        ]
        assert starter_first_inning_split(apps, as_of="2026-08-08") == FirstInningSplit(1, 1)

    def test_no_usable_appearances_is_unmeasured(self):
        apps = [StarterAppearance("2026-07-01", opponent_first_inning_runs=None)]
        assert starter_first_inning_split(apps, as_of="2026-08-08") is None

    def test_empty_is_unmeasured(self):
        assert starter_first_inning_split([], as_of="2026-08-08") is None
