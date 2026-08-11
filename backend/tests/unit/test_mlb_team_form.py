"""Current win/loss streak, for the 'case against' beat."""
from src.services.mlb.team_form import GameResult, Streak, current_streak


class TestCurrentStreak:
    def test_counts_consecutive_losses_from_most_recent(self):
        results = [
            GameResult("2026-08-05", won=True),
            GameResult("2026-08-06", won=False),
            GameResult("2026-08-07", won=False),
        ]
        assert current_streak(results, as_of="2026-08-08") == Streak("lost", 2)

    def test_counts_consecutive_wins(self):
        results = [
            GameResult("2026-08-05", won=False),
            GameResult("2026-08-06", won=True),
            GameResult("2026-08-07", won=True),
        ]
        assert current_streak(results, as_of="2026-08-08") == Streak("won", 2)

    def test_input_order_does_not_matter(self):
        """Callers must not be able to get this wrong by passing rows unsorted."""
        results = [
            GameResult("2026-08-07", won=False),
            GameResult("2026-08-05", won=True),
            GameResult("2026-08-06", won=False),
        ]
        assert current_streak(results, as_of="2026-08-08") == Streak("lost", 2)

    def test_excludes_games_on_or_after_as_of(self):
        """Point-in-time: a game must never appear in its own inputs."""
        results = [
            GameResult("2026-08-06", won=False),
            GameResult("2026-08-07", won=False),
            GameResult("2026-08-08", won=True),
        ]
        assert current_streak(results, as_of="2026-08-08") == Streak("lost", 2)

    def test_no_prior_games_is_unmeasured(self):
        assert current_streak([], as_of="2026-08-08") is None

    def test_single_game_is_a_streak_of_one(self):
        results = [GameResult("2026-08-07", won=True)]
        assert current_streak(results, as_of="2026-08-08") == Streak("won", 1)
