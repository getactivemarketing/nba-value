"""Pitcher quality scoring for MLB model features."""

from dataclasses import dataclass
from typing import Any


@dataclass
class PitcherMetrics:
    """Container for pitcher metrics used in quality scoring."""
    era: float | None
    whip: float | None
    k_per_9: float | None
    bb_per_9: float | None
    fip: float | None = None
    innings_pitched: float | None = None


class PitcherQualityScorer:
    """
    Calculate pitcher quality scores for MLB model.

    Quality score is a composite metric (0-100) that combines:
    - ERA (lower is better)
    - WHIP (lower is better)
    - K/9 (higher is better)
    - BB/9 (lower is better)
    - FIP (lower is better, if available)

    League average benchmarks (2024 season):
    - ERA: 4.00
    - WHIP: 1.25
    - K/9: 8.5
    - BB/9: 3.2
    - FIP: 4.00
    """

    # League average benchmarks
    LEAGUE_AVG_ERA = 4.00
    LEAGUE_AVG_WHIP = 1.25
    LEAGUE_AVG_K9 = 8.5
    LEAGUE_AVG_BB9 = 3.2
    LEAGUE_AVG_FIP = 4.00

    # Weight distribution for quality score
    WEIGHT_ERA = 0.30
    WEIGHT_WHIP = 0.20
    WEIGHT_K9 = 0.20
    WEIGHT_BB9 = 0.15
    WEIGHT_FIP = 0.15

    @classmethod
    def calculate_quality_score(
        cls,
        era: float | None = None,
        whip: float | None = None,
        k_per_9: float | None = None,
        bb_per_9: float | None = None,
        fip: float | None = None,
    ) -> float:
        """
        Calculate composite pitcher quality score (0-100).

        Args:
            era: Earned Run Average
            whip: Walks + Hits per Inning Pitched
            k_per_9: Strikeouts per 9 innings
            bb_per_9: Walks per 9 innings
            fip: Fielding Independent Pitching

        Returns:
            Quality score 0-100 (higher = better)
        """
        components = []
        total_weight = 0.0

        # ERA component (inverted - lower is better)
        if era is not None:
            era_score = cls._normalize_inverse(era, cls.LEAGUE_AVG_ERA, min_val=1.5, max_val=7.0)
            components.append(era_score * cls.WEIGHT_ERA)
            total_weight += cls.WEIGHT_ERA

        # WHIP component (inverted)
        if whip is not None:
            whip_score = cls._normalize_inverse(whip, cls.LEAGUE_AVG_WHIP, min_val=0.80, max_val=1.70)
            components.append(whip_score * cls.WEIGHT_WHIP)
            total_weight += cls.WEIGHT_WHIP

        # K/9 component (direct - higher is better)
        if k_per_9 is not None:
            k9_score = cls._normalize_direct(k_per_9, cls.LEAGUE_AVG_K9, min_val=4.0, max_val=13.0)
            components.append(k9_score * cls.WEIGHT_K9)
            total_weight += cls.WEIGHT_K9

        # BB/9 component (inverted)
        if bb_per_9 is not None:
            bb9_score = cls._normalize_inverse(bb_per_9, cls.LEAGUE_AVG_BB9, min_val=1.0, max_val=5.5)
            components.append(bb9_score * cls.WEIGHT_BB9)
            total_weight += cls.WEIGHT_BB9

        # FIP component (inverted)
        if fip is not None:
            fip_score = cls._normalize_inverse(fip, cls.LEAGUE_AVG_FIP, min_val=2.0, max_val=6.0)
            components.append(fip_score * cls.WEIGHT_FIP)
            total_weight += cls.WEIGHT_FIP

        if total_weight == 0:
            return 50.0  # Return average if no data

        # Calculate weighted average and scale to 0-100
        raw_score = sum(components) / total_weight
        return round(raw_score * 100, 2)

    @staticmethod
    def _normalize_inverse(value: float, avg: float, min_val: float, max_val: float) -> float:
        """
        Normalize an inverse metric (lower is better) to 0-1 scale.

        Args:
            value: The metric value
            avg: League average
            min_val: Best realistic value (elite)
            max_val: Worst realistic value

        Returns:
            Normalized score 0-1 (1 = elite, 0 = poor)
        """
        # Clamp to realistic range
        value = max(min_val, min(max_val, value))

        # Linear interpolation (inverted)
        return 1.0 - (value - min_val) / (max_val - min_val)

    @staticmethod
    def _normalize_direct(value: float, avg: float, min_val: float, max_val: float) -> float:
        """
        Normalize a direct metric (higher is better) to 0-1 scale.

        Args:
            value: The metric value
            avg: League average
            min_val: Worst realistic value
            max_val: Best realistic value (elite)

        Returns:
            Normalized score 0-1 (1 = elite, 0 = poor)
        """
        # Clamp to realistic range
        value = max(min_val, min(max_val, value))

        # Linear interpolation
        return (value - min_val) / (max_val - min_val)

    @classmethod
    def compare_starters(
        cls,
        home_metrics: PitcherMetrics,
        away_metrics: PitcherMetrics,
    ) -> dict[str, float]:
        """
        Compare two starting pitchers.

        Args:
            home_metrics: Home starter metrics
            away_metrics: Away starter metrics

        Returns:
            Dict with comparison metrics
        """
        home_quality = cls.calculate_quality_score(
            era=home_metrics.era,
            whip=home_metrics.whip,
            k_per_9=home_metrics.k_per_9,
            bb_per_9=home_metrics.bb_per_9,
            fip=home_metrics.fip,
        )

        away_quality = cls.calculate_quality_score(
            era=away_metrics.era,
            whip=away_metrics.whip,
            k_per_9=away_metrics.k_per_9,
            bb_per_9=away_metrics.bb_per_9,
            fip=away_metrics.fip,
        )

        # Quality difference (positive = home advantage)
        quality_diff = home_quality - away_quality

        # ERA difference (negative = home advantage, as lower ERA is better)
        era_diff = 0.0
        if home_metrics.era is not None and away_metrics.era is not None:
            era_diff = away_metrics.era - home_metrics.era

        # K/9 difference (positive = home has more strikeouts)
        k9_diff = 0.0
        if home_metrics.k_per_9 is not None and away_metrics.k_per_9 is not None:
            k9_diff = home_metrics.k_per_9 - away_metrics.k_per_9

        return {
            "home_quality_score": home_quality,
            "away_quality_score": away_quality,
            "quality_difference": quality_diff,
            "era_difference": era_diff,
            "k9_difference": k9_diff,
            "home_advantage": quality_diff > 5,  # Significant advantage threshold
        }

    @classmethod
    def get_tier(cls, quality_score: float) -> str:
        """
        Get pitcher tier label based on quality score.

        Args:
            quality_score: Quality score 0-100

        Returns:
            Tier label
        """
        if quality_score >= 80:
            return "elite"
        elif quality_score >= 65:
            return "above_average"
        elif quality_score >= 45:
            return "average"
        elif quality_score >= 30:
            return "below_average"
        else:
            return "poor"

    @classmethod
    def estimate_runs_allowed(
        cls,
        quality_score: float,
        park_factor: float = 1.0,
        innings: float = 6.0,
    ) -> float:
        """
        Estimate runs allowed based on pitcher quality.

        This is a rough approximation for feature engineering.

        Args:
            quality_score: Pitcher quality score 0-100
            park_factor: Park run factor (1.0 = neutral)
            innings: Expected innings to pitch

        Returns:
            Estimated runs allowed
        """
        # Convert quality score to approximate ERA
        # Quality 100 -> ERA ~2.0, Quality 50 -> ERA ~4.0, Quality 0 -> ERA ~6.0
        estimated_era = 6.0 - (quality_score / 100.0 * 4.0)

        # Calculate runs per inning
        runs_per_inning = estimated_era / 9.0

        # Apply park factor and innings
        return runs_per_inning * innings * park_factor
