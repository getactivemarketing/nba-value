"""Report which models are serving, how old they are, and what has degraded.

Written after a night in which every serious defect reported success while
doing nothing: pitcher ingest logged count=0 as success for the life of the
system, team rate stats were never fetched, 24 of 28 features served
league-average constants, 111 Athletics games were skipped on a name lookup,
and the run-diff model sat six months stale without a word.

Nothing asserted what should be true. This answers three questions the system
could not previously answer about itself:

  which model is serving?   how old is it?   is a fallback quietly engaged?

Deliberately pessimistic. Unknown reads as stale, absent reads as degraded — a
monitor that resolves ambiguity in its own favour is how silent failure
survives.

Pure functions; no I/O.
"""

from __future__ import annotations

from datetime import date, datetime

# In-season, a model older than this has stopped tracking the league it is
# predicting. Set against the incumbent's six-month drift, which nothing
# flagged; a shorter window would have caught it in March.
STALE_AFTER_DAYS = 45


def _parse_day(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value)).date()
    except ValueError:
        try:
            return date.fromisoformat(str(value)[:10])
        except ValueError:
            return None


def describe_artifact(
    artifact: dict | None,
    path: str,
    as_of: str | date | None = None,
) -> dict:
    """Summarise one model artifact.

    A missing `trained_at` counts as stale rather than fresh: mlb_totals_v2
    shipped without one, and treating unknown age as acceptable would hide
    exactly the drift this exists to surface.
    """
    today = _parse_day(as_of) if not isinstance(as_of, date) else as_of
    today = today or date.today()

    if artifact is None:
        return {
            "loaded": False,
            "path": path,
            "version": None,
            "trained_at": None,
            "age_days": None,
            "features": None,
            "stale": True,
        }

    trained = _parse_day(artifact.get("trained_at"))
    age = (today - trained).days if trained else None

    return {
        "loaded": True,
        "path": path,
        "version": artifact.get("version"),
        "trained_at": artifact.get("trained_at"),
        "age_days": age,
        "features": len(artifact.get("feature_cols") or []) or None,
        "stale": age is None or age > STALE_AFTER_DAYS,
    }


def summarize_models(
    run_diff: dict | None,
    totals: dict | None,
    run_diff_path: str,
    totals_path: str,
    as_of: str | date | None = None,
) -> dict:
    """Overall model health, with the specific reasons for any degradation."""
    models = {
        "run_diff": describe_artifact(run_diff, run_diff_path, as_of),
        "totals": describe_artifact(totals, totals_path, as_of),
    }

    degraded: list[str] = []

    # A missing run-diff model does not stop scoring — MLBScorer silently falls
    # back to _estimate_run_diff(), a hand-written heuristic with hardcoded
    # coefficients. Picks keep flowing and look normal. This is the single most
    # dangerous quiet failure in the pipeline.
    if not models["run_diff"]["loaded"]:
        degraded.append("run_diff model not loaded — serving the heuristic fallback")
    elif models["run_diff"]["stale"]:
        age = models["run_diff"]["age_days"]
        degraded.append(
            f"run_diff model stale ({age} days)" if age is not None
            else "run_diff model has no trained_at"
        )

    if not models["totals"]["loaded"]:
        degraded.append("totals model not loaded")
    elif models["totals"]["stale"]:
        age = models["totals"]["age_days"]
        degraded.append(
            f"totals model stale ({age} days)" if age is not None
            else "totals model has no trained_at"
        )

    return {
        "status": "degraded" if degraded else "ok",
        "degraded": degraded,
        "models": models,
    }
