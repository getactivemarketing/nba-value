import pandas as pd
import pytest

from src.services.nfl.qb_ratings import (
    shrink, qb_game_dropback_epa, build_qb_timelines, rating_as_of,
    REPLACEMENT_EPA, PRIOR_DROPBACKS,
)


def test_shrink_pulls_low_sample_to_replacement_and_high_sample_to_observed():
    assert shrink(0.0, 0) == REPLACEMENT_EPA                 # 0 dropbacks -> replacement
    assert abs(shrink(2.0, 5) - REPLACEMENT_EPA) < 0.02      # tiny sample -> near replacement
    assert abs(shrink(0.20 * 5000, 5000) - 0.20) < 0.02      # huge sample -> observed
    assert shrink(30.0, 300, k=100) > shrink(30.0, 300, k=500)  # more prior -> pulled harder


def _pbp():
    # 2 dropbacks QB A (epa +1,+2), 1 non-dropback (ignored), 1 dropback QB B (epa -1)
    return pd.DataFrame([
        {"passer_player_id": "A", "season": 2022, "week": 1, "posteam": "KC", "qb_dropback": 1, "qb_epa": 1.0},
        {"passer_player_id": "A", "season": 2022, "week": 1, "posteam": "KC", "qb_dropback": 1, "qb_epa": 2.0},
        {"passer_player_id": None, "season": 2022, "week": 1, "posteam": "KC", "qb_dropback": 0, "qb_epa": 5.0},
        {"passer_player_id": "B", "season": 2022, "week": 1, "posteam": "CIN", "qb_dropback": 1, "qb_epa": -1.0},
    ])


def test_qb_game_dropback_epa_aggregates_dropbacks_only():
    g = qb_game_dropback_epa(_pbp()).set_index("passer_player_id")
    assert g.loc["A", "dropbacks"] == 2 and g.loc["A", "epa_sum"] == 3.0
    assert g.loc["A", "team"] == "KC"
    assert g.loc["B", "dropbacks"] == 1 and g.loc["B", "epa_sum"] == -1.0
    assert g.loc["B", "team"] == "CIN"


def test_rating_as_of_is_point_in_time_and_career_cumulative():
    # QB A: (2022,wk1) 100 db @ +0.30 (30 epa); (2022,wk3) 100 db @ +0.99.
    qge = pd.DataFrame([
        {"passer_player_id": "A", "season": 2022, "week": 1, "team": "KC", "dropbacks": 100, "epa_sum": 30.0},
        {"passer_player_id": "A", "season": 2022, "week": 3, "team": "KC", "dropbacks": 100, "epa_sum": 99.0},
    ])
    tl = build_qb_timelines(qge)
    # entering wk1: zero prior -> replacement (no self-leak)
    assert rating_as_of(tl, "A", 2022, 1) == REPLACEMENT_EPA
    # entering wk2 (A didn't play wk2): only wk1 counts
    assert abs(rating_as_of(tl, "A", 2022, 2) - shrink(30.0, 100)) < 1e-9
    # entering wk3: strictly-before excludes wk3's own 99 -> still only wk1
    assert abs(rating_as_of(tl, "A", 2022, 3) - shrink(30.0, 100)) < 1e-9
    # next season: BOTH games now in the past
    assert abs(rating_as_of(tl, "A", 2023, 1) - shrink(30.0 + 99.0, 200)) < 1e-9
    # unknown QB -> replacement
    assert rating_as_of(tl, "NOBODY", 2022, 5) == REPLACEMENT_EPA


@pytest.mark.integration
def test_nflverse_pbp_has_qb_columns():
    # Confirms the real column names this module assumes actually exist.
    from src.services.nfl.nfl_data import load_pbp
    pbp = load_pbp([2023])
    for col in ("passer_player_id", "qb_dropback", "qb_epa", "posteam", "season", "week"):
        assert col in pbp.columns, col
    g = qb_game_dropback_epa(pbp)
    assert len(g) > 0 and g["dropbacks"].sum() > 1000


from src.services.nfl.qb_ratings import compute_qb_deltas


def _career_pbp(rows):
    # rows: list of (qb, season, week, team, n_dropbacks, epa_each)
    out = []
    for qb, s, w, t, n, e in rows:
        for _ in range(n):
            out.append({"passer_player_id": qb, "season": s, "week": w,
                        "posteam": t, "qb_dropback": 1, "qb_epa": e})
    return pd.DataFrame(out)


def test_qb_delta_exactly_zero_when_same_starter_and_form_qb():
    # STAR is KC's only QB weeks 1-3; COOL is CIN's only QB weeks 1-3.
    pbp = _career_pbp([
        ("STAR", 2022, 1, "KC", 40, 0.3), ("STAR", 2022, 2, "KC", 40, 0.3),
        ("COOL", 2022, 1, "CIN", 40, 0.1), ("COOL", 2022, 2, "CIN", 40, 0.1),
        ("STAR", 2022, 3, "KC", 40, 0.3), ("COOL", 2022, 3, "CIN", 40, 0.1),
    ])
    games = pd.DataFrame([{"game_id": "g3", "season": 2022, "week": 3,
                           "home_team": "KC", "away_team": "CIN",
                           "home_qb_id": "STAR", "away_qb_id": "COOL"}])
    d = compute_qb_deltas(pbp, games).set_index("game_id")
    # each side: starter == form_qb -> (starter-form)=0 exactly -> delta == 0
    assert abs(d.loc["g3", "qb_delta"]) < 1e-12


def test_qb_delta_positive_when_home_upgrades_from_backup_to_star():
    pbp = _career_pbp([
        # STAR: long strong history in 2021 -> high career rating by 2022
        ("STAR", 2021, 1, "KC", 500, 0.30),
        # BACKUP starts KC weeks 1-2 (weak); STAR returns wk3
        ("BACKUP", 2022, 1, "KC", 40, -0.15), ("BACKUP", 2022, 2, "KC", 40, -0.15),
        ("COOL", 2022, 1, "CIN", 40, 0.05), ("COOL", 2022, 2, "CIN", 40, 0.05),
        ("STAR", 2022, 3, "KC", 40, 0.30), ("COOL", 2022, 3, "CIN", 40, 0.05),
    ])
    games = pd.DataFrame([{"game_id": "g3", "season": 2022, "week": 3,
                           "home_team": "KC", "away_team": "CIN",
                           "home_qb_id": "STAR", "away_qb_id": "COOL"}])
    d = compute_qb_deltas(pbp, games).set_index("game_id")
    # KC: starter STAR (high) - form_qb BACKUP (low) >> 0; CIN stable ~0 -> delta > 0
    assert d.loc["g3", "qb_delta"] > 0.15
    # symmetry: swap home/away -> sign flips
    d2 = compute_qb_deltas(pbp, games.assign(
        home_team="CIN", away_team="KC", home_qb_id="COOL", away_qb_id="STAR")
    ).set_index("game_id")
    assert d2.loc["g3", "qb_delta"] < -0.15
