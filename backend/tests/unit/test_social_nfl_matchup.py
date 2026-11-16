from src.edge import matchups
from src.edge.social import nfl_matchup as nm


def _t(key, direction):
    return {"key": key, "direction": direction, "detail": ""}


def test_def_soft_tailwind_is_favorable():
    assert nm.favorable({"tags": [_t("def_soft", "tailwind"), _t("usage_up", "tailwind")]})


def test_any_headwind_blocks():
    assert not nm.favorable({"tags": [_t("def_soft", "tailwind"), _t("questionable", "headwind")]})
    assert not nm.favorable({"tags": [_t("def_soft", "headwind")]})


def test_no_tags_not_favorable():
    assert not nm.favorable({"tags": []})
    assert not nm.favorable({})
    assert not nm.favorable({"tags": [_t("usage_up", "tailwind")]})


def _pg(week, carries, team_carries):
    return matchups.PlayerGame(2026, week, "KC", "joe rb", "Joe RB", "RB", carries, 0, 0, 0,
                               team_carries, 30, 0, team_carries, 30)


def test_panel_reads_rank_and_share_like_tags_for():
    rows = [_pg(w, 10, 20) for w in (1, 2, 3, 4)] + [_pg(5, 20, 20)]
    ctx = matchups.TagContext(
        def_rank={("DEN", "RB"): {"yds": (130.0, 2), "rz": (1.0, 5), "td": (1.0, 5)}},
        players={"joe rb": [("KC", "RB")]}, player_rows={("KC", "joe rb"): rows},
        injuries={}, season=2026, week=6, n_ranked={"RB": 32})
    row = {"player": "Joe RB", "home_team": "Denver Broncos", "away_team": "Kansas City Chiefs",
           "tags": [_t("def_soft", "tailwind")]}
    p = nm.panel(row, ctx)
    assert p.favorable and p.opp_rank == 2 and p.tags == ["def_soft"]
    assert p.share_l4 == (10 + 10 + 10 + 20) / 80 and p.snap_share is None


def test_panel_unknown_player_is_all_none():
    ctx = matchups.TagContext({}, {}, {}, {})
    p = nm.panel({"player": "Nobody", "home_team": "Denver Broncos", "away_team": "Kansas City Chiefs",
                  "tags": []}, ctx)
    assert (p.favorable, p.opp_rank, p.share_l4, p.snap_share) == (False, None, None, None)
