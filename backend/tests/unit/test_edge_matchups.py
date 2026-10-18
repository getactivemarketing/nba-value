import pandas as pd

from src.edge.matchups import POSITIONS, PBP_COLS, PlayerGame, TeamGame, aggregate, window_games

ROSTER = pd.DataFrame([
    ("g_qb", "Pat Q. Back", "QB", "KC", 2024), ("g_rb", "Rudy Runner", "RB", "KC", 2024),
    ("g_rb2", "Ray Backup", "RB", "KC", 2024), ("g_fb", "Fred Fullback", "FB", "KC", 2024),
    ("g_wr", "Will Receiver", "WR", "KC", 2024), ("g_te", "Tim Tightend", "TE", "KC", 2024),
    ("g_k", "Kyle Kicker", "K", "KC", 2024),
], columns=["gsis_id", "full_name", "position", "team", "season"])


def play(**kw):
    base = dict(season=2024, week=1, season_type="REG", game_id="g1", posteam="KC", defteam="LA",
                play_type="run", two_point_attempt=0, yardline_100=50.0, rusher_player_id=None,
                receiver_player_id=None, passer_player_id=None, rushing_yards=float("nan"),
                receiving_yards=float("nan"), passing_yards=float("nan"), rush_touchdown=0,
                pass_touchdown=0, td_player_id=None)
    base.update(kw)
    return base


def frame(*plays):
    return pd.DataFrame(list(plays), columns=list(PBP_COLS))


def team(tg, pos, week=1, name="LAR"):
    return next(t for t in tg if t.team == name and t.week == week and t.pos == pos)


def test_rb_yards_allowed_counts_rush_and_receiving_by_rb():
    pbp = frame(play(rusher_player_id="g_rb", rushing_yards=10),
                play(play_type="pass", receiver_player_id="g_rb", receiving_yards=7, passer_player_id="g_qb", passing_yards=7),
                play(play_type="pass", receiver_player_id="g_wr", receiving_yards=20, passer_player_id="g_qb", passing_yards=20))
    tg, _ = aggregate(pbp, ROSTER)
    assert team(tg, "RB").yds == 17
    assert team(tg, "WR").yds == 20
    assert team(tg, "QB").yds == 27
    assert team(tg, "RB").team == "LAR"  # mapped to BallDontLie code


def test_red_zone_is_yardline_20_or_less():
    pbp = frame(play(rusher_player_id="g_rb", rushing_yards=1, yardline_100=20.0, rush_touchdown=1),
                play(rusher_player_id="g_rb", rushing_yards=1, yardline_100=21.0),
                play(play_type="pass", receiver_player_id="g_rb", receiving_yards=3, yardline_100=5.0))
    tg, pg = aggregate(pbp, ROSTER)
    rb = team(tg, "RB")
    assert rb.rz_opps == 2 and rb.tds == 1
    p = next(p for p in pg if p.player_key == "rudy runner")
    assert (p.rz_carries, p.rz_targets, p.team_rz_opps) == (1, 1, 2)


def test_two_point_tries_and_postseason_never_count():
    pbp = frame(play(rusher_player_id="g_rb", rushing_yards=2, two_point_attempt=1),
                play(rusher_player_id="g_rb", rushing_yards=9, season_type="POST", game_id="g9", week=19),
                play(rusher_player_id="g_rb", rushing_yards=4))
    tg, pg = aggregate(pbp, ROSTER)
    assert team(tg, "RB").yds == 4
    assert not any(t.week == 19 for t in tg)
    assert pg[0].carries == 1


def test_fb_is_rb_and_k_is_excluded_from_players_but_counts_to_team_totals():
    pbp = frame(play(rusher_player_id="g_fb", rushing_yards=3),
                play(play_type="pass", receiver_player_id="g_k", receiving_yards=8, yardline_100=10.0),
                play(rusher_player_id="g_unknown", rushing_yards=5))
    tg, pg = aggregate(pbp, ROSTER)
    assert team(tg, "RB").yds == 3
    assert [p.player_key for p in pg] == ["fred fullback"]
    p = pg[0]
    assert p.pos == "RB" and p.carries == 1
    assert p.team_carries == 2 and p.team_targets == 1 and p.team_rz_opps == 1


def test_every_game_gets_four_position_rows_with_zeros():
    pbp = frame(play(rusher_player_id="g_rb", rushing_yards=3),
                play(game_id="g2", week=2, posteam="LA", defteam="KC", rusher_player_id=None))
    tg, _ = aggregate(pbp, ROSTER)
    assert len(tg) == 8
    assert {t.pos for t in tg} == set(POSITIONS)
    assert all(t.yds == 0 and t.rz_opps == 0 and t.tds == 0 for t in tg if t.week == 2)
    assert sorted((t.team, t.week) for t in tg if t.pos == "QB") == [("KC", 2), ("LAR", 1)]


def test_player_shares_denominators():
    pbp = frame(play(rusher_player_id="g_rb", rushing_yards=3),
                play(rusher_player_id="g_rb", rushing_yards=3, yardline_100=8.0),
                play(rusher_player_id="g_rb2", rushing_yards=1),
                play(rusher_player_id="g_qb", rushing_yards=2),
                play(play_type="pass", receiver_player_id="g_rb", receiving_yards=4),
                play(play_type="pass", receiver_player_id="g_wr", receiving_yards=4, yardline_100=3.0),
                play(play_type="pass", receiver_player_id="g_unknown", receiving_yards=4, yardline_100=3.0))
    _, pg = aggregate(pbp, ROSTER)
    rb = next(p for p in pg if p.player_key == "rudy runner")
    assert (rb.carries, rb.targets, rb.rz_carries, rb.rz_targets) == (2, 1, 1, 0)
    assert (rb.team_carries, rb.team_targets, rb.team_rz_opps) == (4, 3, 3)
    assert (rb.pos_carries, rb.pos_targets) == (3, 1)
    assert (rb.team, rb.player_name, rb.pos) == ("KC", "Rudy Runner", "RB")
    assert {p.player_key for p in pg} == {"rudy runner", "ray backup", "pat q back", "will receiver"}


def test_nan_yards_are_zero_and_missing_play_type_ignored():
    pbp = frame(play(rusher_player_id="g_rb"), play(play_type=None, rusher_player_id="g_rb", rushing_yards=50))
    tg, pg = aggregate(pbp, ROSTER)
    assert team(tg, "RB").yds == 0 and pg[0].carries == 1


def rows(season, weeks):
    return [TeamGame(season, w, "X", "RB", 0.0, 0, 0) for w in weeks]


def test_window_counts_games_not_weeks_and_skips_byes():
    r = rows(2024, [1, 2, 4, 5, 6])
    out = window_games(r, 2024, 7, 3, lambda x: True)
    assert [x.week for x in out] == [4, 5, 6]


def test_window_week1_fills_from_last_season():
    r = rows(2023, range(1, 18)) + rows(2024, [1])
    out = window_games(r, 2024, 1, 8, lambda x: True)
    assert [(x.season, x.week) for x in out] == [(2023, w) for w in range(10, 18)]


def test_window_never_includes_the_requested_week():
    r = rows(2024, [1, 2, 3, 4, 5])
    out = window_games(r, 2024, 3, 8, lambda x: True)
    assert [x.week for x in out] == [1, 2]


def test_window_fills_only_when_fewer_than_four_current_rows():
    r = rows(2023, range(1, 18)) + rows(2024, [1, 2, 3])
    out = window_games(r, 2024, 4, 6, lambda x: True)
    assert [(x.season, x.week) for x in out] == [(2023, 15), (2023, 16), (2023, 17), (2024, 1), (2024, 2), (2024, 3)]
    r = rows(2023, range(1, 18)) + rows(2024, [1, 2, 3, 4])
    out = window_games(r, 2024, 5, 6, lambda x: True)
    assert [(x.season, x.week) for x in out] == [(2024, w) for w in (1, 2, 3, 4)]


def test_window_applies_key_predicate():
    r = rows(2024, [1, 2, 3])
    assert [x.week for x in window_games(r, 2024, 4, 8, lambda x: x.week != 2)] == [1, 3]
