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


def test_null_gsis_roster_row_never_matches_null_passer():
    ros = pd.concat([ROSTER, pd.DataFrame([(None, "Ghost QB", "QB", "KC", 2024)], columns=ROSTER.columns)],
                    ignore_index=True)
    pbp = frame(play(play_type="pass", receiver_player_id="g_wr", receiving_yards=20, passer_player_id=None, passing_yards=20),
                play(play_type="pass", receiver_player_id="g_wr", receiving_yards=5, passer_player_id="g_qb", passing_yards=5))
    tg, _ = aggregate(pbp, ros)
    assert team(tg, "QB").yds == 5


def test_nan_full_name_roster_row_does_not_raise():
    ros = ROSTER.copy()
    ros.loc[ros.gsis_id == "g_rb", "full_name"] = float("nan")
    pbp = frame(play(rusher_player_id="g_rb", rushing_yards=3), play(rusher_player_id="g_rb2", rushing_yards=2))
    _, pg = aggregate(pbp, ros)
    assert [p.player_key for p in pg] == ["ray backup"]
    assert pg[0].team_carries == 2


# ---- Task 5: the seven tags -------------------------------------------------

from src.edge.config import TD_MARKET  # noqa: E402
from src.edge.matchups import (  # noqa: E402
    MARKETS_FOR, POLARITY, TAG_KEYS, Tag, TagContext, applies, build_context, tags_for)

# 32 defenses; TEAMS[r - 1] has rank r (1 = most allowed). JAX is rank 3.
TEAMS = [f"D{i:02d}" for i in range(1, 33)]
TEAMS[2] = "JAX"
S, W = 2024, 9  # the game being tagged


def defense(pos="RB", games=8, yds=lambda r: 121 - r, rz=lambda r: 40 - r, td=lambda r: 40 - r,
            teams=TEAMS, season=S):
    return [TeamGame(season, w, t, pos, float(yds(r)), int(rz(r)), int(td(r)))
            for r, t in enumerate(teams, start=1) for w in range(1, games + 1)]


def pg(week, name="Rudy Runner", team="KC", pos="RB", season=S, carries=5, targets=0,
       rz_carries=0, rz_targets=0, team_carries=20, team_targets=30, team_rz_opps=20,
       pos_carries=20, pos_targets=10):
    from src.edge.players import normalize_name
    return PlayerGame(season, week, team, normalize_name(name), name, pos, carries, targets,
                      rz_carries, rz_targets, team_carries, team_targets, team_rz_opps,
                      pos_carries, pos_targets)


def runner(weeks=range(1, 9), **kw):
    return [pg(w, **kw) for w in weeks]


def keys(tags):
    return {t.key for t in tags}


def ctx_vs(opp_rank, team_games=None, players=None, injuries=None):
    tg = defense() if team_games is None else team_games
    return build_context(tg, runner() if players is None else players, injuries or {}, S, W), TEAMS[opp_rank - 1]


def test_tag_constants():
    assert TAG_KEYS == ("def_soft", "def_tough", "def_rz", "def_td", "usage_up", "rz_role",
                        "teammate_out", "questionable")
    assert POLARITY == {k: (-1 if k in ("def_tough", "questionable") else 1) for k in TAG_KEYS}
    assert MARKETS_FOR["def_td"] == frozenset({TD_MARKET})
    assert MARKETS_FOR["rz_role"] == frozenset({TD_MARKET})
    assert MARKETS_FOR["def_rz"] == frozenset({TD_MARKET, "player_rush_yds", "player_reception_yds"})
    assert "player_pass_yds" in MARKETS_FOR["def_soft"] and TD_MARKET not in MARKETS_FOR["def_soft"]
    assert "player_pass_yds" not in MARKETS_FOR["usage_up"] and TD_MARKET in MARKETS_FOR["usage_up"]
    assert "player_pass_yds" in MARKETS_FOR["questionable"] and TD_MARKET in MARKETS_FOR["questionable"]


def test_applies_table():
    assert applies("def_soft", "RB", "player_rush_reception_yds")
    assert not applies("def_soft", "RB", TD_MARKET)
    assert applies("def_rz", "WR", "player_reception_yds") and not applies("def_rz", "WR", "player_receptions")
    assert not applies("def_rz", "QB", TD_MARKET) and not applies("def_td", "QB", TD_MARKET)
    assert applies("usage_up", "TE", TD_MARKET) and not applies("usage_up", "QB", "player_pass_yds")
    assert applies("questionable", "QB", TD_MARKET) and applies("questionable", "K", "player_kicking_points")
    assert not applies("questionable", "RB", "h2h") and not applies("def_soft", "K", "player_rush_yds")


# Defense ranks

def test_def_soft_fires_at_rank_8_not_9():
    ctx, opp = ctx_vs(8)
    assert "def_soft" in keys(tags_for(ctx, "Rudy Runner", ("KC", opp), "player_rush_yds", "over"))
    ctx, opp = ctx_vs(9)
    assert "def_soft" not in keys(tags_for(ctx, "Rudy Runner", ("KC", opp), "player_rush_yds", "over"))


def test_def_tough_fires_at_rank_25_not_24():
    ctx, opp = ctx_vs(25)
    tags = tags_for(ctx, "Rudy Runner", (opp, "KC"), "player_rush_yds", "over")
    tough = [t for t in tags if t.key == "def_tough"]
    assert tough and tough[0].detail == f"{opp} allows 96 RB yds per game (8th fewest)"
    ctx, opp = ctx_vs(24)
    assert not keys(tags_for(ctx, "Rudy Runner", ("KC", opp), "player_rush_yds", "over")) & {"def_tough", "def_soft"}


def test_def_tough_fewest_copy_uses_number_of_ranked_teams():
    # Only 30 ranked: bottom 8 starts at rank 23, and rank 30 is the fewest.
    ctx, opp = ctx_vs(30, team_games=defense(teams=TEAMS[:30]))
    tough = [t for t in tags_for(ctx, "Rudy Runner", ("KC", opp), "player_rush_yds", "over") if t.key == "def_tough"]
    assert tough and tough[0].detail.endswith("(1st fewest)")
    ctx, opp = ctx_vs(23, team_games=defense(teams=TEAMS[:30]))
    assert "def_tough" in keys(tags_for(ctx, "Rudy Runner", ("KC", opp), "player_rush_yds", "over"))
    ctx, opp = ctx_vs(22, team_games=defense(teams=TEAMS[:30]))
    assert "def_tough" not in keys(tags_for(ctx, "Rudy Runner", ("KC", opp), "player_rush_yds", "over"))
    # With 10 ranked, rank 5 is in both halves: it is soft only.
    ctx, opp = ctx_vs(5, team_games=defense(teams=TEAMS[:10]))
    assert keys(tags_for(ctx, "Rudy Runner", ("KC", opp), "player_rush_yds", "over")) == {"def_soft", "def_rz"}


def test_def_tags_need_four_window_games():
    # The opponent allows the most of everything but has 3 games (and no last season): unranked.
    few = [TeamGame(S, w, "BIG", "RB", 500.0, 99, 99) for w in range(1, 4)]
    ctx = build_context(defense() + few, runner(), {}, S, W)
    assert ("BIG", "RB") not in ctx.def_rank
    assert keys(tags_for(ctx, "Rudy Runner", ("KC", "BIG"), "player_rush_yds", "over")) == set()
    ctx = build_context(defense() + few + [TeamGame(S, 4, "BIG", "RB", 500.0, 99, 99)], runner(), {}, S, W)
    assert ctx.def_rank[("BIG", "RB")]["yds"] == (500.0, 1)
    assert "def_soft" in keys(tags_for(ctx, "Rudy Runner", ("KC", "BIG"), "player_rush_yds", "over"))


def test_def_rank_uses_last_8_games_and_breaks_ties_by_team():
    old = [TeamGame(S, w, "D01", "RB", 0.0, 0, 0) for w in range(1, 3)]
    tg = [r for r in defense(games=10) if not (r.team == "D01" and r.week <= 2)] + old
    ctx = build_context(tg, [], {}, S, 11)
    assert ctx.def_rank[("D01", "RB")]["yds"] == (120.0, 1)  # weeks 3-10 only
    tie = [TeamGame(S, w, t, "WR", 50.0, 1, 1) for t in ("BBB", "AAA", "CCC") for w in range(1, 5)]
    ctx = build_context(tie, [], {}, S, W)
    assert [ctx.def_rank[(t, "WR")]["yds"][1] for t in ("AAA", "BBB", "CCC")] == [1, 2, 3]


def test_def_rz_and_def_td_only_on_their_markets():
    ctx, opp = ctx_vs(1)
    ev = ("KC", opp)
    assert keys(tags_for(ctx, "Rudy Runner", ev, TD_MARKET, "over")) == {"def_rz", "def_td"}
    assert keys(tags_for(ctx, "Rudy Runner", ev, "player_rush_yds", "over")) == {"def_soft", "def_rz"}
    assert keys(tags_for(ctx, "Rudy Runner", ev, "player_rush_attempts", "over")) == {"def_soft"}
    assert keys(tags_for(ctx, "Rudy Runner", ev, "player_rush_reception_yds", "over")) == {"def_soft"}
    # rz/td rank 9 (tied with rank 10, wins on team name) while yds rank 1: only def_soft on the TD-adjacent markets.
    ctx, opp = ctx_vs(1, team_games=defense(rz=lambda r: 40 - (10 if r == 1 else r), td=lambda r: 40 - (10 if r == 1 else r)))
    assert keys(tags_for(ctx, "Rudy Runner", ("KC", opp), "player_rush_yds", "over")) == {"def_soft"}
    assert keys(tags_for(ctx, "Rudy Runner", ("KC", opp), TD_MARKET, "over")) == set()
    # WR and QB.
    tg = defense(pos="WR") + defense(pos="QB")
    players = runner(name="Will Receiver", pos="WR", carries=0, targets=3) + runner(name="Pat Back", pos="QB", carries=1)
    ctx = build_context(tg, players, {}, S, W)
    ev = ("KC", TEAMS[0])
    assert keys(tags_for(ctx, "Will Receiver", ev, "player_reception_yds", "over")) == {"def_soft", "def_rz"}
    assert keys(tags_for(ctx, "Will Receiver", ev, "player_receptions", "over")) == {"def_soft"}
    assert keys(tags_for(ctx, "Pat Back", ev, "player_pass_yds", "over")) == {"def_soft"}
    assert keys(tags_for(ctx, "Pat Back", ev, TD_MARKET, "over")) == set()


def test_def_rz_and_def_td_copy():
    ctx, opp = ctx_vs(3)
    by = {t.key: t.detail for t in tags_for(ctx, "Rudy Runner", ("KC", opp), TD_MARKET, "over")}
    assert by == {"def_rz": "JAX allows 37.0 RB red-zone looks per game (3rd most)",
                  "def_td": "JAX allows 37.0 RB TDs per game (3rd most)"}


def test_detail_copy_def_soft():
    ctx, opp = ctx_vs(3)
    assert opp == "JAX"
    t = [t for t in tags_for(ctx, "Rudy Runner", ("JAX", "KC"), "player_rush_yds", "over") if t.key == "def_soft"]
    assert t == [Tag("def_soft", "tailwind", "JAX allows 118 RB yds per game (3rd most)")]


def test_ordinals_in_copy():
    for rank, word in ((1, "1st"), (2, "2nd"), (11, "11th"), (12, "12th"), (13, "13th"), (21, "21st"), (22, "22nd"), (23, "23rd")):
        tg = defense(yds=lambda r: 121 - r)
        ctx = build_context(tg, runner(), {}, S, W)
        ctx.def_rank[(TEAMS[0], "RB")]["yds"] = (120.0, rank)
        if rank <= 8:
            d = [t.detail for t in tags_for(ctx, "Rudy Runner", ("KC", TEAMS[0]), "player_rush_yds", "over")
                 if t.key == "def_soft"][0]
            assert d.endswith(f"({word} most)")
    from src.edge.matchups import ordinal
    assert [ordinal(n) for n in (1, 2, 3, 4, 11, 12, 13, 21, 22, 23, 101, 111)] == \
        ["1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "22nd", "23rd", "101st", "111th"]


# Usage

def usage_rows(earlier, recent, team_carries=20):
    weeks = iter(range(1, 9))
    return ([pg(next(weeks), carries=c, team_carries=team_carries) for c in earlier]
            + [pg(next(weeks), carries=c, team_carries=team_carries) for c in recent])


def usage(rows, market="player_rush_yds", side="over", name="Rudy Runner"):
    ctx = build_context([], rows, {}, S, W)
    return [t for t in tags_for(ctx, name, ("KC", "BUF"), market, side) if t.key == "usage_up"]


def test_usage_up_at_exactly_5_points_fires_and_4_9_does_not():
    tags = usage(usage_rows([4, 4], [5, 5, 5]))  # 25% vs 20%
    assert tags == [Tag("usage_up", "tailwind", "Carry share 25% last 3 vs 20% before")]
    assert usage(usage_rows([200, 200], [249, 249, 249], team_carries=1000)) == []  # 24.9% vs 20%
    assert usage(usage_rows([4, 4], [5, 5, 5]), market=TD_MARKET)
    assert usage(usage_rows([4, 4], [5, 5, 5]), side="under")[0].direction == "headwind"


def test_usage_up_needs_two_earlier_games():
    last_season = [pg(w, season=S - 1, carries=1) for w in range(10, 18)]
    assert usage(last_season + usage_rows([2], [6, 6, 6])) == []  # one earlier 2024 game
    assert usage(last_season + usage_rows([2, 2], [6, 6, 6]))
    assert usage(usage_rows([2, 2], [6, 6])) == []  # four games in all: 3 recent leaves 1 earlier


def test_usage_up_wr_target_share_copy():
    rows = [pg(w, name="Will Receiver", pos="WR", carries=0, targets=t, team_targets=30)
            for w, t in zip(range(1, 6), (3, 3, 9, 9, 9))]
    tags = usage(rows, market="player_receptions", name="Will Receiver")
    assert [t.detail for t in tags] == ["Target share 30% last 3 vs 10% before"]
    assert usage(rows, market=TD_MARKET, name="Will Receiver")
    qb = [pg(w, name="Pat Back", pos="QB", carries=c) for w, c in zip(range(1, 6), (0, 0, 9, 9, 9))]
    assert usage(qb, market="player_pass_yds", name="Pat Back") == []


def rz(rows_kw, market=TD_MARKET):
    ctx = build_context([], [pg(w, **rows_kw) for w in range(1, 9)], {}, S, W)
    return [t for t in tags_for(ctx, "Rudy Runner", ("KC", "BUF"), market, "over") if t.key == "rz_role"]


def test_rz_role_25_percent_fires_24_does_not():
    assert rz(dict(rz_carries=3, rz_targets=2, team_rz_opps=20)) == [
        Tag("rz_role", "tailwind", "25% of team red-zone looks, last 4")]
    assert rz(dict(rz_carries=24, team_rz_opps=100)) == []
    assert rz(dict(rz_carries=3, rz_targets=2, team_rz_opps=20), market="player_rush_yds") == []
    assert rz(dict(team_rz_opps=0)) == []


def test_rz_role_uses_last_4_games_only():
    rows = [pg(w, rz_carries=0) for w in range(1, 5)] + [pg(w, rz_carries=10) for w in range(5, 9)]
    ctx = build_context([], rows, {}, S, W)
    assert "rz_role" in keys(tags_for(ctx, "Rudy Runner", ("KC", "BUF"), TD_MARKET, "over"))
    ctx = build_context([], list(reversed(rows[4:] + [pg(w, rz_carries=0) for w in range(9, 13)])), {}, S, 13)
    assert "rz_role" not in keys(tags_for(ctx, "Rudy Runner", ("KC", "BUF"), TD_MARKET, "over"))


# Injuries

def backfield(backup_carries=6, status="out"):
    rows = runner() + runner(name="Ray Backup", carries=backup_carries)
    return build_context([], rows, {"KC": {"ray backup": status}}, S, W)


def mate(ctx, name="Rudy Runner", market="player_rush_yds", side="over"):
    return [t for t in tags_for(ctx, name, ("KC", "BUF"), market, side) if t.key == "teammate_out"]


def test_teammate_out_requires_out_or_doubtful_and_30_percent():
    assert mate(backfield()) == [Tag("teammate_out", "tailwind", "Ray Backup (Out) — 30% of RB carries, last 4")]
    assert [t.detail for t in mate(backfield(status="doubtful"))] == ["Ray Backup (Doubtful) — 30% of RB carries, last 4"]
    assert mate(backfield(status="questionable")) == []
    rows = runner() + runner(name="Ray Backup", carries=29, pos_carries=100)
    assert mate(build_context([], rows, {"KC": {"ray backup": "out"}}, S, W)) == []  # 29%
    assert mate(backfield(), market=TD_MARKET)
    assert mate(backfield(), side="under")[0].direction == "headwind"


def test_teammate_out_needs_same_position_and_team():
    rows = runner() + runner(name="Will Receiver", pos="WR", carries=0, targets=8)
    assert mate(build_context([], rows, {"KC": {"will receiver": "out"}}, S, W)) == []
    rows = runner() + runner(name="Ray Backup", team="BUF", carries=10)
    assert mate(build_context([], rows, {"BUF": {"ray backup": "out"}}, S, W)) == []
    wr = runner(name="Will Receiver", pos="WR", carries=0, targets=3) + runner(name="Wes Deep", pos="WR", carries=0, targets=4)
    ctx = build_context([], wr, {"KC": {"wes deep": "out"}}, S, W)
    assert [t.detail for t in mate(ctx, name="Will Receiver", market="player_receptions")] == \
        ["Wes Deep (Out) — 40% of WR targets, last 4"]


def test_teammate_out_share_is_over_the_teams_last_4_games_not_his_own():
    inj = {"KC": {"ray backup": "out"}}
    # On IR since Week 3: he led the backfield in Weeks 1-2, then six team games without him.
    ir = runner() + runner(weeks=range(1, 3), name="Ray Backup", carries=15)
    assert mate(build_context([], ir, inj, S, W)) == []
    # Hurt last season: his only rows are 2023's, the team has played 8 games since.
    last_season = runner() + runner(weeks=range(10, 18), season=S - 1, name="Ray Backup", carries=15)
    assert mate(build_context([], last_season, inj, S, W)) == []
    # Led each of the team's last 4 games: fires.
    led = runner() + runner(weeks=range(5, 9), name="Ray Backup", carries=12)
    assert [t.detail for t in mate(build_context([], led, inj, S, W))] == \
        ["Ray Backup (Out) — 60% of RB carries, last 4"]
    # Missed one of the last 4: that game counts 0 for him (3 x 12 / 4 x 20 = 45%).
    missed = runner() + runner(weeks=(5, 6, 8), name="Ray Backup", carries=12)
    assert [t.detail for t in mate(build_context([], missed, inj, S, W))] == \
        ["Ray Backup (Out) — 45% of RB carries, last 4"]


def test_teammate_out_window_skips_byes():
    # Team bye in Week 7: the last 4 team games before Week 9 are 4, 5, 6, 8.
    weeks = (1, 2, 3, 4, 5, 6, 8)
    rows = runner(weeks=weeks) + runner(weeks=(4, 5, 6, 8), name="Ray Backup", carries=6)
    assert [t.detail for t in mate(build_context([], rows, {"KC": {"ray backup": "out"}}, S, W))] == \
        ["Ray Backup (Out) — 30% of RB carries, last 4"]


def test_teammate_out_not_on_the_injured_player_himself():
    ctx = backfield()
    assert mate(ctx, name="Ray Backup") == []
    assert mate(ctx) != []


def test_questionable_is_headwind_on_over_tailwind_on_under():
    ctx = build_context([], runner(), {"KC": {"rudy runner": "questionable"}}, S, W)
    over = tags_for(ctx, "Rudy Runner", ("KC", "BUF"), "player_rush_yds", "over")
    under = tags_for(ctx, "Rudy Runner", ("KC", "BUF"), TD_MARKET, "under")
    assert Tag("questionable", "headwind", "Listed Questionable") in over
    assert Tag("questionable", "tailwind", "Listed Questionable") in under
    ctx = build_context([], runner(), {"KC": {"rudy runner": "out"}}, S, W)
    assert "questionable" not in keys(tags_for(ctx, "Rudy Runner", ("KC", "BUF"), "player_rush_yds", "over"))


# Direction

def test_def_tough_is_headwind_for_over_and_tailwind_for_under():
    ctx, opp = ctx_vs(32)
    over = [t for t in tags_for(ctx, "Rudy Runner", ("KC", opp), "player_rush_yds", "over") if t.key == "def_tough"]
    under = [t for t in tags_for(ctx, "Rudy Runner", ("KC", opp), "player_rush_yds", "under") if t.key == "def_tough"]
    assert [t.direction for t in over] == ["headwind"] and [t.direction for t in under] == ["tailwind"]
    ctx, opp = ctx_vs(1)
    under = [t for t in tags_for(ctx, "Rudy Runner", ("KC", opp), "player_rush_yds", "under") if t.key == "def_soft"]
    assert [t.direction for t in under] == ["headwind"]


# Identity

def test_traded_player_whose_latest_team_is_not_in_the_event_gets_no_tags():
    rows = runner(weeks=range(1, 5)) + runner(weeks=range(5, 9), team="BUF") + runner(name="Ray Backup", carries=10)
    inj = {"KC": {"rudy runner": "questionable", "ray backup": "out"}}
    ctx = build_context(defense(), rows, inj, S, W)
    assert ctx.players["rudy runner"] == [("BUF", "RB")]
    for market in ("player_rush_yds", TD_MARKET):
        assert tags_for(ctx, "Rudy Runner", ("KC", TEAMS[0]), market, "over") == []
    # His new team is in the event: tags are his new team's, opponent the other event team.
    assert "def_soft" in keys(tags_for(ctx, "Rudy Runner", (TEAMS[0], "BUF"), "player_rush_yds", "over"))
    assert "teammate_out" not in keys(tags_for(ctx, "Rudy Runner", (TEAMS[0], "BUF"), "player_rush_yds", "over"))


def test_duplicate_name_key_on_event_teams_gets_no_tags():
    rows = runner() + runner(team=TEAMS[0])
    ctx = build_context(defense(), rows, {"KC": {"rudy runner": "questionable"}}, S, W)
    assert len(ctx.players["rudy runner"]) == 2
    assert tags_for(ctx, "Rudy Runner", ("KC", TEAMS[0]), "player_rush_yds", "over") == []
    # Same team twice in one game: also ambiguous.
    ctx = build_context(defense(), runner() + runner(pos="WR"), {"KC": {"rudy runner": "questionable"}}, S, W)
    assert len(ctx.players["rudy runner"]) == 2
    assert tags_for(ctx, "Rudy Runner", ("KC", TEAMS[0]), "player_rush_yds", "over") == []
    # A namesake elsewhere in the league doesn't block the event's player.
    ctx = build_context(defense(), runner() + runner(team="SEA"), {}, S, W)
    assert "def_soft" in keys(tags_for(ctx, "Rudy Runner", ("KC", TEAMS[0]), "player_rush_yds", "over"))


def test_kicker_and_unknown_player_get_no_tags():
    ctx = build_context(defense(), runner(), {"KC": {"kyle kicker": "questionable", "nobody": "questionable"}}, S, W)
    assert tags_for(ctx, "Kyle Kicker", ("KC", TEAMS[0]), "player_kicking_points", "over") == []
    assert tags_for(ctx, "Nobody Atall", ("KC", TEAMS[0]), "player_rush_yds", "over") == []


def test_display_name_is_normalized():
    ctx, opp = ctx_vs(1)
    assert "def_soft" in keys(tags_for(ctx, "RUDY RUNNER Jr.", ("KC", opp), "player_rush_yds", "over"))


# Point-in-time

def test_build_context_excludes_the_games_own_week_and_later():
    later = [TeamGame(S, w, TEAMS[31], "RB", 999.0, 99, 99) for w in (9, 10)]
    rows = runner() + runner(weeks=(9, 10), team="BUF", carries=19)
    ctx = build_context(defense() + later, rows, {}, S, W)
    assert ctx.def_rank[(TEAMS[31], "RB")]["yds"] == (89.0, 32)
    assert ctx.players["rudy runner"] == [("KC", "RB")]
    assert all((r.season, r.week) < (S, W) for rs in ctx.player_rows.values() for r in rs)
    assert ("BUF", "rudy runner") not in ctx.player_rows
    assert [r.week for r in ctx.player_rows[("KC", "rudy runner")]] == list(range(1, 9))
    # Week 1 uses last season's final games.
    ctx = build_context(defense(season=S - 1), runner(season=S - 1), {}, S, 1)
    assert ctx.def_rank[(TEAMS[0], "RB")]["yds"] == (120.0, 1)
    assert "def_soft" in keys(tags_for(ctx, "Rudy Runner", ("KC", TEAMS[0]), "player_rush_yds", "over"))


def test_def_rz_and_def_td_copy_shows_one_decimal():
    others = defense(teams=[t for t in TEAMS if t != "JAX"], rz=lambda r: 5 if r == 1 else 0,
                     td=lambda r: 2 if r == 1 else 0)
    jax = [TeamGame(S, w, "JAX", "RB", 100.0, rz_, td_)
           for w, rz_, td_ in zip(range(1, 6), (5, 4, 4, 5, 4), (2, 1, 1, 2, 1))]
    ctx = build_context(others + jax, runner(), {}, S, W)
    by = {t.key: t.detail for t in tags_for(ctx, "Rudy Runner", ("KC", "JAX"), TD_MARKET, "over")}
    assert by == {"def_rz": "JAX allows 4.4 RB red-zone looks per game (2nd most)",
                  "def_td": "JAX allows 1.4 RB TDs per game (2nd most)"}
