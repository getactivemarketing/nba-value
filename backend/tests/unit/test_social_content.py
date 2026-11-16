from datetime import date, datetime, timedelta, timezone

from src.config import settings
from src.edge.social import content

TIP = datetime(2026, 11, 17, 3, 30, tzinfo=timezone.utc)  # 10:30 PM ET on 11/16


def test_final_recap_for_game_finishing_after_midnight():
    # tips 10:30 PM ET 11/16, ends after midnight ET: still the 11/16 slate
    assert content.slate_for(TIP) == date(2026, 11, 16)


def prop(i, ev=0.05, hits=8, market="player_points", pid=None):
    return {"market": market, "ev": ev, "line": 20.5, "side": "over", "bdl_player_id": pid or i + 1, "player": f"P{i}"}


def games(hit):
    """10 games: `hit` of them over 20.5."""
    return [{"minutes": 30, "game_date": date(2026, 11, 1 + i), "pts": 25 if i < hit else 10, "season": 2026}
            for i in range(10)]


def sheet(n, **kw):
    rows = [prop(i, **kw) for i in range(n)]
    return rows, {i + 1: games(8 - i % 3) for i in range(n)}


def test_cheatsheet_skips_under_six_rows():
    rows, g = sheet(5)
    assert content.cheatsheet_rows(rows, g) == []
    rows, g = sheet(6)
    assert len(content.cheatsheet_rows(rows, g)) == 6


def test_cheatsheet_excludes_negative_ev():
    rows, g = sheet(6)
    rows[0]["ev"] = -0.02
    assert content.cheatsheet_rows(rows, g) == []  # five left
    rows, g = sheet(7)
    rows[0]["ev"] = 0.0
    out = content.cheatsheet_rows(rows, g)
    assert len(out) == 6 and all(r["ev"] > 0 for r in out)


def test_cheatsheet_sorted_and_capped_and_props_only():
    rows, g = sheet(12)
    rows.append({**prop(99, market="h2h"), "bdl_player_id": 99})
    g[99] = games(10)
    out = content.cheatsheet_rows(rows, g)
    assert len(out) == 10 and all(r["market"].startswith("player_") for r in out)
    rates = [r["l10"]["rate"] for r in out]
    assert rates == sorted(rates, reverse=True)


def event(**kw):
    return {"away_team": "Detroit Pistons", "home_team": "Boston Celtics", "away_score": 104,
            "home_score": 110, **kw}


def test_final_text_shows_push_and_units():
    t = content.final_text("nba", event(), 2, 1, 1, 1.35)
    assert t == ("FINAL\n\nDetroit Pistons 104, Boston Celtics 110\n\n"
                 "TruLine plays in this game: 2-1-1 ✅ +1.35u\n\n#NBA #Pistons #Celtics")
    t = content.final_text("nba", event(), 1, 2, 0, -1.10)
    assert "1-2 -1.10u" in t and "✅" not in t and "1-2-" not in t


def test_nicknames():
    assert content.nickname("Portland Trail Blazers") == "TrailBlazers"
    assert content.nickname("Philadelphia 76ers") == "76ers"


def test_links_carry_utm():
    site = settings.site_url.rstrip("/")
    assert content.blind_reply("NBA", 7) == (
        f"Reveal + full breakdown → {site}/blind/nba/7?utm_source=x&utm_medium=social&utm_campaign=blind")
    v = content.value_text("nba", datetime(2026, 11, 16, 22, 30, tzinfo=timezone.utc))
    assert "Locked 5:30 PM ET." in v
    assert f"{site}/?utm_source=x&utm_medium=social&utm_campaign=value" in v
    assert "utm_campaign=receipt" in content.blind_receipt_text(7, "A One", 116, "nba")
    assert "https://x.com/t/1" in content.blind_receipt_text(7, "A One", 116, "nba", "https://x.com/t/1")


def test_other_templates():
    assert content.blind_text("nba", 3) == "🙈 Blind Play #3 · NBA\n\nNo names. No teams. No bias. Who is it? 👇"
    assert content.cheatsheet_text("nba", date(2026, 11, 16)).startswith("📋 NBA Cheat Sheet · Mon 11/16\n\n")
    assert content.value_receipt_text("A One Over 20.5 Points", -110) == (
        "✅ CASHED. A One Over 20.5 Points at -110 from tonight's Top 5.")
    assert content.blind_receipt_text(7, "A One", 116, "nba").startswith(
        "✅ CASHED. Blind Play #7 was A One.\n\nCalled it before tip at +116. Did you guess right?")


def test_final_text_no_check_for_sub_cent_units():
    assert "✅" not in content.final_text("nba", event(), 1, 1, 0, 0.004)


def test_record_line_counts_only_graded():
    plays = [{"result": "win", "profit_units": 1.16}, {"result": "loss", "profit_units": -1.0},
             {"result": "push", "profit_units": 0.0}, {"result": "void", "profit_units": 0.0},
             {"result": None, "profit_units": None}]
    assert content.record_line(plays) == (1, 1, 1, 0.16)
    assert content.record_line([]) == (0, 0, 0, 0.0)


def _nfl_games(recs: list[int]):
    """Synthetic nfl_players.load_games rows, newest first: receptions per game."""
    return [{"game_date": date(2026, 11, 9) - timedelta(days=7 * i), "season": 2026, "week": 10 - i,
             "opponent": "DAL", "is_home": True, "receptions": r, "rec_yds": 10 * r, "rush_yds": 0}
            for i, r in enumerate(recs)]


def test_nfl_cheatsheet_ranks_by_hit_rate_from_nfl_box_scores():
    rows, games = [], {}
    for i in range(7):
        rows.append({"player": f"WR{i}", "market": "player_receptions", "side": "over", "line": 4.5,
                     "ev": 0.02 + i / 1000, "verify": False, "bdl_player_id": 100 + i})
        games[100 + i] = _nfl_games([6] * (9 - i) + [3] * (1 + i))  # WR0 hits 9 of 10, WR6 3 of 10
    rows.append({**rows[0], "player": "Thin", "bdl_player_id": 200})
    games[200] = _nfl_games([9, 9, 9, 9])  # under 5 games: no form
    out = content.cheatsheet_rows(rows, games, "nfl")
    assert [r["player"] for r in out] == [f"WR{i}" for i in range(7)]
    assert out[0]["l10"] == {"hits": 9, "games": 10, "rate": 0.9}
    assert content.cheatsheet_rows(rows[:5], games, "nfl") == []  # five is too thin


def test_nfl_recent_form_handles_unders_pushes_and_unknown_markets():
    g = _nfl_games([3, 4, 5, 4, 2, 6])
    assert content.nfl_recent_form(g, "player_receptions", 4.0, "under") == {"hits": 2, "games": 4, "rate": 0.5}
    assert content.nfl_recent_form(g, "player_rush_reception_yds", 45.5, "over")["hits"] == 2
    assert content.nfl_recent_form(g, "player_points", 4.5, "over") is None
