from datetime import date, datetime, timezone

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
