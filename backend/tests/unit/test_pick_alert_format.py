"""Pick-alert SMS formatting: bet labels, odds, ET time, optional fields."""

from datetime import datetime, timezone

from src.services.notifications.pick_alerts import format_pick_alert


def test_runline_away_pick_full_format():
    msg = format_pick_alert(
        away_team="DET", home_team="LAA", bet_type="runline", team="DET",
        line=1.5, odds_decimal=2.5, value_score=90, edge=0.23,
        game_time=datetime(2026, 7, 18, 1, 38, tzinfo=timezone.utc),
        away_starter="Tarik Skubal", home_starter="Jose Soriano",
    )
    assert msg == (
        "TruLine pick: DET +1.5 (+150) @ LAA, 9:38 PM ET\n"
        "Score 90 | Edge 23% | Skubal vs Soriano"
    )


def test_moneyline_home_pick_vs_and_negative_odds():
    msg = format_pick_alert(
        away_team="SF", home_team="SEA", bet_type="moneyline", team="SEA",
        line=None, odds_decimal=1.8, value_score=55, edge=0.08,
        game_time=None,
    )
    assert msg == "TruLine pick: SEA ML (-125) vs SF\nScore 55 | Edge 8%"


def test_missing_optional_fields_omitted():
    msg = format_pick_alert(
        away_team="CWS", home_team="TOR", bet_type="moneyline", team="CWS",
        line=None, odds_decimal=None, value_score=52, edge=None, game_time=None,
    )
    assert msg == "TruLine pick: CWS ML @ TOR\nScore 52"


def test_total_pick_future_proofing():
    msg = format_pick_alert(
        away_team="CHC", home_team="NYM", bet_type="total", team=None,
        line=8.5, odds_decimal=1.91, value_score=61, edge=0.06, game_time=None,
    )
    assert msg == "TruLine pick: O/U 8.5 (-110) CHC @ NYM\nScore 61 | Edge 6%"


def test_invalid_odds_sentinel_omitted():
    msg = format_pick_alert(
        away_team="BOS", home_team="NYY", bet_type="moneyline", team="BOS",
        line=None, odds_decimal=1.0, value_score=45, edge=None, game_time=None,
    )
    assert msg == "TruLine pick: BOS ML @ NYY\nScore 45"


def test_runline_missing_line_does_not_crash():
    msg = format_pick_alert(
        away_team="SD", home_team="KC", bet_type="runline", team="SD",
        line=None, odds_decimal=2.3, value_score=82, edge=0.20, game_time=None,
    )
    assert msg == "TruLine pick: SD RL (+130) @ KC\nScore 82 | Edge 20%"
