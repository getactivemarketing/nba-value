import dataclasses
import io
import time
from datetime import date, datetime, timezone

import pytest
from PIL import Image

from src.edge.social import cards
from src.edge.social.blind import BlindCard
from src.edge.social.nba_matchup import Panel
from src.edge.social.nfl_matchup import NflPanel

SLATE = date(2026, 11, 15)
LOCKED = datetime(2026, 11, 15, 23, 10, tzinfo=timezone.utc)
MAX_BYTES = int(1.5 * 1024 * 1024)
LONG = "Giannis Antetokounmpo-Wolfeschlegelsteinhausen Jr."


def png_bytes(size=(350, 254), color=(200, 120, 60, 255)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGBA", size, color).save(buf, "PNG")
    return buf.getvalue()


def check(data: bytes) -> None:
    assert isinstance(data, bytes) and len(data) < MAX_BYTES
    img = Image.open(io.BytesIO(data))
    assert img.format == "PNG"
    assert img.size == (1080, 1350)
    img.load()


def nba_card(panel=None) -> BlindCard:
    return BlindCard(sport="nba", n=1, position="F", market_label="Pts+Reb+Ast", side="over", line=32.5,
                     price=116, ev=0.05, n_books=9, l10_hits=5, l10_games=10, l10_avg=34.1,
                     tip_label="1:10 PM ET", panel=panel)


NBA_PANEL = Panel(favorable=True, opp_vs_pos_rank=27, drtg_rank=22, pace_rank=6, usage_l10=26.8,
                  usage_change=2.7, minutes_l5=35.2)


def prop(i=0, player="Jalen Brunson", **kw):
    r = {"player": player, "market": "player_points", "market_label": "Points", "side": "over", "line": 24.5,
         "price": -110, "fair_prob": 0.58, "ev": 0.04, "book": "fanduel", "n_books": 9,
         "matchup": "New York Knicks @ Boston Celtics", "team": "NYK", "rank": i + 1,
         "l10": {"hits": 9, "games": 10, "rate": 0.9}}
    r.update(kw)
    return r


EVENT = {"sport": "nba", "away_team": "Indiana Pacers", "home_team": "Philadelphia 76ers",
         "away_score": 112, "home_score": 118}
RECORDS = {"IND": {"w": 7, "l": 6, "t": 0, "games": 13, "label": "7-6"},
           "PHI": {"w": 8, "l": 5, "t": 0, "games": 13, "label": "8-5"}}
PLAY = {"kind": "blind", "sport": "nba", "n": 1, "player": "Pascal Siakam", "team": "IND",
        "matchup": "Indiana Pacers @ Philadelphia 76ers", "market": "player_points_rebounds_assists",
        "market_label": "Pts+Reb+Ast", "side": "over", "line": 32.5, "price": 116, "result": "win",
        "profit_units": 1.16, "actual": 36, "stat_line": "24 pts · 7 reb · 5 ast", "slate_date": SLATE,
        "panel": {"opp_vs_pos_rank": 27}, "position": "F"}


def timed(fn, *a, **kw) -> bytes:
    t0 = time.perf_counter()
    out = fn(*a, **kw)
    assert time.perf_counter() - t0 < 5
    return out


# ---- every renderer: decodable 1080x1350 PNG under 1.5 MB, in < 5 s, with no faces or logos ---------

def test_blind_nba_with_panel():
    check(timed(cards.render_blind, nba_card(NBA_PANEL)))


def test_blind_panel_as_stored_dict():
    import dataclasses
    check(timed(cards.render_blind, nba_card(dataclasses.asdict(NBA_PANEL))))


def test_blind_without_panel_or_form():
    card = BlindCard(sport="nba", n=3, position=None, market_label="Threes", side="under", line=2.5, price=-105,
                     ev=0.031, n_books=7, l10_hits=None, l10_games=None, l10_avg=None, tip_label="Tonight",
                     panel=None)
    check(timed(cards.render_blind, card))


def test_blind_nfl_panel_without_snap_share():
    panel = NflPanel(favorable=True, opp_rank=3, share_l4=0.27, snap_share=None, tags=["def_soft"])
    card = BlindCard(sport="nfl", n=2, position="WR", market_label="Rec Yds", side="over", line=64.5,
                     price=-110, ev=0.042, n_books=8, l10_hits=6, l10_games=10, l10_avg=71.2,
                     tip_label="1:00 PM ET", panel=panel)
    check(timed(cards.render_blind, card))


def test_cheatsheet_no_faces():
    rows = [prop(i) for i in range(10)]
    check(timed(cards.render_cheatsheet, "nba", SLATE, rows, {}))


def test_value_no_faces():
    legs = [prop(i, player=f"Player Number {i}") for i in range(5)]
    check(timed(cards.render_value, "nba", SLATE, legs, {}, LOCKED))


def test_final_no_logos_no_faces():
    plays = [{**PLAY, "kind": "blind"}, {**PLAY, "kind": "value", "player": "Tyrese Maxey", "result": "loss",
                                         "profit_units": -1.0}]
    check(timed(cards.render_final, EVENT, RECORDS, {}, plays, {}))


def test_receipt_no_face_no_logo():
    check(timed(cards.render_receipt, PLAY, None, None))


# ---- with images ------------------------------------------------------------------------------

def test_renderers_with_faces_logos_and_original():
    face, logo = png_bytes(), png_bytes((500, 500), (0, 45, 98, 255))
    faces = {"Jalen Brunson": face, "Pascal Siakam": face}
    logos = {"IND": logo, "PHI": logo, "NYK": logo}
    check(cards.render_cheatsheet("nba", SLATE, [prop(i) for i in range(6)], faces, logos=logos))
    check(cards.render_value("nba", SLATE, [prop(i) for i in range(5)], faces, LOCKED, logos=logos))
    check(cards.render_final(EVENT, RECORDS, logos, [PLAY], faces))
    original = cards.render_blind(nba_card(NBA_PANEL))
    check(cards.render_receipt(PLAY, face, logo, original))


def test_corrupt_image_bytes_fall_back():
    junk = b"not a png"
    check(cards.render_receipt(PLAY, junk, junk, junk))
    check(cards.render_final(EVENT, RECORDS, {"IND": junk, "PHI": junk}, [PLAY], {"Pascal Siakam": junk}))


def test_long_names_and_many_plays_render():
    rows = [prop(i, player=LONG, matchup="Golden State Warriors @ Portland Trail Blazers") for i in range(10)]
    check(cards.render_cheatsheet("nba", SLATE, rows, {}))
    check(cards.render_value("nba", SLATE, rows[:5], {}, LOCKED))
    plays = [{**PLAY, "player": LONG, "result": r, "profit_units": u}
             for r, u in [("win", 1.0), ("loss", -1.0), ("push", 0.0), ("void", None), (None, None)] * 2]
    check(cards.render_final(EVENT, {}, {}, plays, {}))
    check(cards.render_receipt({**PLAY, "player": LONG, "stat_line": "x " * 80}, None, None))


def test_nfl_final_and_receipt():
    event = {"sport": "nfl", "away_team": "Buffalo Bills", "home_team": "Kansas City Chiefs",
             "away_score": 24, "home_score": 24}
    play = {**PLAY, "sport": "nfl", "kind": "value", "player": "Josh Allen", "team": "BUF",
            "market": "player_pass_yds", "market_label": "Pass Yds", "line": 245.5, "actual": 281,
            "stat_line": None, "panel": None, "matchup": "Buffalo Bills @ Kansas City Chiefs"}
    check(cards.render_final(event, {}, {}, [play], {}))
    check(cards.render_receipt(play, None, None))


# ---- helpers ----------------------------------------------------------------------------------

def test_ordinal():
    assert [cards.ordinal(n) for n in (1, 2, 3, 4, 11, 12, 13, 21, 22, 27)] == \
        ["1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "22nd", "27th"]


def test_record_label_needs_three_games():
    recs = {"IND": {"games": 13, "label": "7-6"}, "PHI": {"games": 2, "label": "1-1"}}
    assert cards.record_label(recs, "IND") == "7-6"
    assert cards.record_label(recs, "PHI") is None
    assert cards.record_label(recs, "BOS") is None
    assert cards.record_label({}, None) is None


def test_ellipsize_fits_width():
    font = cards.font("sans_bold", 40)
    out = cards.ellipsize(LONG, font, 300)
    assert out.endswith("…") and font.getlength(out) <= 300
    assert cards.ellipsize("Short", font, 300) == "Short"


@pytest.mark.parametrize("name,short", [("Jalen Brunson", "J. Brunson"), ("Nikola Jokić", "N. Jokić"),
                                        ("Nene", "Nene")])
def test_short_name(name, short):
    assert cards.short_name(name) == short


def test_last_name_skips_suffix():
    assert cards.last_name("Jaren Jackson Jr.") == "Jackson"
    assert cards.last_name("Pascal Siakam") == "Siakam"


def test_sample_cli_writes_cards(tmp_path):
    cards.main(["--sample", str(tmp_path)])
    names = sorted(p.name for p in tmp_path.glob("*.png"))
    assert {"blind_nba.png", "blind_nfl.png", "cheatsheet.png", "value.png", "final.png",
            "receipt.png"} <= set(names)
    for p in tmp_path.glob("*.png"):
        check(p.read_bytes())


# ---- fix round 1: labels, None-safety, identity ---------------------------------------------

@pytest.fixture
def drawn(monkeypatch):
    """Every string drawn with ImageDraw.text, in order."""
    from PIL import ImageDraw
    out: list[str] = []
    orig = ImageDraw.ImageDraw.text

    def spy(self, xy, text, *a, **kw):
        out.append(str(text))
        return orig(self, xy, text, *a, **kw)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", spy)
    return out


def test_form_labels():
    assert cards.form_labels(9, 10) == ("LAST 9", "AVG L10")
    assert cards.form_labels(9, 7) == ("LAST 9", "AVG L7")
    assert cards.form_labels(None, None) == ("LAST 10", "L10 AVG")
    assert cards.form_labels(9) == ("LAST 9", "L10 AVG")


def test_blind_form_tile_labels_follow_games(drawn):
    card = BlindCard(sport="nfl", n=2, position="WR", market_label="Rec Yds", side="over", line=64.5, price=-110,
                     ev=0.042, n_books=8, l10_hits=6, l10_games=9, l10_avg=71.2, tip_label="Tonight", panel=None,
                     l10_avg_games=8)
    cards.render_blind(card)
    text = "".join(drawn)
    assert "6/9" in text and "LAST 9" in text and "71.2" in text and "AVG L8" in text
    assert "LAST 10" not in text and "L10 AVG" not in text and "AVG L9" not in text


def test_blind_avg_tile_counts_its_own_games_after_a_push(drawn):
    # 10 played games, one push: hit rate over 9 decided games, the average still over all 10
    card = dataclasses.replace(nba_card(NBA_PANEL), l10_hits=6, l10_games=9, l10_avg=30.2, l10_avg_games=10)
    cards.render_blind(card)
    text = "".join(drawn)
    assert "LAST 9" in text and "AVG L10" in text and "AVG L9" not in text


def test_blind_avg_tile_without_avg(drawn):
    cards.render_blind(dataclasses.replace(nba_card(NBA_PANEL), l10_avg=None, l10_avg_games=None))
    text = "".join(drawn)
    assert "L10 AVG" in text and "—" in text


def test_cheatsheet_header_says_hit_rate_and_rows_show_hits_over_games(drawn):
    cards.render_cheatsheet("nba", SLATE, [prop(i, l10={"hits": 7, "games": 9, "rate": 0.778}) for i in range(6)], {})
    text = "".join(drawn)
    assert "HIT RATE" in text and "7/9" in text


NULLABLE_PROP = ["price", "line", "ev", "fair_prob", "team", "matchup", "player", "side", "market_label", "l10",
                 "book", "rank"]
NULLABLE_PLAY = ["price", "line", "ev", "actual", "stat_line", "team", "matchup", "player", "profit_units",
                 "result", "side", "market_label", "panel", "slate_date", "n", "kind", "sport"]
NULLABLE_EVENT = ["away_team", "home_team", "away_score", "home_score", "sport"]
NULLABLE_CARD = ["position", "market_label", "side", "line", "price", "ev", "n_books", "l10_hits", "l10_games",
                 "l10_avg", "tip_label", "panel", "sport", "n"]


def _variants(base: dict, fields: list[str]):
    for f in fields:
        yield pytest.param({**base, f: None}, id=f"{f}=None")
        yield pytest.param({k: v for k, v in base.items() if k != f}, id=f"no-{f}")


@pytest.mark.parametrize("field", NULLABLE_CARD)
def test_blind_none_fields(field):
    import dataclasses
    check(cards.render_blind(dataclasses.replace(nba_card(NBA_PANEL), **{field: None})))


@pytest.mark.parametrize("row", list(_variants(prop(), NULLABLE_PROP)))
def test_cheatsheet_none_fields(row):
    check(cards.render_cheatsheet("nba", SLATE, [row] * 6, None))


@pytest.mark.parametrize("row", list(_variants(prop(), NULLABLE_PROP)))
def test_value_none_fields(row):
    check(cards.render_value("nba", SLATE, [row] * 5, None, LOCKED))


def test_value_without_lock_time():
    check(cards.render_value("nba", SLATE, [prop()] * 5, None, None))


@pytest.mark.parametrize("play", list(_variants(PLAY, NULLABLE_PLAY)) + [{**PLAY, "actual": "n/a"},
                                                                        {**PLAY, "profit_units": "x"}])
def test_receipt_none_fields(play):
    check(cards.render_receipt(play, None, None))


@pytest.mark.parametrize("event", list(_variants(EVENT, NULLABLE_EVENT)))
def test_final_none_event_fields(event):
    check(cards.render_final(event, None, None, [PLAY], None))


@pytest.mark.parametrize("play", list(_variants(PLAY, NULLABLE_PLAY)))
def test_final_none_play_fields(play):
    check(cards.render_final(EVENT, RECORDS, {}, [play], None))


def test_final_no_plays_none_records():
    check(cards.render_final(EVENT, None, None, None, None))


def _blind_card_from_play(play: dict, panel) -> BlindCard:
    from src.edge.social import blind
    return blind.to_card({**play, "panel": panel}, {"e1": play["commence_time"], "e2": play["commence_time"]})


def test_blind_card_draws_no_identity(drawn):
    import dataclasses
    tip = datetime(2026, 11, 16, 0, 30, tzinfo=timezone.utc)
    nba = {"sport": "nba", "n": 7, "event_id": "e1", "player": "Jayson Tatum", "team": "BOS", "position": "F",
           "home_team": "Boston Celtics", "away_team": "Miami Heat", "matchup": "Miami Heat @ Boston Celtics",
           "market": "player_points", "market_label": "Points", "side": "over", "line": 27.5, "price": 105,
           "ev": 0.04, "n_books": 9, "book": "fanduel", "form": {"hits": 7, "games": 10, "avg": 29.1},
           "commence_time": tip}
    nfl = {**nba, "sport": "nfl", "player": "Josh Allen", "team": "BUF", "position": "QB",
           "home_team": "Kansas City Chiefs", "away_team": "Buffalo Bills",
           "matchup": "Buffalo Bills @ Kansas City Chiefs", "market": "player_pass_yds", "market_label": "Pass Yds",
           "line": 245.5}
    cards.render_blind(_blind_card_from_play(nba, dataclasses.asdict(NBA_PANEL)))
    cards.render_blind(_blind_card_from_play(nfl, NflPanel(True, 3, 0.27, None, ["def_soft"])))
    assert drawn
    joined = "".join(drawn)
    forbidden = ["Jayson", "Tatum", "BOS", "MIA", "Boston", "Celtics", "Miami", "Heat",
                 "Josh", "Allen", "BUF", "KC", "Kansas", "Chiefs", "Buffalo", "Bills"]
    for word in forbidden:
        assert word not in joined and word.upper() not in joined, word
