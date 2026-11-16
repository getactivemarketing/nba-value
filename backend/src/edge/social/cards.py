# backend/src/edge/social/cards.py
"""PNG cards for the X posts (spec §5): blind play, cheat sheet, value plays, final recap, receipt.

Every renderer is pure: it draws from the data it is given and returns PNG bytes, 1080x1350.
Headshots and logos arrive as PNG bytes (or None) from headshots.py; nothing here touches the
network. A missing or unreadable face draws a neutral silhouette; a missing logo draws the team
abbreviation. Long text is cut with an ellipsis so it never overflows.

Row contracts (dict keys read; anything else is ignored):
- cheat sheet row / value leg: player, market, market_label, side, line, price, fair_prob, ev, book,
  matchup ("Away Team @ Home Team"), team (abbr, optional), rank (value), l10 {hits, games} (cheat
  sheet), tags [{key, direction}] (NFL cheat sheet, optional). `faces` is keyed by player name,
  `logos` by team abbreviation.
- final recap: event {sport, away_team, home_team, away_score, home_score}; records from
  content.team_records; plays from content.posted_plays.
- receipt play: kind (blind|value), n, sport, player, team, matchup, market, market_label, side, line,
  price, profit_units, actual (graded stat, optional), stat_line (optional), slate_date (optional),
  panel (optional: its opp_vs_pos_rank / opp_rank fills the third tile), position (optional).

Run `python -m src.edge.social.cards --sample OUT_DIR [--assets DIR]` to write one of each card from
fixture data (DIR may hold face_<First_Last>.png and logo_<ABBR>.png to try real images).
"""
from __future__ import annotations

import argparse
import dataclasses
import io
import logging
from datetime import date, datetime, timezone
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from src.edge.players import ET
from src.edge.social.blind import BOOK_NAMES, BlindCard
from src.edge.social.content import nickname, record_line
from src.edge.tagging import TAG_LABEL
from src.edge.teams import ABBR_BY_NAME, NFL_ABBR_BY_NAME

log = logging.getLogger(__name__)

W, H = 1080, 1350
PAD = 60
FONT_DIR = Path(__file__).resolve().parent / "fonts"
FONT_FILES = {"sans": "DejaVuSans.ttf", "sans_bold": "DejaVuSans-Bold.ttf",
              "mono": "DejaVuSansMono.ttf", "mono_bold": "DejaVuSansMono-Bold.ttf"}

# dark theme (mockup .tl-card)
BG_TOP, BG_BOTTOM = (11, 14, 20), (5, 7, 10)
TEXT = (241, 245, 249)
CYAN = (164, 230, 255)
DIM = (100, 116, 139)
MUTED = (148, 163, 184)
LINE = (30, 41, 59)
TILE = (15, 23, 42)
GREEN = (74, 222, 128)
LIME = (163, 230, 53)
RED = (248, 113, 113)
CHIP_BG = (20, 83, 45)
SIL_BG, SIL_FG = (30, 41, 59), (71, 85, 105)
# light theme (final recap)
L_TOP, L_BOTTOM = (248, 250, 252), (226, 232, 240)
L_TEXT, L_BLUE, L_DIM, L_BORDER = (15, 23, 42), (29, 78, 216), (100, 116, 139), (203, 213, 225)
L_WIN, L_LOSS, L_FADED = (22, 163, 74), (220, 38, 38), (148, 163, 184)

MIN_RECORD_GAMES = 3
STAMP_RESERVE = 280  # receipt: the kicker stops this far short of the right margin, clear of the CASHED band
MAX_FINAL_PLAYS = 6
SHORT_MARKET = {
    "Points": "Pts", "Rebounds": "Reb", "Assists": "Ast", "Threes": "3PM", "Pts+Reb+Ast": "PRA",
    "Pts+Reb": "P+R", "Pts+Ast": "P+A", "Reb+Ast": "R+A", "Anytime TD": "TD",
}
POSITION_NAME = {"G": "GUARD", "F": "FORWARD", "C": "CENTER", "QB": "QUARTERBACK", "RB": "RUNNING BACK",
                 "WR": "WIDE RECEIVER", "TE": "TIGHT END"}
POSITION_GROUP = {"G": "guards", "F": "forwards", "C": "centers"}
NAME_SUFFIXES = {"jr", "jr.", "sr", "sr.", "ii", "iii", "iv", "v"}


# ---- small helpers ----------------------------------------------------------------------------

@lru_cache(maxsize=None)
def font(kind: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_DIR / FONT_FILES[kind]), size)


def ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def ellipsize(text: str, f: ImageFont.FreeTypeFont, max_w: float) -> str:
    if f.getlength(text) <= max_w:
        return text
    while text and f.getlength(text.rstrip() + "…") > max_w:
        text = text[:-1]
    return text.rstrip() + "…"


def _name_parts(name: str) -> list[str]:
    parts = name.split()
    while len(parts) > 2 and parts[-1].lower() in NAME_SUFFIXES:
        parts = parts[:-1]
    return parts


def short_name(name: str) -> str:
    """'Jalen Brunson' -> 'J. Brunson'."""
    parts = name.split()
    return f"{parts[0][0]}. {' '.join(parts[1:])}" if len(parts) > 1 else name


def last_name(name: str) -> str:
    parts = _name_parts(name)
    return parts[-1] if parts else name


def record_label(records: dict, abbr: str | None) -> str | None:
    """A team's W-L label, or None when unknown or under MIN_RECORD_GAMES games (too early to mean much)."""
    r = (records or {}).get(abbr) if abbr else None
    if not r or (r.get("games") or 0) < MIN_RECORD_GAMES:
        return None
    return r.get("label")


def _num(x) -> float | None:
    """A float, or None for None / non-numeric input (cards render '—' rather than raise)."""
    if x is None or isinstance(x, bool):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _price(p) -> str:
    v = _num(p)
    if v is None:
        return "—"
    v = int(v)
    return f"+{v}" if v > 0 else str(v)


def _pct(x, signed: bool = False) -> str:
    v = _num(x)
    if v is None:
        return "—"
    return f"{v * 100:+.1f}%" if signed else f"{v * 100:.1f}%"


def form_labels(games) -> tuple[str, str]:
    """Tile labels for hits/games and the average: 'LAST 9' / 'AVG L9' when 9 games decided the form."""
    g = _num(games)
    if g is None or g <= 0:
        return "LAST 10", "L10 AVG"
    return f"LAST {int(g)}", f"AVG L{int(g)}"


def _nick(name: str | None) -> str:
    return nickname(name) if name and name.strip() else ""


def _md(d: date) -> str:
    return f"{d.month}/{d.day}"


def _abbr(sport: str | None, name: str) -> str:
    name = (name or "").strip()
    maps = [NFL_ABBR_BY_NAME, ABBR_BY_NAME] if sport == "nfl" else [ABBR_BY_NAME, NFL_ABBR_BY_NAME]
    for m in maps:
        if name in m:
            return m[name]
    if name.isupper() and len(name) <= 4:
        return name
    return (name.split()[-1][:3] if name else "?").upper()


def _matchup_abbrs(sport: str | None, matchup: str | None) -> tuple[str, str] | None:
    away, sep, home = (matchup or "").partition(" @ ")
    return (_abbr(sport, away), _abbr(sport, home)) if sep else None


def _market_short(row: dict) -> str:
    label = row.get("market_label") or ""
    return SHORT_MARKET.get(label, label)


def _bet_short(row: dict) -> str:
    """'O 32.5 PRA'; anytime TD reads 'TD Yes' / 'TD No'."""
    if row.get("market_label") == "Anytime TD" or row.get("market") == "player_anytime_td":
        return "TD Yes" if row.get("side") == "over" else "TD No"
    side = {"over": "O", "under": "U"}.get(row.get("side"), str(row.get("side") or "").title())
    line = _num(row.get("line"))
    return " ".join(s for s in (side, f"{line:g}" if line is not None else "", _market_short(row)) if s)


def _image(data: bytes | None) -> Image.Image | None:
    if not data:
        return None
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        return img.convert("RGBA")
    except Exception:  # noqa: BLE001 - a bad image is drawn as the fallback, never an error
        log.warning("card: unreadable image (%d bytes), using fallback", len(data))
        return None


def _png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "PNG", optimize=False, compress_level=6)
    return buf.getvalue()


# ---- drawing primitives -----------------------------------------------------------------------

def _canvas(light: bool = False) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    top, bottom = (L_TOP, L_BOTTOM) if light else (BG_TOP, BG_BOTTOM)
    grad = Image.new("RGB", (1, 256))
    for y in range(256):
        t = y / 255 if light else min(1.0, y / 255 / 0.7)  # dark: 0% -> 70% like the mockup's gradient
        grad.putpixel((0, y), tuple(round(a + (b - a) * t) for a, b in zip(top, bottom)))
    img = grad.resize((W, H), Image.BILINEAR).convert("RGBA")
    return img, ImageDraw.Draw(img)


def _spaced(draw, xy, text: str, f, fill, spacing: float) -> float:
    """Letter-spaced text (the mockup's kickers); returns the drawn width."""
    x, y = xy
    for ch in text:
        draw.text((x, y), ch, font=f, fill=fill)
        x += f.getlength(ch) + spacing
    return x - xy[0] - spacing


def _spaced_width(text: str, f, spacing: float) -> float:
    return sum(f.getlength(c) for c in text) + spacing * max(0, len(text) - 1)


def _kicker(draw, text: str, color=CYAN, y: int = PAD, max_w: int = W - 2 * PAD) -> None:
    f = font("mono_bold", 30)
    if _spaced_width(text, f, 5) > max_w:
        while text and _spaced_width(text.rstrip() + "…", f, 5) > max_w:
            text = text[:-1]
        text = text.rstrip() + "…"
    _spaced(draw, (PAD, y), text, f, color, 5)


def _footer(draw, left: str, right: str, light: bool = False) -> None:
    y = H - PAD - 46
    draw.line([(PAD, y), (W - PAD, y)], fill=L_BORDER if light else LINE, width=3)
    f = font("mono", 30)
    color = L_BLUE if light else DIM
    draw.text((PAD, y + 22), ellipsize(left, f, 640 if right else W - 2 * PAD), font=f, fill=color)
    if right:
        draw.text((W - PAD, y + 22), right, font=f, fill=color, anchor="ra")


def _round_mask(size: tuple[int, int], radius: int) -> Image.Image:
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, size[0] - 1, size[1] - 1], radius=radius, fill=255)
    return mask


def _cover_top(src: Image.Image, w: int, h: int) -> Image.Image:
    """CSS object-fit: cover; object-position: top."""
    scale = max(w / src.width, h / src.height)
    img = src.resize((max(w, round(src.width * scale)), max(h, round(src.height * scale))), Image.LANCZOS)
    left = (img.width - w) // 2
    return img.crop((left, 0, left + w, h))


def _silhouette(w: int, h: int, bg=SIL_BG, fg=SIL_FG) -> Image.Image:
    tile = Image.new("RGBA", (w, h), bg)
    d = ImageDraw.Draw(tile)
    r = min(w, h) * 0.2
    cx, cy = w / 2, h * 0.40
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fg)
    d.ellipse([cx - r * 2.1, h * 0.68, cx + r * 2.1, h * 0.68 + r * 3.4], fill=fg)
    return tile


def _paste_face(img: Image.Image, data: bytes | None, box: tuple[int, int, int, int], radius: int,
                bg=SIL_BG, fg=SIL_FG) -> None:
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    src = _image(data)
    tile = Image.new("RGBA", (w, h), bg)
    if src is None:
        tile = _silhouette(w, h, bg, fg)
    else:
        face = _cover_top(src, w, h)
        tile.alpha_composite(face)
    img.paste(tile, (x0, y0), _round_mask((w, h), radius))


def _paste_logo(img: Image.Image, draw, data: bytes | None, center: tuple[int, int], size: int, abbr: str,
                light: bool = False) -> None:
    cx, cy = center
    src = _image(data)
    if src is not None:
        src.thumbnail((size, size), Image.LANCZOS)
        img.alpha_composite(src, (cx - src.width // 2, cy - src.height // 2))
        return
    draw.ellipse([cx - size // 2, cy - size // 2, cx + size // 2, cy + size // 2],
                 fill=(255, 255, 255) if light else SIL_BG, outline=L_BORDER if light else LINE, width=3)
    f = font("sans_bold", max(14, int(size * 0.3)))
    draw.text((cx, cy), ellipsize(abbr or "?", f, size * 0.8), font=f, fill=L_TEXT if light else MUTED,
              anchor="mm")


def _tiles(draw, y: int, items: list[tuple[str, str, tuple]], h: int = 150, gap: int = 18,
           x0: int = PAD, x1: int = W - PAD) -> None:
    n = len(items)
    w = (x1 - x0 - gap * (n - 1)) / n
    fv, fl = font("mono_bold", 50 if n <= 3 else 46), font("mono", 24)
    for i, (value, label, color) in enumerate(items):
        a = x0 + i * (w + gap)
        draw.rounded_rectangle([a, y, a + w, y + h], radius=22, fill=TILE, outline=LINE, width=3)
        draw.text((a + w / 2, y + h * 0.40), ellipsize(value, fv, w - 20), font=fv, fill=color, anchor="mm")
        lw = _spaced_width(label, fl, 2)
        _spaced(draw, (a + (w - lw) / 2, y + h * 0.68), label, fl, DIM, 2)


def _bar(draw, x: int, y: int, frac: float | None, color, w: int = 150, h: int = 16) -> None:
    draw.rounded_rectangle([x, y, x + w, y + h], radius=h // 2, fill=LINE)
    if frac:
        fw = max(h, round(w * max(0.0, min(1.0, frac))))
        draw.rounded_rectangle([x, y, x + fw, y + h], radius=h // 2, fill=color)


def _chip(draw, x: int, y: int, text: str, fg=GREEN, bg=CHIP_BG, size: int = 24) -> int:
    f = font("mono_bold", size)
    tw = _spaced_width(text, f, 2)
    draw.rounded_rectangle([x, y, x + tw + 32, y + size + 18], radius=(size + 18) // 2, fill=bg)
    _spaced(draw, (x + 16, y + 7), text, f, fg, 2)
    return int(tw + 32)


def _section(draw, y: int, text: str, color=DIM) -> int:
    f = font("mono_bold", 28)
    return int(_spaced(draw, (PAD, y), text, f, color, 5))


# ---- blind play -------------------------------------------------------------------------------

def _panel_dict(panel) -> dict | None:
    if panel is None:
        return None
    if dataclasses.is_dataclass(panel):
        return dataclasses.asdict(panel)
    return dict(panel)


def _toward(rank: int, n: int, high_is_good: bool) -> float:
    """How much a 1..n rank helps the bet, 0..1."""
    return rank / n if high_is_good else (n + 1 - rank) / n


def _matchup_rows(card: BlindCard, p: dict) -> list[tuple[str, str, str | None, float | None, tuple]]:
    """(label, value, dim note, bar fraction or None, bar color). Exact ranks, as on the mockup."""
    over = card.side == "over"
    rows = []
    if card.sport == "nfl":
        pos = (card.position or "").upper()
        r = p.get("opp_rank")
        if r:
            rows.append((f"Opp. vs {pos + 's' if pos else 'position'} (yds allowed)", f"{ordinal(r)} most", None,
                         _toward(r, 32, not over), GREEN))
        share = p.get("share_l4")
        if share is not None:
            what = "Carry" if pos == "RB" else "Target"
            rows.append((f"{what} share (L4)", f"{share * 100:.1f}%", None, None, GREEN))
        snap = p.get("snap_share")
        if snap is not None:
            rows.append(("Snap share", f"{snap * 100:.1f}%", None, None, GREEN))
        return rows
    group = POSITION_GROUP.get((card.position or "").upper(), "position")
    vs, drtg, pace = p.get("opp_vs_pos_rank") or 0, p.get("drtg_rank") or 0, p.get("pace_rank") or 0
    stat = _market_short({"market_label": card.market_label}).upper()
    rows.append((f"Opp. vs {group} ({stat} allowed)", ordinal(vs) if vs else "—", None,
                 _toward(vs, 30, over) if vs else None, GREEN))
    rows.append(("Opp. defensive rating", ordinal(drtg) if drtg else "—", None,
                 _toward(drtg, 30, over) if drtg else None, LIME))
    rows.append(("Opp. pace", f"{ordinal(pace)} fastest" if pace else "—", None,
                 _toward(pace, 30, not over) if pace else None, GREEN))
    if _num(p.get("usage_l10")) is not None:
        u, ch = _num(p["usage_l10"]), _num(p.get("usage_change")) or 0.0
        note = "steady" if abs(ch) < 0.5 else f"{'↑' if ch > 0 else '↓'} from {u - ch:.1f}%"
        rows.append(("Usage rate (L10)", f"{u:.1f}%", note, None, GREEN))
    if _num(p.get("minutes_l5")) is not None:
        rows.append(("Minutes (L5)", f"{_num(p['minutes_l5']):.1f}", None, None, GREEN))
    return rows


def render_blind(card: BlindCard) -> bytes:
    """The hook: the bet and its numbers, no player, team or opponent anywhere."""
    img, d = _canvas()
    sport = (card.sport or "").upper()
    _kicker(d, " ".join(s for s in ("TRULINE ·", sport, "BLIND PLAY", f"#{card.n}" if card.n is not None else "")
                        if s))

    # silhouette + bet
    top = 140
    size = 230
    d.ellipse([PAD, top, PAD + size, top + size], fill=SIL_BG)
    d.text((PAD + size / 2, top + size / 2), "?", font=font("sans_bold", 130), fill=SIL_FG, anchor="mm")
    x = PAD + size + 44
    max_w = W - PAD - x
    pos = POSITION_NAME.get((card.position or "").upper(), (card.position or "").upper())
    head = " · ".join(s for s in (pos, (card.market_label or "").upper()) if s)
    d.text((x, top + 22), ellipsize(head, font("sans", 36), max_w), font=font("sans", 36), fill=MUTED)
    if card.market_label == "Anytime TD":
        parts = [("YES" if card.side == "over" else "NO", TEXT)]
    else:
        line = _num(card.line)
        parts = [((card.side or "").upper() + " ", TEXT)]
        if line is not None:
            parts.append((f"{line:g}", CYAN))
    parts.append(("  " + _price(card.price), TEXT))
    size = 78
    while size > 40 and sum(font("sans_bold", size).getlength(t) for t, _ in parts) > max_w:
        size -= 4
    fb = font("sans_bold", size)
    cx = x
    for text, color in parts:
        d.text((cx, top + 78), text, font=fb, fill=color)
        cx += fb.getlength(text)
    when = "tonight" if card.tip_label == "Tonight" else card.tip_label
    verb = "kicks off" if card.sport == "nfl" else "tips"
    sub = " · ".join(s for s in (f"{card.n_books} books priced" if card.n_books else "",
                                 f"{verb} {when}" if when else "") if s)
    d.text((x, top + 182), ellipsize(sub, font("sans", 32), max_w), font=font("sans", 32), fill=MUTED)

    hits, games, avg = _num(card.l10_hits), _num(card.l10_games), _num(card.l10_avg)
    form = f"{int(hits)}/{int(games)}" if hits is not None and games else "—"
    form_label, avg_label = form_labels(games)
    _tiles(d, 420, [(_pct(card.ev, signed=True), "EDGE", CYAN),
                    (str(card.n_books) if card.n_books is not None else "—", "BOOKS", TEXT),
                    (form, form_label, TEXT), (f"{avg:g}" if avg is not None else "—", avg_label, TEXT)])

    p = _panel_dict(card.panel)
    if p:
        y = 640
        w = _section(d, y + 6, "THE MATCHUP")
        if p.get("favorable"):
            _chip(d, PAD + w + 24, y, "FAVORABLE")
        y += 80
        tags = [t for t in (p.get("tags") or []) if t]
        for i, (label, value, note, frac, color) in enumerate(_matchup_rows(card, p)):
            _matchup_row(d, y, label, value, note, frac, color, highlight=i == 0 and frac is not None)
            y += 92
        if tags:
            y += 24
            cx = PAD
            for t in tags[:4]:
                text = TAG_LABEL.get(t, t).upper()
                if cx + _spaced_width(text, font("mono_bold", 24), 2) + 32 > W - PAD:
                    break
                cx += _chip(d, cx, y, text) + 14
    _footer(d, "Reveal → truline.app/blind", "@trulineapp")
    return _png(img)


def _matchup_row(d, y: int, label: str, value: str, note: str | None, frac: float | None, color,
                 highlight: bool) -> None:
    fl, fv, fn = font("mono", 31), font("sans_bold", 36), font("sans", 28)
    right = W - PAD
    if frac is not None:
        _bar(d, right - 150, y + 20, frac, color)
        right -= 150 + 22
    if note:
        d.text((right, y + 30), note, font=fn, fill=DIM, anchor="rm")
        right -= fn.getlength(note) + 16
    d.text((right, y + 28), value, font=fv, fill=GREEN if highlight else TEXT, anchor="rm")
    right -= fv.getlength(value) + 30
    d.text((PAD, y + 28), ellipsize(label, fl, right - PAD), font=fl, fill=TEXT, anchor="lm")
    d.line([(PAD, y + 70), (W - PAD, y + 70)], fill=LINE, width=2)


# ---- cheat sheet and value plays --------------------------------------------------------------

def _list_header(d, kicker: str, title: str, left: str, right: str) -> int:
    _kicker(d, kicker)
    d.text((PAD, 112), title, font=font("sans_bold", 70), fill=TEXT)
    y = 222
    f = font("mono_bold", 26)
    _spaced(d, (PAD, y), left, f, DIM, 3)
    _spaced(d, (W - PAD - _spaced_width(right, f, 3), y), right, f, DIM, 3)
    d.line([(PAD, y + 48), (W - PAD, y + 48)], fill=LINE, width=2)
    return y + 60


def _prop_row(img, d, y: int, h: int, row: dict, face: bytes | None, logos: dict, sport: str,
              right_parts: list[tuple[str, tuple, ImageFont.FreeTypeFont]], prefix: str | None = None,
              sub_extra: str | None = None) -> None:
    """One list row: [rank] face | name / bet · matchup | right-aligned figures."""
    x = PAD
    if prefix:
        d.text((x, y + h / 2), prefix, font=font("mono_bold", 36), fill=DIM, anchor="lm")
        x += 54
    fh = h - 22
    fw = round(fh * (1.33 if fh < 110 else 1.15))
    _paste_face(img, face, (x, y + 11, x + fw, y + 11 + fh), radius=14)
    x += fw + 24

    # right side, drawn right to left
    rx = W - PAD
    for text, color, f in reversed(right_parts):
        d.text((rx, y + h / 2), text, font=f, fill=color, anchor="rm")
        rx -= f.getlength(text) + 26
    max_w = rx - x - 10

    big = h >= 120
    fs = font("mono", 27 if big else 24)
    # the bet is never cut: shrink the line first, then shorten only the name
    bet = "  " + _bet_short(row)
    name = short_name(row.get("player") or "")
    size = 38 if big else 33
    while size > 26 and font("sans_bold", size).getlength(name + bet) > max_w:
        size -= 2
    fn = font("sans_bold", size)
    name = ellipsize(name, fn, max(40, max_w - fn.getlength(bet)))
    d.text((x, y + h * 0.36), name + bet, font=fn, fill=TEXT, anchor="lm")
    sy = y + h * 0.72
    sx = x
    team = row.get("team")
    if team and logos.get(team):
        ls = 34 if big else 30
        _paste_logo(img, d, logos.get(team), (sx + ls // 2, round(sy)), ls, team)
        sx += ls + 10
    ab = _matchup_abbrs(sport, row.get("matchup"))
    sub = " · ".join(s for s in ((f"{ab[0]} @ {ab[1]}" if ab else None), sub_extra) if s)
    d.text((sx, sy), ellipsize(sub, fs, max_w - (sx - x)), font=fs, fill=MUTED, anchor="lm")
    d.line([(PAD, y + h), (W - PAD, y + h)], fill=LINE, width=2)


def _tailwind(row: dict) -> str | None:
    for t in row.get("tags") or []:
        if t.get("direction") == "tailwind":
            return TAG_LABEL.get(t.get("key"), t.get("key"))
    return None


def render_cheatsheet(sport: str, slate: date, rows: list[dict], faces: dict[str, bytes | None],
                      *, logos: dict[str, bytes | None] | None = None) -> bytes:
    """Hit-rate leaders among tonight's +EV props: face, bet, L10 hits, market fair %."""
    img, d = _canvas()
    logos, faces = logos or {}, faces or {}
    y = _list_header(d, f"TRULINE · {(sport or '').upper()} CHEAT SHEET · {_md(slate)}", "Hit-rate leaders",
                     "PROP", "HIT RATE   FAIR")
    rows = (rows or [])[:10]
    avail = H - PAD - 46 - 30 - y
    h = min(150, avail // max(1, len(rows)))
    fr = font("mono_bold", 34 if h >= 110 else 30)
    fd = font("mono", 30 if h >= 110 else 27)
    for r in rows:
        l10 = r.get("l10") or {}
        hits, games = _num(l10.get("hits")), _num(l10.get("games"))
        rate = hits / games if hits is not None and games else None
        hit_txt = f"{int(hits)}/{int(games)}" if rate is not None else "—"
        fair_p = _num(r.get("fair_prob"))
        fair = f"{round(fair_p * 100)}%" if fair_p is not None else "—"
        _prop_row(img, d, y, h, r, faces.get(r.get("player")), logos, sport,
                  [(hit_txt, GREEN if rate is not None and rate >= 0.7 else TEXT, fr), (fair, DIM, fd)],
                  sub_extra=_tailwind(r))
        y += h
    _footer(d, "truline.app", "@trulineapp")
    return _png(img)


def render_value(sport: str, slate: date, legs: list[dict], faces, locked_at: datetime,
                 *, logos: dict[str, bytes | None] | None = None) -> bytes:
    """Tonight's Top 5 at their locked prices: face, bet, best book, price, edge."""
    img, d = _canvas()
    logos = logos or {}
    y = _list_header(d, f"TRULINE · TONIGHT'S TOP 5 · {_md(slate)}", "Value plays", "BET", "PRICE · EDGE")
    legs = (legs or [])[:5]
    h = 160
    fp, fe = font("mono_bold", 36), font("mono_bold", 36)
    for i, r in enumerate(legs):
        book = BOOK_NAMES.get(r.get("book"), r.get("book") or "")
        _prop_row(img, d, y, h, r, (faces or {}).get(r.get("player")), logos, sport,
                  [(_price(r.get("price")), TEXT, fp), ("·", DIM, fp), (_pct(r.get("ev")), CYAN, fe)],
                  prefix=str(r.get("rank") or i + 1), sub_extra=book)
        y += h
    note = "best book on each row · 1 unit each"
    if locked_at is not None:
        note = f"Locked {locked_at.astimezone(ET).strftime('%I:%M %p').lstrip('0')} ET · {note}"
    else:
        note = note[0].upper() + note[1:]
    d.text((PAD, y + 34), ellipsize(note, font("mono", 27), W - 2 * PAD), font=font("mono", 27), fill=MUTED)
    _footer(d, "All plays + odds → truline.app", "@trulineapp")
    return _png(img)


# ---- final recap ------------------------------------------------------------------------------

def _result_mark(result: str | None) -> tuple[str, tuple]:
    return {"win": ("W", L_WIN), "loss": ("L", L_LOSS), "push": ("P", L_DIM),
            "void": ("VOID", L_DIM)}.get(result or "", ("—", L_FADED))


def render_final(event: dict, records: dict, logos: dict[str, bytes | None], plays: list[dict],
                 faces) -> bytes:
    """Light card: logos, nicknames, records, score, and every TruLine posted play in the game."""
    img, d = _canvas(light=True)
    event = event or {}
    sport = event.get("sport")
    logos, faces, plays = logos or {}, faces or {}, plays or []
    plays = [{**p, "profit_units": _num(p.get("profit_units"))} for p in plays]
    _kicker(d, "FINAL", L_BLUE)

    teams = [(event.get("away_team"), _num(event.get("away_score"))),
             (event.get("home_team"), _num(event.get("home_score")))]
    centers = [W * 0.27, W * 0.73]
    fnick, frec = font("sans_bold", 50), font("sans", 32)
    for (name, _), cx in zip(teams, centers):
        ab = _abbr(sport, name)
        _paste_logo(img, d, logos.get(ab), (round(cx), 270), 190, ab, light=True)
        d.text((cx, 400), ellipsize((_nick(name) or ab).upper(), fnick, 420), font=fnick, fill=L_TEXT,
               anchor="mt")
        rec = record_label(records, ab)
        if rec:
            d.text((cx, 462), rec, font=frec, fill=L_DIM, anchor="mt")

    fs = font("sans_bold", 132)
    (a_name, a_pts), (h_name, h_pts) = teams
    d.text((W / 2, 610), "—", font=fs, fill=L_FADED, anchor="mm")
    d.text((W / 2 - 90, 610), "" if a_pts is None else f"{a_pts:g}", font=fs, fill=L_TEXT, anchor="rm")
    d.text((W / 2 + 90, 610), "" if h_pts is None else f"{h_pts:g}", font=fs, fill=L_TEXT, anchor="lm")
    if a_pts is not None and h_pts is not None:
        if a_pts == h_pts:
            verdict = "TIE"
        else:
            winner = a_name if a_pts > h_pts else h_name
            verdict = f"{(_nick(winner) or _abbr(sport, winner)).upper()} WIN"
        d.text((W / 2, 715), ellipsize(verdict, font("sans_bold", 42), W - 2 * PAD), font=font("sans_bold", 42),
               fill=L_TEXT, anchor="mm")

    box_top = 790
    row_h = 84 if len(plays) <= 3 else 64
    room = H - PAD - 46 - 30 - box_top - 76 - 76  # inside the box: rows (+ "more" line) between header and total
    fit = room // row_h if len(plays) * row_h <= room else (room - 40) // row_h
    shown = plays[:max(1, min(MAX_FINAL_PLAYS, fit))]
    box_bottom = box_top + 76 + row_h * len(shown) + (40 if len(plays) > len(shown) else 0) + 76
    d.rounded_rectangle([PAD, box_top, W - PAD, box_bottom], radius=26, fill=(255, 255, 255), outline=L_BORDER,
                        width=3)
    ix0, ix1 = PAD + 30, W - PAD - 30
    _spaced(d, (ix0, box_top + 28), "TRULINE PLAYS THIS GAME", font("mono_bold", 26), L_DIM, 4)
    y = box_top + 76
    fp = font("sans", 34 if row_h > 70 else 30)
    fm = font("sans_bold", 36 if row_h > 70 else 32)
    for p in shown:
        fh = row_h - 18
        _paste_face(img, faces.get(p.get("player")), (ix0, y + 9, ix0 + round(fh * 1.3), y + 9 + fh), radius=10,
                    bg=L_BOTTOM, fg=L_FADED)
        tx = ix0 + round(fh * 1.3) + 20
        mark, color = _result_mark(p.get("result"))
        d.text((ix1, y + row_h / 2), mark, font=fm, fill=color, anchor="rm")
        bet = f" {_bet_short(p)} {_price(p.get('price'))}"
        name = ellipsize(last_name(p.get("player") or ""), fp,
                         max(40, ix1 - tx - fm.getlength(mark) - 24 - fp.getlength(bet)))
        d.text((tx, y + row_h / 2), name + bet, font=fp, fill=L_TEXT, anchor="lm")
        y += row_h
    if len(plays) > len(shown):
        d.text((ix0, y + 8), f"+{len(plays) - len(shown)} more", font=font("sans", 28), fill=L_DIM)
        y += 40
    w, l, pu, units = record_line(plays)
    if w + l + pu:
        rec = f"{w}-{l}" + (f"-{pu}" if pu else "")
        d.line([(ix0, y + 10), (ix1, y + 10)], fill=L_BORDER, width=2)
        d.text((ix0, y + 44), "Game total", font=font("sans", 32), fill=L_DIM, anchor="lm")
        d.text((ix1, y + 44), f"{rec}  {units:+.2f}u", font=font("sans_bold", 34),
               fill=L_WIN if units > 0 else L_LOSS if units < 0 else L_DIM, anchor="rm")
    _footer(d, "truline.app", "@trulineapp", light=True)
    return _png(img)


# ---- receipt ----------------------------------------------------------------------------------

def _stamp(img: Image.Image) -> None:
    band = Image.new("RGBA", (620, 84), GREEN)
    bd = ImageDraw.Draw(band)
    f = font("sans_bold", 44)
    tw = _spaced_width("CASHED", f, 4)
    _spaced(bd, ((620 - tw) / 2, 14), "CASHED", f, BG_BOTTOM, 4)
    rot = band.rotate(-35, resample=Image.BICUBIC, expand=True)
    img.alpha_composite(rot, (W - rot.width + 150, -110))


def _receipt_third_tile(play: dict) -> tuple[str, str, tuple] | None:
    p = _panel_dict(play.get("panel")) or {}
    pos = (play.get("position") or "").upper()
    if p.get("opp_vs_pos_rank"):
        return ordinal(int(p["opp_vs_pos_rank"])), f"OPP vs {pos or 'POS'}"[:12], CYAN
    if p.get("opp_rank"):
        return f"{ordinal(int(p['opp_rank']))}", f"OPP vs {pos or 'POS'}"[:12], CYAN
    if _num(play.get("ev")) is not None:
        return _pct(play["ev"], signed=True), "EDGE", CYAN
    return None


def render_receipt(play: dict, face: bytes | None, logo: bytes | None, original: bytes | None = None) -> bytes:
    """A win: CASHED stamp, face, name, logo + matchup, bet -> stat, price/units, and the original card."""
    img, d = _canvas()
    play = play or {}
    sport = play.get("sport")
    day = play.get("slate_date")
    _kicker(d, "TRULINE · RECEIPT" + (f" · {_md(day)}" if isinstance(day, date) else ""),
            max_w=W - 2 * PAD - STAMP_RESERVE)

    top = 140
    fw, fh = 400, 292
    _paste_face(img, face, (PAD, top, PAD + fw, top + fh), radius=30)
    x = PAD + fw + 40
    max_w = W - PAD - x
    ty = top + 40
    if play.get("kind") == "blind" and play.get("n") is not None:
        lead = f"BLIND PLAY #{play['n']} WAS"
    elif play.get("kind") == "value":
        lead = "FROM TONIGHT'S TOP 5"
    else:
        lead = None
    if lead:
        _spaced(d, (x, ty), ellipsize(lead, font("mono_bold", 24), max_w), font("mono_bold", 24), MUTED, 3)
        ty += 48
    fn = font("sans_bold", 60)
    name = play.get("player") or ""
    if fn.getlength(name) > max_w:
        fn = font("sans_bold", 48)
    d.text((x, ty), ellipsize(name, fn, max_w), font=fn, fill=TEXT)
    ty += 92
    ab = _matchup_abbrs(sport, play.get("matchup"))
    lx = x
    team = play.get("team")
    if team:
        _paste_logo(img, d, logo, (lx + 28, ty + 22), 56, team)
        lx += 72
    if ab:
        d.text((lx, ty + 22), ellipsize(f"{ab[0]} @ {ab[1]}", font("sans", 36), W - PAD - lx),
               font=font("sans", 36), fill=MUTED, anchor="lm")

    y = top + fh + 80
    fb, fr = font("sans", 56), font("sans_bold", 56)
    actual = play.get("actual")
    left = f"{_bet_short(play)}  →  "
    actual = _num(actual)
    right = f"{actual:g} ✓" if actual is not None else "HIT ✓"
    total = fb.getlength(left) + fr.getlength(right)
    if total > W - 2 * PAD:
        fb, fr = font("sans", 44), font("sans_bold", 44)
        total = fb.getlength(left) + fr.getlength(right)
    sx = (W - total) / 2
    d.text((sx, y), left, font=fb, fill=TEXT, anchor="ls")
    d.text((sx + fb.getlength(left), y), right, font=fr, fill=GREEN, anchor="ls")
    if play.get("stat_line"):
        fsl = font("sans", 32)
        d.text((W / 2, y + 50), ellipsize(str(play["stat_line"]), fsl, W - 2 * PAD), font=fsl, fill=MUTED,
               anchor="mt")

    units = _num(play.get("profit_units"))
    tiles = [(_price(play.get("price")), "PRICE", TEXT),
             (f"{units:+.2f}u" if units is not None else "WIN", "RESULT", GREEN)]
    third = _receipt_third_tile(play)
    if third:
        tiles.append(third)
    _tiles(d, y + 130, tiles)

    orig = _image(original)
    if orig is not None:
        oy = y + 130 + 150 + 50
        f = font("mono_bold", 24)
        bottom = H - PAD - 46 - 30
        th = bottom - oy - 44
        if th > 120:
            label = "THE ORIGINAL CALL"
            lw = _spaced_width(label, f, 4)
            _spaced(d, ((W - lw) / 2, oy), label, f, DIM, 4)
            orig.thumbnail((round(th * W / H), th), Image.LANCZOS)
            ox = (W - orig.width) // 2
            d.rounded_rectangle([ox - 3, oy + 41, ox + orig.width + 2, oy + 44 + orig.height + 2], radius=14,
                                outline=LINE, width=3)
            img.paste(orig, (ox, oy + 44), _round_mask(orig.size, 12))
    _stamp(img)
    _footer(d, "Every result, wins + losses → truline.app/results", "")
    return _png(img)


# ---- sample CLI -------------------------------------------------------------------------------

def _samples(assets: Path | None) -> dict[str, bytes]:
    from src.edge.social.nba_matchup import Panel
    from src.edge.social.nfl_matchup import NflPanel

    def load(name: str) -> bytes | None:
        p = assets / name if assets else None
        return p.read_bytes() if p and p.exists() else None

    def face(player: str) -> bytes | None:
        return load(f"face_{player.replace(' ', '_')}.png")

    slate = date(2026, 11, 15)
    locked = datetime(2026, 11, 15, 17, 10, tzinfo=timezone.utc)
    names = ["Jalen Brunson", "Nikola Jokić", "Tyrese Maxey", "Scottie Barnes", "Darius Garland",
             "Pascal Siakam", "Giannis Antetokounmpo", "Shai Gilgeous-Alexander", "Karl-Anthony Towns",
             "Jaren Jackson Jr."]
    teams = ["NYK", "DEN", "PHI", "TOR", "CLE", "IND", "MIL", "OKC", "NYK", "MEM"]
    matchups = ["New York Knicks @ Boston Celtics", "Denver Nuggets @ Utah Jazz", "Indiana Pacers @ Philadelphia 76ers",
                "Toronto Raptors @ Golden State Warriors", "Cleveland Cavaliers @ Atlanta Hawks",
                "Indiana Pacers @ Philadelphia 76ers", "Milwaukee Bucks @ Chicago Bulls",
                "Oklahoma City Thunder @ Portland Trail Blazers", "New York Knicks @ Boston Celtics",
                "Memphis Grizzlies @ Houston Rockets"]
    markets = [("player_points", "Points", 24.5), ("player_rebounds", "Rebounds", 11.5),
               ("player_threes", "Threes", 2.5), ("player_rebounds", "Rebounds", 7.5),
               ("player_rebounds", "Rebounds", 2.5), ("player_points_rebounds_assists", "Pts+Reb+Ast", 32.5),
               ("player_points", "Points", 30.5), ("player_assists", "Assists", 6.5),
               ("player_points_rebounds", "Pts+Reb", 36.5), ("player_points", "Points", 21.5)]
    rows = []
    for i, (n, t, m, (mk, ml, line)) in enumerate(zip(names, teams, matchups, markets)):
        rows.append({"player": n, "team": t, "matchup": m, "market": mk, "market_label": ml, "side": "over",
                     "line": line, "price": [-110, 105, 116, -120, 135, 116, -105, 110, 120, 150][i],
                     "fair_prob": [.58, .55, .52, .54, .45, .49, .56, .51, .50, .44][i],
                     "ev": [.055, .05, .049, .046, .043, .041, .038, .036, .033, .031][i],
                     "book": ["fanduel", "draftkings", "betmgm", "fanatics", "espnbet"][i % 5], "rank": i + 1,
                     "l10": {"hits": [9, 9, 8, 8, 8, 7, 7, 7, 6, 6][i], "games": 10}})
    logos = {t: load(f"logo_{t}.png") for t in set(teams) | {"PHI", "BOS"}}
    faces = {n: face(n) for n in names}

    nba = BlindCard(sport="nba", n=1, position="F", market_label="Pts+Reb+Ast", side="over", line=32.5, price=116,
                    ev=0.05, n_books=9, l10_hits=5, l10_games=10, l10_avg=34.1, tip_label="1:10 PM ET",
                    panel=dataclasses.asdict(Panel(True, 27, 22, 6, 26.8, 2.7, 35.2)))
    nfl = BlindCard(sport="nfl", n=4, position="WR", market_label="Rec Yds", side="over", line=64.5, price=-110,
                    ev=0.042, n_books=8, l10_hits=6, l10_games=9, l10_avg=71.2, tip_label="Tonight",
                    panel=NflPanel(True, 3, 0.274, None, ["def_soft"]))
    blind_png = render_blind(nba)
    event = {"sport": "nba", "away_team": "Indiana Pacers", "home_team": "Philadelphia 76ers",
             "away_score": 112, "home_score": 118}
    records = {"IND": {"w": 7, "l": 6, "t": 0, "games": 13, "label": "7-6"},
               "PHI": {"w": 8, "l": 5, "t": 0, "games": 13, "label": "8-5"}}
    siakam = {"kind": "blind", "sport": "nba", "n": 1, "player": "Pascal Siakam", "team": "IND",
              "matchup": "Indiana Pacers @ Philadelphia 76ers", "market": "player_points_rebounds_assists",
              "market_label": "Pts+Reb+Ast", "side": "over", "line": 32.5, "price": 116, "result": "win",
              "profit_units": 1.16, "actual": 36, "stat_line": "24 pts · 7 reb · 5 ast", "slate_date": slate,
              "panel": {"opp_vs_pos_rank": 27}, "position": "F"}
    maxey = {**siakam, "kind": "value", "player": "Tyrese Maxey", "team": "PHI", "market": "player_threes",
             "market_label": "Threes", "line": 2.5, "price": -120, "result": "loss", "profit_units": -1.0}
    long_name = "Giannis Antetokounmpo-Wolfeschlegelsteinhausen Jr."
    long_rows = [{**r, "player": long_name, "matchup": "Golden State Warriors @ Portland Trail Blazers"}
                 for r in rows]
    return {
        "blind_nba.png": blind_png,
        "blind_nfl.png": render_blind(nfl),
        "cheatsheet.png": render_cheatsheet("nba", slate, rows, faces, logos=logos),
        "value.png": render_value("nba", slate, rows[:5], faces, locked, logos=logos),
        "final.png": render_final(event, records, logos, [siakam, maxey], faces),
        "receipt.png": render_receipt(siakam, face("Pascal Siakam"), logos.get("IND"), blind_png),
        "receipt_no_original.png": render_receipt(siakam, None, None),
        "long_names_cheatsheet.png": render_cheatsheet("nba", slate, long_rows, {}),
        "long_names_receipt.png": render_receipt({**siakam, "player": long_name}, None, None),
        "long_names_final.png": render_final(event, {}, {}, [{**siakam, "player": long_name}] * 8, {}),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Render sample social cards from fixture data.")
    ap.add_argument("--sample", required=True, metavar="OUT_DIR", help="directory to write the PNGs to")
    ap.add_argument("--assets", metavar="DIR", help="optional face_<First_Last>.png / logo_<ABBR>.png files")
    args = ap.parse_args(argv)
    out = Path(args.sample)
    out.mkdir(parents=True, exist_ok=True)
    for name, data in _samples(Path(args.assets) if args.assets else None).items():
        (out / name).write_bytes(data)
        print(f"{out / name}  {len(data) // 1024} KB")


if __name__ == "__main__":
    main()
