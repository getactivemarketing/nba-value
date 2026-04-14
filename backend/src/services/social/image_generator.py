"""Generate social media card images for TruLine tweets."""

import io
import urllib.request
from pathlib import Path
from typing import Literal

import structlog

logger = structlog.get_logger()


# Brand colors — light theme
BG = (248, 250, 252)           # slate-50 (base)
SURFACE = (226, 232, 240)      # slate-200 (outlines)
CARD = (255, 255, 255)         # pure white (surfaces)
ACCENT = (37, 99, 235)         # blue-600 (links, labels)
GREEN = (5, 150, 105)          # emerald-600 (wins, NRFI hit)
AMBER = (180, 83, 9)           # amber-700 (YRFI, caution)
RED = (185, 28, 28)            # red-700 (losses)
WHITE = (15, 23, 42)           # slate-900 (primary text — name kept as "WHITE" for compat)
MUTED = (71, 85, 105)          # slate-600 (secondary text)
DIM = (148, 163, 184)          # slate-400 (labels)
GRID = (226, 232, 240)         # slate-200 (legacy — kept for nba card)

# Team brand colors: (primary_rgb, secondary_rgb)
MLB_TEAM_COLORS = {
    "ARI": ((167, 25, 48), (227, 212, 173)),
    "ATL": ((206, 17, 65), (19, 39, 79)),
    "BAL": ((223, 70, 1), (39, 37, 31)),
    "BOS": ((189, 48, 57), (12, 35, 64)),
    "CHC": ((14, 51, 134), (204, 52, 51)),
    "CWS": ((39, 37, 31), (196, 206, 212)),
    "CIN": ((198, 1, 31), (0, 0, 0)),
    "CLE": ((0, 56, 93), (227, 25, 55)),
    "COL": ((51, 0, 111), (196, 206, 212)),
    "DET": ((12, 35, 64), (250, 70, 22)),
    "HOU": ((0, 45, 98), (235, 110, 31)),
    "KC": ((0, 70, 135), (189, 155, 96)),
    "LAA": ((186, 0, 33), (0, 50, 99)),
    "LAD": ((0, 90, 156), (239, 62, 66)),
    "MIA": ((0, 163, 224), (239, 51, 64)),
    "MIL": ((18, 40, 75), (182, 146, 46)),
    "MIN": ((0, 43, 92), (211, 17, 69)),
    "NYM": ((0, 45, 114), (252, 89, 16)),
    "NYY": ((0, 48, 135), (196, 206, 212)),
    "OAK": ((0, 56, 49), (239, 178, 30)),
    "PHI": ((232, 24, 40), (0, 45, 114)),
    "PIT": ((253, 184, 39), (39, 37, 31)),
    "SD": ((47, 36, 28), (255, 196, 37)),
    "SF": ((253, 90, 30), (39, 37, 31)),
    "SEA": ((0, 92, 92), (196, 206, 212)),
    "STL": ((196, 30, 58), (12, 35, 64)),
    "TB": ((9, 44, 92), (143, 188, 230)),
    "TEX": ((0, 50, 120), (192, 17, 31)),
    "TOR": ((19, 74, 142), (232, 41, 28)),
    "WSH": ((171, 0, 3), (20, 34, 90)),
}

NBA_TEAM_COLORS = {
    "ATL": ((225, 68, 52), (196, 214, 0)),
    "BOS": ((0, 122, 51), (255, 255, 255)),
    "BKN": ((0, 0, 0), (255, 255, 255)),
    "CHA": ((29, 17, 96), (0, 120, 140)),
    "CHI": ((206, 17, 65), (0, 0, 0)),
    "CLE": ((134, 0, 56), (253, 187, 48)),
    "DAL": ((0, 83, 188), (0, 43, 92)),
    "DEN": ((13, 34, 64), (255, 198, 39)),
    "DET": ((200, 16, 46), (29, 66, 138)),
    "GSW": ((29, 66, 138), (255, 199, 44)),
    "HOU": ((206, 17, 65), (0, 0, 0)),
    "IND": ((0, 45, 98), (253, 187, 48)),
    "LAC": ((200, 16, 46), (29, 66, 148)),
    "LAL": ((85, 37, 130), (253, 185, 39)),
    "MEM": ((93, 118, 169), (18, 23, 63)),
    "MIA": ((152, 0, 46), (249, 160, 27)),
    "MIL": ((0, 71, 27), (240, 235, 210)),
    "MIN": ((12, 35, 64), (35, 97, 146)),
    "NOP": ((0, 22, 65), (225, 58, 62)),
    "NYK": ((0, 107, 182), (245, 132, 38)),
    "OKC": ((0, 125, 195), (239, 59, 36)),
    "ORL": ((0, 125, 197), (196, 206, 211)),
    "PHI": ((0, 107, 182), (237, 23, 76)),
    "PHX": ((29, 17, 96), (229, 95, 32)),
    "POR": ((224, 58, 62), (0, 0, 0)),
    "SAC": ((91, 43, 130), (99, 113, 122)),
    "SAS": ((196, 206, 211), (0, 0, 0)),
    "TOR": ((206, 17, 65), (0, 0, 0)),
    "UTA": ((0, 43, 92), (249, 160, 27)),
    "WAS": ((0, 43, 92), (227, 24, 55)),
}


def _get_team_color(team_abbr: str, sport: str = "mlb") -> tuple[tuple, tuple]:
    """Return (primary, secondary) color tuple for a team. Falls back to accent colors."""
    colors = MLB_TEAM_COLORS if sport == "mlb" else NBA_TEAM_COLORS
    return colors.get(team_abbr.upper(), (ACCENT, MUTED))


def _draw_gradient_bg_fast(img, W: int, H: int):
    """Light gradient: pure white center fading to cool slate-100 at top/bottom edges."""
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    center_y = H // 2
    for y in range(H):
        dy = abs(y - center_y) / center_y  # 0 at center, 1 at edges
        t = dy ** 1.5                       # sharper falloff
        # White (255,255,255) → cool slate (226,232,240)
        r = int(255 - 29 * t)
        g = int(255 - 23 * t)
        b = int(255 - 15 * t)
        draw.line([(0, y), (W, y)], fill=(r, g, b))


def _draw_shadow(img, xy, radius=20, shadow_offset=8, shadow_blur=12, shadow_color=(0, 0, 0, 40)):
    """Draw a soft blurred drop shadow beneath a rounded-rect region.

    Call BEFORE drawing the card surface so the shadow sits underneath.
    Requires img to be in RGBA mode; caller handles mode conversion.
    """
    from PIL import Image, ImageDraw, ImageFilter
    x0, y0, x1, y1 = xy
    pad = shadow_blur * 2
    w = (x1 - x0) + pad * 2
    h = (y1 - y0) + pad * 2
    shadow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    sx0, sy0 = pad, pad
    sx1, sy1 = pad + (x1 - x0), pad + (y1 - y0)
    sd.rectangle([sx0 + radius, sy0, sx1 - radius, sy1], fill=shadow_color)
    sd.rectangle([sx0, sy0 + radius, sx1, sy1 - radius], fill=shadow_color)
    sd.pieslice([sx0, sy0, sx0 + 2 * radius, sy0 + 2 * radius], 180, 270, fill=shadow_color)
    sd.pieslice([sx1 - 2 * radius, sy0, sx1, sy0 + 2 * radius], 270, 360, fill=shadow_color)
    sd.pieslice([sx0, sy1 - 2 * radius, sx0 + 2 * radius, sy1], 90, 180, fill=shadow_color)
    sd.pieslice([sx1 - 2 * radius, sy1 - 2 * radius, sx1, sy1], 0, 90, fill=shadow_color)
    shadow = shadow.filter(ImageFilter.GaussianBlur(shadow_blur))
    img.paste(shadow, (x0 - pad, y0 - pad + shadow_offset), shadow)


def _fit_font(text: str, max_width: int, size_candidates=(72, 64, 56, 48, 42)):
    """Return the largest TrueType font that fits `text` within `max_width` pixels."""
    from PIL import Image, ImageDraw, ImageFont
    bundled = Path(__file__).resolve().parent.parent.parent.parent / "assets" / "fonts" / "DejaVuSans-Bold.ttf"
    font_path = str(bundled) if bundled.exists() else None
    if not font_path:
        # Fall back to the first candidate with PIL default
        return ImageFont.load_default()
    tmp = Image.new("RGB", (10, 10))
    td = ImageDraw.Draw(tmp)
    for size in size_candidates:
        f = ImageFont.truetype(font_path, size)
        bbox = td.textbbox((0, 0), text, font=f)
        if (bbox[2] - bbox[0]) <= max_width:
            return f
    return ImageFont.truetype(font_path, size_candidates[-1])


def _draw_stats_table(draw, x0: int, y0: int, x1: int, y1: int,
                      away_l10: str | None, home_l10: str | None,
                      away_ats: str | None, home_ats: str | None,
                      away_ou: str | None, home_ou: str | None, fonts: dict):
    """Draw a 3-row stats table: [away_value]  LABEL  [home_value]."""
    _draw_rounded_rect(draw, [x0, y0, x1, y1], fill=CARD, outline=SURFACE, radius=16, width=1)

    rows = [
        ("L10", away_l10, home_l10),
        ("ATS", away_ats, home_ats),
        ("O/U", away_ou, home_ou),
    ]
    rows = [r for r in rows if r[1] or r[2]]
    if not rows:
        return

    row_h = (y1 - y0) / len(rows)
    center_x = (x0 + x1) // 2
    away_cx = x0 + (x1 - x0) // 4
    home_cx = x1 - (x1 - x0) // 4

    for i, (label, a_val, h_val) in enumerate(rows):
        cy = int(y0 + row_h * i + row_h / 2)
        draw.text((center_x, cy), label, font=fonts["s"], fill=DIM, anchor="mm")
        if a_val:
            draw.text((away_cx, cy), a_val, font=fonts["m"], fill=WHITE, anchor="mm")
        if h_val:
            draw.text((home_cx, cy), h_val, font=fonts["m"], fill=WHITE, anchor="mm")

    for i in range(1, len(rows)):
        sy = int(y0 + row_h * i)
        draw.line([(x0 + 40, sy), (x1 - 40, sy)], fill=SURFACE, width=1)


def _draw_team_bars(draw, W: int, H: int, away_team: str, home_team: str, sport: str = "mlb"):
    """Draw team-colored accent bars at top and bottom of card."""
    away_color = _get_team_color(away_team, sport)[0]
    home_color = _get_team_color(home_team, sport)[0]
    bar_h = 8
    half = W // 2
    # Top bar
    draw.rectangle([0, 0, half, bar_h], fill=away_color)
    draw.rectangle([half, 0, W, bar_h], fill=home_color)
    # Bottom bar
    draw.rectangle([0, H - bar_h, half, H], fill=away_color)
    draw.rectangle([half, H - bar_h, W, H], fill=home_color)


def _draw_rounded_rect(draw, xy, fill, outline=None, radius=16, width=1):
    """Draw a rounded rectangle. xy = [x0, y0, x1, y1]."""
    x0, y0, x1, y1 = xy
    draw.rectangle([x0 + radius, y0, x1 - radius, y1], fill=fill)
    draw.rectangle([x0, y0 + radius, x1, y1 - radius], fill=fill)
    draw.pieslice([x0, y0, x0 + 2 * radius, y0 + 2 * radius], 180, 270, fill=fill)
    draw.pieslice([x1 - 2 * radius, y0, x1, y0 + 2 * radius], 270, 360, fill=fill)
    draw.pieslice([x0, y1 - 2 * radius, x0 + 2 * radius, y1], 90, 180, fill=fill)
    draw.pieslice([x1 - 2 * radius, y1 - 2 * radius, x1, y1], 0, 90, fill=fill)
    if outline:
        draw.arc([x0, y0, x0 + 2 * radius, y0 + 2 * radius], 180, 270, fill=outline, width=width)
        draw.arc([x1 - 2 * radius, y0, x1, y0 + 2 * radius], 270, 360, fill=outline, width=width)
        draw.arc([x0, y1 - 2 * radius, x0 + 2 * radius, y1], 90, 180, fill=outline, width=width)
        draw.arc([x1 - 2 * radius, y1 - 2 * radius, x1, y1], 0, 90, fill=outline, width=width)
        draw.line([x0 + radius, y0, x1 - radius, y0], fill=outline, width=width)
        draw.line([x0 + radius, y1, x1 - radius, y1], fill=outline, width=width)
        draw.line([x0, y0 + radius, x0, y1 - radius], fill=outline, width=width)
        draw.line([x1, y0 + radius, x1, y1 - radius], fill=outline, width=width)


def _get_fonts():
    """Load fonts with fallbacks."""
    from PIL import ImageFont

    # Bundled font checked first — guarantees availability on Railway/Docker
    # File lives at backend/assets/fonts/, this file is at backend/src/services/social/
    # so we need 4 .parent calls to get from src/services/social/ -> backend/
    here = Path(__file__).resolve()
    bundled_candidates = [
        here.parent.parent.parent.parent / "assets" / "fonts" / "DejaVuSans-Bold.ttf",  # backend/assets/fonts/
        here.parent.parent.parent / "assets" / "fonts" / "DejaVuSans-Bold.ttf",          # src/assets/fonts/ (legacy)
    ]
    font_paths = [str(p) for p in bundled_candidates] + [
        "/System/Library/Fonts/Helvetica.ttc",            # macOS
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux Debian
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",    # Linux RHEL
    ]

    font_path = None
    for p in font_paths:
        if Path(p).exists():
            font_path = p
            logger.info(f"Loaded font from: {p}")
            break

    if not font_path:
        logger.warning("No TTF font found, using PIL default")
        default = ImageFont.load_default()
        return {size: default for size in ["hero", "huge", "xl", "l", "m", "s", "xs", "footer"]}

    try:
        return {
            "hero": ImageFont.truetype(font_path, 200),
            "huge": ImageFont.truetype(font_path, 160),
            "xl": ImageFont.truetype(font_path, 108),
            "l": ImageFont.truetype(font_path, 72),
            "m": ImageFont.truetype(font_path, 56),
            "s": ImageFont.truetype(font_path, 42),
            "xs": ImageFont.truetype(font_path, 36),
            "footer": ImageFont.truetype(font_path, 32),
        }
    except Exception as e:
        logger.warning(f"Font load failed: {e}")
        default = ImageFont.load_default()
        return {size: default for size in ["hero", "huge", "xl", "l", "m", "s", "xs", "footer"]}


NBA_ESPN_MAP = {
    "ATL": "atl", "BOS": "bos", "BKN": "bkn", "CHA": "cha",
    "CHI": "chi", "CLE": "cle", "DAL": "dal", "DEN": "den",
    "DET": "det", "GSW": "gs", "HOU": "hou", "IND": "ind",
    "LAC": "lac", "LAL": "lal", "MEM": "mem", "MIA": "mia",
    "MIL": "mil", "MIN": "min", "NOP": "no", "NYK": "ny",
    "OKC": "okc", "ORL": "orl", "PHI": "phi", "PHX": "phx",
    "POR": "por", "SAC": "sac", "SAS": "sa", "TOR": "tor",
    "UTA": "utah", "WAS": "wsh",
}


def _fetch_logo(team_abbr: str, sport: str = "mlb"):
    """Download ESPN team logo. Returns PIL Image or None."""
    from PIL import Image

    mlb_map = {
        "ARI": "ari", "ATL": "atl", "BAL": "bal", "BOS": "bos",
        "CHC": "chc", "CWS": "chw", "CIN": "cin", "CLE": "cle",
        "COL": "col", "DET": "det", "HOU": "hou", "KC": "kc",
        "LAA": "laa", "LAD": "lad", "MIA": "mia", "MIL": "mil",
        "MIN": "min", "NYM": "nym", "NYY": "nyy", "OAK": "oak",
        "PHI": "phi", "PIT": "pit", "SD": "sd", "SF": "sf",
        "SEA": "sea", "STL": "stl", "TB": "tb", "TEX": "tex",
        "TOR": "tor", "WSH": "wsh",
    }

    if sport == "nba":
        espn_abbr = NBA_ESPN_MAP.get(team_abbr.upper())
    else:
        espn_abbr = mlb_map.get(team_abbr.upper())
    if not espn_abbr:
        return None

    url = f"https://a.espncdn.com/i/teamlogos/{sport}/500/{espn_abbr}.png"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            return Image.open(io.BytesIO(r.read())).convert("RGBA")
    except Exception as e:
        logger.warning(f"Failed to fetch logo for {team_abbr}: {e}")
        return None


def _draw_background(draw, W: int, H: int):
    """Draw dark background with subtle grid."""
    for x in range(0, W, 50):
        draw.line([(x, 0), (x, H)], fill=GRID, width=1)
    for y in range(0, H, 50):
        draw.line([(0, y), (W, y)], fill=GRID, width=1)


def _paste_logo(img, logo, x: int, y: int, max_size: int = 180):
    """Paste a logo centered at (x, y) with max size."""
    if not logo:
        return
    logo_copy = logo.copy()
    logo_copy.thumbnail((max_size, max_size))
    px = x - logo_copy.width // 2
    py = y - logo_copy.height // 2
    img.paste(logo_copy, (px, py), logo_copy)


def _nrfi_tier(pct: float) -> tuple[tuple[int, int, int], str]:
    """Returns (color, label) for NRFI %."""
    if pct >= 70:
        return GREEN, "STRONG NRFI"
    if pct >= 55:
        return ACCENT, "LEAN NRFI"
    if pct >= 40:
        return MUTED, "TOSSUP"
    return AMBER, "LEAN YRFI"


def _build_team_sub(record: str | None, div_rank: str | None) -> str:
    """Build subtitle like '5-3 | 1st NL West' from optional parts."""
    parts = [p for p in [record, div_rank] if p]
    return " | ".join(parts) if parts else ""


def generate_nrfi_card(
    away_team: str,
    home_team: str,
    away_name: str,
    home_name: str,
    nrfi_pct: float,
    away_pitcher: str | None = None,
    away_era: float | None = None,
    home_pitcher: str | None = None,
    home_era: float | None = None,
    away_handle: str | None = None,
    home_handle: str | None = None,
    game_time: str | None = None,
    away_record: str | None = None,
    home_record: str | None = None,
    away_l10: str | None = None,
    home_l10: str | None = None,
    away_div_rank: str | None = None,
    home_div_rank: str | None = None,
    away_ats: str | None = None,
    home_ats: str | None = None,
    away_ou: str | None = None,
    home_ou: str | None = None,
) -> bytes:
    """Generate an NRFI pick card image (1200x1200). Returns PNG bytes."""
    from PIL import Image, ImageDraw

    W, H = 1200, 1200
    img = Image.new("RGB", (W, H), BG)
    _draw_gradient_bg_fast(img, W, H)

    # Drop shadows beneath card surfaces — drawn on RGBA then flattened
    img = img.convert("RGBA")
    _draw_shadow(img, [350, 430, 850, 650], radius=20, shadow_offset=10, shadow_blur=15)
    _draw_shadow(img, [50, 680, 1150, 850], radius=16, shadow_offset=8, shadow_blur=12)
    _draw_shadow(img, [50, 890, 1150, 1110], radius=16, shadow_offset=8, shadow_blur=12)
    img = img.convert("RGB")

    draw = ImageDraw.Draw(img)
    _draw_team_bars(draw, W, H, away_team, home_team)
    fonts = _get_fonts()

    draw.text((60, 30), "NRFI PICK", font=fonts["s"], fill=ACCENT)
    if game_time:
        bbox = draw.textbbox((0, 0), game_time, font=fonts["s"])
        draw.text((W - (bbox[2] - bbox[0]) - 60, 30), game_time, font=fonts["s"], fill=MUTED)

    away_logo = _fetch_logo(away_team)
    home_logo = _fetch_logo(home_team)
    _paste_logo(img, away_logo, 200, 190, max_size=200)
    _paste_logo(img, home_logo, 1000, 190, max_size=200)

    draw = ImageDraw.Draw(img)
    # Auto-fit team names within ~360px (avoids clipping on long names like ATHLETICS, D-BACKS)
    away_font = _fit_font(away_name.upper(), max_width=360)
    home_font = _fit_font(home_name.upper(), max_width=360)
    draw.text((200, 330), away_name.upper(), font=away_font, fill=WHITE, anchor="mm")
    draw.text((1000, 330), home_name.upper(), font=home_font, fill=WHITE, anchor="mm")

    away_sub = _build_team_sub(away_record, away_div_rank)
    home_sub = _build_team_sub(home_record, home_div_rank)
    if away_sub:
        away_sub_font = _fit_font(away_sub, max_width=380, size_candidates=(36, 32, 28, 24))
        draw.text((200, 390), away_sub, font=away_sub_font, fill=MUTED, anchor="mm")
    if home_sub:
        home_sub_font = _fit_font(home_sub, max_width=380, size_candidates=(36, 32, 28, 24))
        draw.text((1000, 390), home_sub, font=home_sub_font, fill=MUTED, anchor="mm")

    # NRFI % badge
    tier_color, tier_label = _nrfi_tier(nrfi_pct)
    _draw_rounded_rect(draw, [350, 430, 850, 650], fill=CARD, outline=SURFACE, radius=20, width=2)
    draw.text((W // 2, 510), f"{nrfi_pct:.0f}%", font=fonts["huge"], fill=tier_color, anchor="mm")
    draw.text((W // 2, 605), tier_label, font=fonts["m"], fill=tier_color, anchor="mm")

    # Pitchers card
    _draw_rounded_rect(draw, [50, 680, W - 50, 850], fill=CARD, outline=SURFACE, radius=16, width=1)
    draw.text((W // 2, 705), "STARTING PITCHERS", font=fonts["xs"], fill=DIM, anchor="mm")

    if away_pitcher:
        draw.text((250, 755), away_pitcher, font=fonts["m"], fill=WHITE, anchor="mm")
        if away_era is not None:
            draw.text((250, 810), f"{away_era:.2f} ERA", font=fonts["s"], fill=MUTED, anchor="mm")

    draw.text((W // 2, 780), "vs", font=fonts["s"], fill=DIM, anchor="mm")

    if home_pitcher:
        draw.text((950, 755), home_pitcher, font=fonts["m"], fill=WHITE, anchor="mm")
        if home_era is not None:
            draw.text((950, 810), f"{home_era:.2f} ERA", font=fonts["s"], fill=MUTED, anchor="mm")

    # Stats table (3 rows, [away] LABEL [home])
    _draw_stats_table(draw, 50, 890, W - 50, 1110,
                      away_l10, home_l10, away_ats, home_ats, away_ou, home_ou, fonts)

    draw.text((60, H - 50), "truline.app", font=fonts["footer"], fill=ACCENT)
    draw.text((W - 60, H - 50), "@trulineapp", font=fonts["footer"], fill=ACCENT, anchor="ra")

    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return buf.getvalue()


def generate_recap_card(
    away_team: str,
    home_team: str,
    away_name: str,
    home_name: str,
    away_first: int,
    home_first: int,
    is_nrfi: bool,
    predicted_nrfi_pct: float | None = None,
    away_record: str | None = None,
    home_record: str | None = None,
    away_div_rank: str | None = None,
    home_div_rank: str | None = None,
) -> bytes:
    """Generate a 1st inning recap card image (1200x1200). Returns PNG bytes."""
    from PIL import Image, ImageDraw

    W, H = 1200, 1200
    img = Image.new("RGB", (W, H), BG)
    _draw_gradient_bg_fast(img, W, H)

    # Shadow beneath the NRFI/YRFI result badge
    img = img.convert("RGBA")
    _draw_shadow(img, [300, 690, 900, 900], radius=20, shadow_offset=10, shadow_blur=15)
    img = img.convert("RGB")

    draw = ImageDraw.Draw(img)
    _draw_team_bars(draw, W, H, away_team, home_team)
    fonts = _get_fonts()

    draw.text((60, 30), "1ST INNING RECAP", font=fonts["s"], fill=ACCENT)

    away_logo = _fetch_logo(away_team)
    home_logo = _fetch_logo(home_team)
    _paste_logo(img, away_logo, 200, 190, max_size=200)
    _paste_logo(img, home_logo, 1000, 190, max_size=200)

    draw = ImageDraw.Draw(img)

    away_font = _fit_font(away_name.upper(), max_width=360)
    home_font = _fit_font(home_name.upper(), max_width=360)
    draw.text((200, 330), away_name.upper(), font=away_font, fill=WHITE, anchor="mm")
    draw.text((1000, 330), home_name.upper(), font=home_font, fill=WHITE, anchor="mm")

    away_sub = _build_team_sub(away_record, away_div_rank)
    home_sub = _build_team_sub(home_record, home_div_rank)
    if away_sub:
        away_sub_font = _fit_font(away_sub, max_width=380, size_candidates=(36, 32, 28, 24))
        draw.text((200, 390), away_sub, font=away_sub_font, fill=MUTED, anchor="mm")
    if home_sub:
        home_sub_font = _fit_font(home_sub, max_width=380, size_candidates=(36, 32, 28, 24))
        draw.text((1000, 390), home_sub, font=home_sub_font, fill=MUTED, anchor="mm")

    score_text = f"{away_first}  —  {home_first}"
    draw.text((W // 2, 510), score_text, font=fonts["hero"], fill=WHITE, anchor="mm")
    draw.text((W // 2, 620), "1ST INNING", font=fonts["s"], fill=MUTED, anchor="mm")

    # Result badge
    _draw_rounded_rect(draw, [300, 690, 900, 900], fill=CARD, outline=SURFACE, radius=20, width=2)
    if is_nrfi:
        draw.text((W // 2, 760), "NRFI", font=fonts["xl"], fill=GREEN, anchor="mm")
        draw.text((W // 2, 855), "MODEL CALLED IT", font=fonts["s"], fill=GREEN, anchor="mm")
    else:
        draw.text((W // 2, 760), "YRFI", font=fonts["xl"], fill=AMBER, anchor="mm")
        draw.text((W // 2, 855), "RUNS IN THE 1ST", font=fonts["s"], fill=AMBER, anchor="mm")

    if predicted_nrfi_pct is not None:
        draw.text(
            (W // 2, 960),
            f"Model predicted {predicted_nrfi_pct:.0f}% NRFI",
            font=fonts["s"],
            fill=MUTED,
            anchor="mm",
        )

    draw.text((60, H - 50), "truline.app", font=fonts["footer"], fill=ACCENT)
    draw.text((W - 60, H - 50), "@trulineapp", font=fonts["footer"], fill=ACCENT, anchor="ra")

    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return buf.getvalue()


def _value_tier(score: float) -> tuple[tuple[int, int, int], str]:
    if score >= 75:
        return GREEN, "STRONG VALUE"
    if score >= 65:
        return ACCENT, "GOOD VALUE"
    if score >= 55:
        return MUTED, "LEAN"
    return AMBER, "LOW VALUE"


def generate_nba_card(
    away_team: str,
    home_team: str,
    away_name: str,
    home_name: str,
    pick_team: str,
    pick_label: str,
    value_score: float,
    edge_pct: float,
    model_prob: float,
    market_prob: float,
    odds_american: int,
    away_handle: str | None = None,
    home_handle: str | None = None,
    game_time: str | None = None,
) -> bytes:
    """Generate an NBA pick card image (1200x900). Returns PNG bytes."""
    from PIL import Image, ImageDraw

    W, H = 1200, 900
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    fonts = _get_fonts()

    _draw_background(draw, W, H)

    draw.text((60, 35), "NBA PICK", font=fonts["s"], fill=ACCENT)
    if game_time:
        bbox = draw.textbbox((0, 0), game_time, font=fonts["s"])
        draw.text((W - (bbox[2] - bbox[0]) - 60, 35), game_time, font=fonts["s"], fill=MUTED)

    away_logo = _fetch_logo(away_team, sport="nba")
    home_logo = _fetch_logo(home_team, sport="nba")
    _paste_logo(img, away_logo, 200, 210, max_size=180)
    _paste_logo(img, home_logo, 1000, 210, max_size=180)

    draw.text((200, 360), away_name.upper(), font=fonts["l"], fill=WHITE, anchor="mm")
    draw.text((1000, 360), home_name.upper(), font=fonts["l"], fill=WHITE, anchor="mm")

    if away_handle:
        draw.text((200, 415), away_handle, font=fonts["xs"], fill=DIM, anchor="mm")
    if home_handle:
        draw.text((1000, 415), home_handle, font=fonts["xs"], fill=DIM, anchor="mm")

    tier_color, tier_label = _value_tier(value_score)

    # Center big pick label
    draw.text((W // 2, 170), pick_label, font=fonts["xl"], fill=WHITE, anchor="mm")
    odds_str = f"+{odds_american}" if odds_american > 0 else f"{odds_american}"
    draw.text((W // 2, 250), odds_str, font=fonts["m"], fill=MUTED, anchor="mm")

    # Value score badge
    draw.text((W // 2, 350), f"{value_score:.0f}/100", font=fonts["huge"], fill=tier_color, anchor="mm")
    draw.text((W // 2, 450), tier_label, font=fonts["s"], fill=tier_color, anchor="mm")

    # Stats bar
    bar_y = 550
    bar_h = 240
    draw.rectangle([40, bar_y, W - 40, bar_y + bar_h], fill=CARD, outline=SURFACE, width=2)
    draw.text((W // 2, bar_y + 30), "MODEL vs MARKET", font=fonts["xs"], fill=DIM, anchor="mm")

    model_txt = f"{model_prob * 100:.0f}%"
    market_txt = f"{market_prob * 100:.0f}%"
    edge_txt = f"+{edge_pct:.1f}%"

    draw.text((260, bar_y + 95), "MODEL", font=fonts["xs"], fill=DIM, anchor="mm")
    draw.text((260, bar_y + 160), model_txt, font=fonts["l"], fill=GREEN, anchor="mm")

    draw.text((W // 2, bar_y + 95), "MARKET", font=fonts["xs"], fill=DIM, anchor="mm")
    draw.text((W // 2, bar_y + 160), market_txt, font=fonts["l"], fill=WHITE, anchor="mm")

    draw.text((W - 260, bar_y + 95), "EDGE", font=fonts["xs"], fill=DIM, anchor="mm")
    draw.text((W - 260, bar_y + 160), edge_txt, font=fonts["l"], fill=tier_color, anchor="mm")

    draw.text((60, H - 45), "truline.app", font=fonts["xs"], fill=ACCENT)
    draw.text((W - 60, H - 45), "@trulineapp", font=fonts["xs"], fill=ACCENT, anchor="ra")

    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return buf.getvalue()


def generate_final_card(
    away_team: str,
    home_team: str,
    away_name: str,
    home_name: str,
    away_score: int,
    home_score: int,
    away_first: int | None = None,
    home_first: int | None = None,
    pick_team: str | None = None,
    pick_type: str | None = None,
    pick_line: float | None = None,
    pick_result: str | None = None,
    away_record: str | None = None,
    home_record: str | None = None,
    away_l10: str | None = None,
    home_l10: str | None = None,
    away_div_rank: str | None = None,
    home_div_rank: str | None = None,
    away_ats: str | None = None,
    home_ats: str | None = None,
    away_ou: str | None = None,
    home_ou: str | None = None,
) -> bytes:
    """Generate a final score recap card image (1200x1200). Returns PNG bytes."""
    from PIL import Image, ImageDraw

    W, H = 1200, 1200
    img = Image.new("RGB", (W, H), BG)
    _draw_gradient_bg_fast(img, W, H)

    # Shadow beneath stats table
    img = img.convert("RGBA")
    _draw_shadow(img, [50, 870, 1150, 1090], radius=16, shadow_offset=8, shadow_blur=12)
    # Shadow for pick result if we'll draw one
    if pick_team and pick_result:
        _draw_shadow(img, [200, 670, 1000, 745], radius=12, shadow_offset=6, shadow_blur=10)
    img = img.convert("RGB")

    draw = ImageDraw.Draw(img)
    _draw_team_bars(draw, W, H, away_team, home_team)
    fonts = _get_fonts()

    draw.text((60, 30), "FINAL", font=fonts["s"], fill=ACCENT)

    away_logo = _fetch_logo(away_team)
    home_logo = _fetch_logo(home_team)
    _paste_logo(img, away_logo, 200, 190, max_size=200)
    _paste_logo(img, home_logo, 1000, 190, max_size=200)

    draw = ImageDraw.Draw(img)

    away_font = _fit_font(away_name.upper(), max_width=360)
    home_font = _fit_font(home_name.upper(), max_width=360)
    draw.text((200, 330), away_name.upper(), font=away_font, fill=WHITE, anchor="mm")
    draw.text((1000, 330), home_name.upper(), font=home_font, fill=WHITE, anchor="mm")

    if away_record:
        draw.text((200, 385), away_record, font=fonts["xs"], fill=MUTED, anchor="mm")
    if home_record:
        draw.text((1000, 385), home_record, font=fonts["xs"], fill=MUTED, anchor="mm")

    score_text = f"{away_score}  —  {home_score}"
    draw.text((W // 2, 490), score_text, font=fonts["hero"], fill=WHITE, anchor="mm")

    y_mid = 610
    if away_first is not None and home_first is not None:
        is_nrfi = (away_first + home_first) == 0
        tag = "NRFI" if is_nrfi else "YRFI"
        tag_color = GREEN if is_nrfi else AMBER
        fi_text = f"1st Inning: {away_first}-{home_first} ({tag})"
        draw.text((W // 2, y_mid), fi_text, font=fonts["m"], fill=tag_color, anchor="mm")
        y_mid = 680

    if pick_team and pick_result:
        result_color = GREEN if pick_result == "win" else (RED if pick_result == "loss" else MUTED)
        result_label = "W" if pick_result == "win" else ("L" if pick_result == "loss" else "P")
        pick_label = pick_type.upper() if pick_type else ""
        line_str = f" {pick_line:+g}" if pick_line is not None else ""
        pick_text = f"Our Pick: {pick_team} {pick_label}{line_str} — {result_label}"
        _draw_rounded_rect(draw, [200, 670, 1000, 745], fill=CARD, outline=SURFACE, radius=12)
        draw.text((W // 2, 707), pick_text, font=fonts["s"], fill=result_color, anchor="mm")
        y_mid = 795

    if away_score > home_score:
        winner_name = away_name
    elif home_score > away_score:
        winner_name = home_name
    else:
        winner_name = None
    if winner_name:
        winner_font = _fit_font(f"{winner_name.upper()} WINS", max_width=1000)
        draw.text((W // 2, y_mid), f"{winner_name.upper()} WINS", font=winner_font, fill=WHITE, anchor="mm")

    # Stats table at bottom
    _draw_stats_table(draw, 50, 870, W - 50, 1090,
                      away_l10, home_l10, away_ats, home_ats, away_ou, home_ou, fonts)

    draw.text((60, H - 50), "truline.app", font=fonts["footer"], fill=ACCENT)
    draw.text((W - 60, H - 50), "@trulineapp", font=fonts["footer"], fill=ACCENT, anchor="ra")

    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return buf.getvalue()
