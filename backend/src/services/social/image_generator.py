"""Generate social media card images for TruLine tweets."""

import io
import urllib.request
from pathlib import Path
from typing import Literal

import structlog

logger = structlog.get_logger()


# Brand colors
BG = (10, 14, 23)
SURFACE = (21, 29, 46)
CARD = (11, 14, 20)
ACCENT = (164, 230, 255)
GREEN = (102, 247, 150)
AMBER = (245, 158, 11)
RED = (239, 68, 68)
WHITE = (241, 245, 249)
MUTED = (148, 163, 184)
DIM = (100, 116, 139)
GRID = (15, 25, 35)


def _get_fonts():
    """Load fonts with fallbacks."""
    from PIL import ImageFont

    font_paths = [
        "/System/Library/Fonts/Helvetica.ttc",           # macOS
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux Debian
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",   # Linux RHEL
    ]

    font_path = None
    for p in font_paths:
        if Path(p).exists():
            font_path = p
            break

    if not font_path:
        logger.warning("No TTF font found, using PIL default")
        default = ImageFont.load_default()
        return {size: default for size in ["huge", "xl", "l", "m", "s", "xs"]}

    try:
        return {
            "huge": ImageFont.truetype(font_path, 90),
            "xl": ImageFont.truetype(font_path, 64),
            "l": ImageFont.truetype(font_path, 48),
            "m": ImageFont.truetype(font_path, 34),
            "s": ImageFont.truetype(font_path, 26),
            "xs": ImageFont.truetype(font_path, 20),
        }
    except Exception as e:
        logger.warning(f"Font load failed: {e}")
        default = ImageFont.load_default()
        return {size: default for size in ["huge", "xl", "l", "m", "s", "xs"]}


def _fetch_logo(team_abbr: str):
    """Download ESPN MLB team logo. Returns PIL Image or None."""
    from PIL import Image

    espn_map = {
        "ARI": "ari", "ATL": "atl", "BAL": "bal", "BOS": "bos",
        "CHC": "chc", "CWS": "chw", "CIN": "cin", "CLE": "cle",
        "COL": "col", "DET": "det", "HOU": "hou", "KC": "kc",
        "LAA": "laa", "LAD": "lad", "MIA": "mia", "MIL": "mil",
        "MIN": "min", "NYM": "nym", "NYY": "nyy", "OAK": "oak",
        "PHI": "phi", "PIT": "pit", "SD": "sd", "SF": "sf",
        "SEA": "sea", "STL": "stl", "TB": "tb", "TEX": "tex",
        "TOR": "tor", "WSH": "wsh",
    }

    espn_abbr = espn_map.get(team_abbr.upper())
    if not espn_abbr:
        return None

    url = f"https://a.espncdn.com/i/teamlogos/mlb/500/{espn_abbr}.png"
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
) -> bytes:
    """
    Generate an NRFI pick card image.

    Returns PNG bytes ready to upload.
    """
    from PIL import Image, ImageDraw

    W, H = 1200, 675
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    fonts = _get_fonts()

    _draw_background(draw, W, H)

    # Top label
    draw.text((60, 40), "NRFI PICK", font=fonts["s"], fill=ACCENT)
    if game_time:
        bbox = draw.textbbox((0, 0), game_time, font=fonts["s"])
        draw.text((W - (bbox[2] - bbox[0]) - 60, 40), game_time, font=fonts["s"], fill=MUTED)

    # Team logos - smaller and positioned higher for clear spacing from text
    away_logo = _fetch_logo(away_team)
    home_logo = _fetch_logo(home_team)

    _paste_logo(img, away_logo, 200, 170, max_size=150)
    _paste_logo(img, home_logo, 1000, 170, max_size=150)

    # Team names well below logos
    draw.text((200, 310), away_name.upper(), font=fonts["l"], fill=WHITE, anchor="mm")
    draw.text((1000, 310), home_name.upper(), font=fonts["l"], fill=WHITE, anchor="mm")

    # Team handles
    if away_handle:
        draw.text((200, 360), away_handle, font=fonts["xs"], fill=DIM, anchor="mm")
    if home_handle:
        draw.text((1000, 360), home_handle, font=fonts["xs"], fill=DIM, anchor="mm")

    # Center NRFI badge
    tier_color, tier_label = _nrfi_tier(nrfi_pct)

    # Big NRFI %
    pct_text = f"{nrfi_pct:.0f}%"
    draw.text((W // 2, 170), pct_text, font=fonts["huge"], fill=tier_color, anchor="mm")
    draw.text((W // 2, 250), "NRFI CHANCE", font=fonts["s"], fill=MUTED, anchor="mm")
    draw.text((W // 2, 305), tier_label, font=fonts["m"], fill=tier_color, anchor="mm")

    # Pitcher matchup bar at bottom
    bar_y = 440
    bar_h = 170
    draw.rectangle([40, bar_y, W - 40, bar_y + bar_h], fill=CARD, outline=SURFACE, width=2)

    draw.text((W // 2, bar_y + 25), "STARTING PITCHERS", font=fonts["xs"], fill=DIM, anchor="mm")

    # Away pitcher (left)
    if away_pitcher:
        draw.text((200, bar_y + 70), away_pitcher, font=fonts["m"], fill=WHITE, anchor="mm")
        if away_era is not None:
            era_text = f"{away_era:.2f} ERA"
            draw.text((200, bar_y + 110), era_text, font=fonts["s"], fill=MUTED, anchor="mm")

    # "vs" in middle
    draw.text((W // 2, bar_y + 90), "vs", font=fonts["s"], fill=DIM, anchor="mm")

    # Home pitcher (right)
    if home_pitcher:
        draw.text((1000, bar_y + 70), home_pitcher, font=fonts["m"], fill=WHITE, anchor="mm")
        if home_era is not None:
            era_text = f"{home_era:.2f} ERA"
            draw.text((1000, bar_y + 110), era_text, font=fonts["s"], fill=MUTED, anchor="mm")

    # Footer
    draw.text((60, H - 40), "truline.app", font=fonts["xs"], fill=ACCENT)
    # Right-aligned handle
    draw.text((W - 60, H - 40), "@trulineapp", font=fonts["xs"], fill=ACCENT, anchor="ra")

    # Convert to bytes
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
) -> bytes:
    """
    Generate a 1st inning recap card image.

    Returns PNG bytes.
    """
    from PIL import Image, ImageDraw

    W, H = 1200, 675
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    fonts = _get_fonts()

    _draw_background(draw, W, H)

    # Top label
    draw.text((60, 40), "1ST INNING RECAP", font=fonts["s"], fill=ACCENT)

    # Logos — higher, smaller
    away_logo = _fetch_logo(away_team)
    home_logo = _fetch_logo(home_team)
    _paste_logo(img, away_logo, 200, 170, max_size=150)
    _paste_logo(img, home_logo, 1000, 170, max_size=150)

    # Team names — well below logos
    draw.text((200, 310), away_name.upper(), font=fonts["l"], fill=WHITE, anchor="mm")
    draw.text((1000, 310), home_name.upper(), font=fonts["l"], fill=WHITE, anchor="mm")

    # Center: 1st inning score
    score_text = f"{away_first} - {home_first}"
    draw.text((W // 2, 170), score_text, font=fonts["huge"], fill=WHITE, anchor="mm")
    draw.text((W // 2, 250), "1ST INNING", font=fonts["s"], fill=MUTED, anchor="mm")

    # Result tag
    if is_nrfi:
        draw.text((W // 2, 400), "NRFI ✓", font=fonts["xl"], fill=GREEN, anchor="mm")
        if predicted_nrfi_pct is not None:
            draw.text((W // 2, 470), f"Model: {predicted_nrfi_pct:.0f}% NRFI chance", font=fonts["s"], fill=MUTED, anchor="mm")
    else:
        draw.text((W // 2, 400), "YRFI ✗", font=fonts["xl"], fill=AMBER, anchor="mm")
        if predicted_nrfi_pct is not None:
            draw.text((W // 2, 470), f"Model: {predicted_nrfi_pct:.0f}% NRFI chance", font=fonts["s"], fill=MUTED, anchor="mm")

    # Footer
    draw.text((60, H - 40), "truline.app", font=fonts["xs"], fill=ACCENT)
    draw.text((W - 60, H - 40), "@trulineapp", font=fonts["xs"], fill=ACCENT, anchor="ra")

    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return buf.getvalue()
