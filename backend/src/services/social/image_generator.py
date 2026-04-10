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

    # Result tag — use plain ASCII instead of Unicode checkmarks
    # (DejaVu/Helvetica don't reliably have the ✓/✗ glyphs on Linux containers)
    if is_nrfi:
        draw.text((W // 2, 420), "NRFI", font=fonts["xl"], fill=GREEN, anchor="mm")
        draw.text((W // 2, 480), "MODEL CALLED IT", font=fonts["s"], fill=GREEN, anchor="mm")
    else:
        draw.text((W // 2, 420), "YRFI", font=fonts["xl"], fill=AMBER, anchor="mm")
        draw.text((W // 2, 480), "RUNS IN THE 1ST", font=fonts["s"], fill=AMBER, anchor="mm")

    if predicted_nrfi_pct is not None:
        draw.text(
            (W // 2, 540),
            f"Model: {predicted_nrfi_pct:.0f}% NRFI chance",
            font=fonts["s"],
            fill=MUTED,
            anchor="mm",
        )

    # Footer
    draw.text((60, H - 40), "truline.app", font=fonts["xs"], fill=ACCENT)
    draw.text((W - 60, H - 40), "@trulineapp", font=fonts["xs"], fill=ACCENT, anchor="ra")

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
    """Generate an NBA pick card image. Returns PNG bytes."""
    from PIL import Image, ImageDraw

    W, H = 1200, 675
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    fonts = _get_fonts()

    _draw_background(draw, W, H)

    draw.text((60, 40), "NBA PICK", font=fonts["s"], fill=ACCENT)
    if game_time:
        bbox = draw.textbbox((0, 0), game_time, font=fonts["s"])
        draw.text((W - (bbox[2] - bbox[0]) - 60, 40), game_time, font=fonts["s"], fill=MUTED)

    away_logo = _fetch_logo(away_team, sport="nba")
    home_logo = _fetch_logo(home_team, sport="nba")
    _paste_logo(img, away_logo, 200, 170, max_size=150)
    _paste_logo(img, home_logo, 1000, 170, max_size=150)

    draw.text((200, 310), away_name.upper(), font=fonts["l"], fill=WHITE, anchor="mm")
    draw.text((1000, 310), home_name.upper(), font=fonts["l"], fill=WHITE, anchor="mm")

    if away_handle:
        draw.text((200, 360), away_handle, font=fonts["xs"], fill=DIM, anchor="mm")
    if home_handle:
        draw.text((1000, 360), home_handle, font=fonts["xs"], fill=DIM, anchor="mm")

    tier_color, tier_label = _value_tier(value_score)

    # Center big pick label
    draw.text((W // 2, 150), pick_label, font=fonts["xl"], fill=WHITE, anchor="mm")
    odds_str = f"+{odds_american}" if odds_american > 0 else f"{odds_american}"
    draw.text((W // 2, 210), odds_str, font=fonts["m"], fill=MUTED, anchor="mm")

    # Value score badge
    draw.text((W // 2, 290), f"{value_score:.0f}/100", font=fonts["huge"], fill=tier_color, anchor="mm")
    draw.text((W // 2, 365), tier_label, font=fonts["s"], fill=tier_color, anchor="mm")

    # Stats bar
    bar_y = 440
    bar_h = 170
    draw.rectangle([40, bar_y, W - 40, bar_y + bar_h], fill=CARD, outline=SURFACE, width=2)
    draw.text((W // 2, bar_y + 25), "MODEL vs MARKET", font=fonts["xs"], fill=DIM, anchor="mm")

    model_txt = f"{model_prob * 100:.0f}%"
    market_txt = f"{market_prob * 100:.0f}%"
    edge_txt = f"+{edge_pct:.1f}%"

    draw.text((260, bar_y + 75), "MODEL", font=fonts["xs"], fill=DIM, anchor="mm")
    draw.text((260, bar_y + 115), model_txt, font=fonts["l"], fill=GREEN, anchor="mm")

    draw.text((W // 2, bar_y + 75), "MARKET", font=fonts["xs"], fill=DIM, anchor="mm")
    draw.text((W // 2, bar_y + 115), market_txt, font=fonts["l"], fill=WHITE, anchor="mm")

    draw.text((W - 260, bar_y + 75), "EDGE", font=fonts["xs"], fill=DIM, anchor="mm")
    draw.text((W - 260, bar_y + 115), edge_txt, font=fonts["l"], fill=tier_color, anchor="mm")

    draw.text((60, H - 40), "truline.app", font=fonts["xs"], fill=ACCENT)
    draw.text((W - 60, H - 40), "@trulineapp", font=fonts["xs"], fill=ACCENT, anchor="ra")

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
) -> bytes:
    """Generate a final score recap card image. Returns PNG bytes."""
    from PIL import Image, ImageDraw

    W, H = 1200, 675
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    fonts = _get_fonts()

    _draw_background(draw, W, H)

    # Top label
    draw.text((60, 40), "FINAL SCORE", font=fonts["s"], fill=ACCENT)

    # Logos
    away_logo = _fetch_logo(away_team)
    home_logo = _fetch_logo(home_team)
    _paste_logo(img, away_logo, 200, 170, max_size=150)
    _paste_logo(img, home_logo, 1000, 170, max_size=150)

    # Team names
    draw.text((200, 310), away_name.upper(), font=fonts["l"], fill=WHITE, anchor="mm")
    draw.text((1000, 310), home_name.upper(), font=fonts["l"], fill=WHITE, anchor="mm")

    # Center: final score
    score_text = f"{away_score} - {home_score}"
    draw.text((W // 2, 170), score_text, font=fonts["huge"], fill=WHITE, anchor="mm")
    draw.text((W // 2, 250), "FINAL", font=fonts["s"], fill=MUTED, anchor="mm")

    # 1st inning result
    if away_first is not None and home_first is not None:
        is_nrfi = (away_first + home_first) == 0
        tag = "NRFI" if is_nrfi else "YRFI"
        tag_color = GREEN if is_nrfi else AMBER
        fi_text = f"1st Inning: {away_first}-{home_first} ({tag})"
        draw.text((W // 2, 380), fi_text, font=fonts["m"], fill=tag_color, anchor="mm")

    # Pick result (if we had a bet on this game)
    if pick_team and pick_result:
        result_color = GREEN if pick_result == "win" else (RED if pick_result == "loss" else MUTED)
        result_label = "W" if pick_result == "win" else ("L" if pick_result == "loss" else "P")
        pick_label = pick_type.upper() if pick_type else ""
        line_str = f" {pick_line:+g}" if pick_line is not None else ""
        pick_text = f"Pick: {pick_team} {pick_label}{line_str} -- {result_label}"
        draw.text((W // 2, 460), pick_text, font=fonts["m"], fill=result_color, anchor="mm")
    elif pick_team:
        pick_label = pick_type.upper() if pick_type else ""
        line_str = f" {pick_line:+g}" if pick_line is not None else ""
        pick_text = f"Pick: {pick_team} {pick_label}{line_str}"
        draw.text((W // 2, 460), pick_text, font=fonts["m"], fill=MUTED, anchor="mm")

    # Winner highlight
    if away_score > home_score:
        winner_name = away_name
    elif home_score > away_score:
        winner_name = home_name
    else:
        winner_name = None
    if winner_name:
        draw.text((W // 2, 540), f"{winner_name.upper()} WINS", font=fonts["l"], fill=WHITE, anchor="mm")

    # Footer
    draw.text((60, H - 40), "truline.app", font=fonts["xs"], fill=ACCENT)
    draw.text((W - 60, H - 40), "@trulineapp", font=fonts["xs"], fill=ACCENT, anchor="ra")

    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return buf.getvalue()
