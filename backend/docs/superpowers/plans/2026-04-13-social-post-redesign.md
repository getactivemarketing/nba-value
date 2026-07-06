# Social Post Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix team stats display, redesign card images for readability on social feeds, and rewrite tweet copy with data-backed narrative voice.

**Architecture:** PIL-based card generation with team color accents, larger fonts, and 1200x1200 square layout. Tweet copy enhanced with new helper functions that pull pitcher 1st-inning records, team streaks, and bet type summaries from existing DB tables. No new dependencies.

**Tech Stack:** Python, Pillow (PIL), SQLAlchemy async, structlog

**Spec:** `docs/superpowers/specs/2026-04-13-social-post-redesign.md`

---

### Task 1: Fix `get_team_card_stats()` and add logging

**Files:**
- Modify: `src/services/social/content.py:112-171`

- [ ] **Step 1: Add debug logging and L10 fallback to `get_team_card_stats()`**

Replace the existing `get_team_card_stats` function (lines 112-171) with this version that adds logging and builds `last_10_record` from integer columns when the string is None:

```python
async def get_team_card_stats(session: AsyncSession, team_abbr: str) -> dict:
    """Fetch team stats for social media card images.

    Returns dict with: record, l10, div_rank, ats, ou (all as formatted strings or None).
    """
    from src.models.mlb_team import MLBTeam

    result = {}

    # Get latest team stats
    stat_row = await session.execute(
        select(MLBTeamStats).where(
            MLBTeamStats.team_abbr == team_abbr
        ).order_by(desc(MLBTeamStats.stat_date)).limit(1)
    )
    stats = stat_row.scalar_one_or_none()

    if not stats:
        logger.warning("get_team_card_stats: no MLBTeamStats row", team=team_abbr)
        return result

    if stats.wins is not None and stats.losses is not None:
        result["record"] = f"{stats.wins}-{stats.losses}"
    # L10: prefer the string column, fall back to building from ints
    if stats.last_10_record:
        result["l10"] = stats.last_10_record
    elif stats.last_10_wins is not None and stats.last_10_losses is not None:
        result["l10"] = f"{stats.last_10_wins}-{stats.last_10_losses}"
    if stats.ats_wins is not None and stats.ats_losses is not None:
        result["ats"] = f"{stats.ats_wins}-{stats.ats_losses}"
    if stats.ou_overs is not None and stats.ou_unders is not None:
        result["ou"] = f"{stats.ou_overs}-{stats.ou_unders}"

    logger.debug("get_team_card_stats: stats found", team=team_abbr, result=result)

    # Get division + compute rank (separate from core stats so a failure here
    # doesn't prevent W-L/L10/ATS/O-U from showing)
    try:
        team_row = await session.execute(
            select(MLBTeam).where(MLBTeam.team_abbr == team_abbr)
        )
        team = team_row.scalar_one_or_none()

        if not team:
            logger.warning("get_team_card_stats: no MLBTeam row — skipping div rank", team=team_abbr)
            return result

        if stats.wins is not None:
            div_teams = await session.execute(
                select(MLBTeam.team_abbr).where(
                    and_(MLBTeam.league == team.league, MLBTeam.division == team.division)
                )
            )
            div_abbrs = [r[0] for r in div_teams.fetchall()]

            div_records = []
            for abbr in div_abbrs:
                s = await session.execute(
                    select(MLBTeamStats.team_abbr, MLBTeamStats.win_pct).where(
                        MLBTeamStats.team_abbr == abbr
                    ).order_by(desc(MLBTeamStats.stat_date)).limit(1)
                )
                row = s.first()
                if row and row[1] is not None:
                    div_records.append((row[0], float(row[1])))

            div_records.sort(key=lambda x: x[1], reverse=True)
            rank = next((i + 1 for i, (a, _) in enumerate(div_records) if a == team_abbr), None)
            if rank:
                suffix = {1: "st", 2: "nd", 3: "rd"}.get(rank, "th")
                result["div_rank"] = f"{rank}{suffix} {team.league} {team.division}"
    except Exception as e:
        logger.warning("get_team_card_stats: div rank lookup failed", team=team_abbr, error=str(e))

    return result
```

- [ ] **Step 2: Test locally**

Run from `backend/`:
```bash
python3 -c "
from src.services.social.content import get_team_card_stats
print('Function loaded OK')
"
```
Expected: prints without import errors.

- [ ] **Step 3: Commit**

```bash
git add src/services/social/content.py
git commit -m "fix: make get_team_card_stats resilient with logging and L10 fallback"
```

---

### Task 2: Add team colors and visual helper functions to image_generator.py

**Files:**
- Modify: `src/services/social/image_generator.py` (add to top of file, after existing constants)

- [ ] **Step 1: Add `TEAM_COLORS` dict after the existing color constants (after line 24)**

```python
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
```

- [ ] **Step 2: Add visual helper functions after `_get_team_color`**

```python
def _draw_gradient_bg(img, W: int, H: int):
    """Draw a radial gradient background — lighter center, darker edges."""
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    center_x, center_y = W // 2, H // 2
    max_dist = ((center_x ** 2) + (center_y ** 2)) ** 0.5
    # Precompute rows for speed — draw horizontal lines with blended color
    for y in range(H):
        dy = abs(y - center_y)
        for x in range(0, W, 4):  # step by 4px for performance
            dx = abs(x - center_x)
            dist = ((dx ** 2) + (dy ** 2)) ** 0.5
            t = min(dist / max_dist, 1.0)
            # Center: (15, 21, 37), Edges: (10, 14, 23)
            r = int(15 + (10 - 15) * t)
            g = int(21 + (14 - 21) * t)
            b = int(37 + (23 - 37) * t)
            draw.rectangle([x, y, x + 3, y], fill=(r, g, b))


def _draw_gradient_bg_fast(img, W: int, H: int):
    """Fast gradient background using horizontal bands instead of per-pixel."""
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    center_y = H // 2
    for y in range(H):
        dy = abs(y - center_y) / center_y  # 0 at center, 1 at edges
        t = dy * dy  # quadratic falloff
        r = int(18 - 8 * t)
        g = int(24 - 10 * t)
        b = int(42 - 19 * t)
        draw.line([(0, y), (W, y)], fill=(r, g, b))


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
```

- [ ] **Step 3: Update `_get_fonts()` to include new size tiers**

Replace the font sizes dict (inside `_get_fonts`, the `return` block around line 52-59):

```python
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
```

Also update the fallback dict at line 49 to match:
```python
        return {size: default for size in ["hero", "huge", "xl", "l", "m", "s", "xs", "footer"]}
```

- [ ] **Step 4: Test imports and helper functions locally**

```bash
cd /Applications/XAMPP/xamppfiles/htdocs/Sites/Truline/backend && python3 -c "
from src.services.social.image_generator import (
    _get_team_color, _draw_gradient_bg_fast, _draw_team_bars,
    _draw_rounded_rect, MLB_TEAM_COLORS, NBA_TEAM_COLORS
)
print(f'MLB colors: {len(MLB_TEAM_COLORS)} teams')
print(f'NBA colors: {len(NBA_TEAM_COLORS)} teams')
print(f'NYY: {_get_team_color(\"NYY\")}')
print(f'LAL: {_get_team_color(\"LAL\", \"nba\")}')
print('All helpers loaded OK')
"
```
Expected: 30 MLB teams, 30 NBA teams, color tuples printed.

- [ ] **Step 5: Commit**

```bash
git add src/services/social/image_generator.py
git commit -m "feat: add team colors and visual helpers for card redesign"
```

---

### Task 3: Redesign NRFI pick card (1200x1200)

**Files:**
- Modify: `src/services/social/image_generator.py` — replace `generate_nrfi_card()` (lines 146-263)

- [ ] **Step 1: Replace `generate_nrfi_card` with the redesigned version**

Replace the entire function (lines 146-263) with:

```python
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
    draw = ImageDraw.Draw(img)
    fonts = _get_fonts()

    # Background gradient + team color bars
    _draw_gradient_bg_fast(img, W, H)
    draw = ImageDraw.Draw(img)  # refresh after gradient
    _draw_team_bars(draw, W, H, away_team, home_team)

    # Top label row (y=30)
    draw.text((60, 30), "NRFI PICK", font=fonts["s"], fill=ACCENT)
    if game_time:
        bbox = draw.textbbox((0, 0), game_time, font=fonts["s"])
        draw.text((W - (bbox[2] - bbox[0]) - 60, 30), game_time, font=fonts["s"], fill=MUTED)

    # Logos (y=120-340)
    away_logo = _fetch_logo(away_team)
    home_logo = _fetch_logo(home_team)
    _paste_logo(img, away_logo, 200, 200, max_size=220)
    _paste_logo(img, home_logo, 1000, 200, max_size=220)

    # Team names (y=340)
    draw = ImageDraw.Draw(img)  # refresh after logo paste
    draw.text((200, 340), away_name.upper(), font=fonts["l"], fill=WHITE, anchor="mm")
    draw.text((1000, 340), home_name.upper(), font=fonts["l"], fill=WHITE, anchor="mm")

    # Record + division rank (y=395)
    away_sub = _build_team_sub(away_record, away_div_rank)
    home_sub = _build_team_sub(home_record, home_div_rank)
    if away_sub:
        draw.text((200, 400), away_sub, font=fonts["xs"], fill=MUTED, anchor="mm")
    if home_sub:
        draw.text((1000, 400), home_sub, font=fonts["xs"], fill=MUTED, anchor="mm")

    # Center: NRFI percentage in a rounded card (y=450-650)
    tier_color, tier_label = _nrfi_tier(nrfi_pct)
    _draw_rounded_rect(draw, [350, 450, 850, 670], fill=CARD, outline=SURFACE, radius=20, width=2)
    pct_text = f"{nrfi_pct:.0f}%"
    draw.text((W // 2, 530), pct_text, font=fonts["huge"], fill=tier_color, anchor="mm")
    draw.text((W // 2, 625), tier_label, font=fonts["m"], fill=tier_color, anchor="mm")

    # Starting pitchers bar (y=700-850)
    _draw_rounded_rect(draw, [50, 700, W - 50, 870], fill=CARD, outline=SURFACE, radius=16, width=1)
    draw.text((W // 2, 725), "STARTING PITCHERS", font=fonts["xs"], fill=DIM, anchor="mm")

    if away_pitcher:
        draw.text((250, 775), away_pitcher, font=fonts["m"], fill=WHITE, anchor="mm")
        if away_era is not None:
            draw.text((250, 830), f"{away_era:.2f} ERA", font=fonts["s"], fill=MUTED, anchor="mm")

    draw.text((W // 2, 800), "vs", font=fonts["s"], fill=DIM, anchor="mm")

    if home_pitcher:
        draw.text((950, 775), home_pitcher, font=fonts["m"], fill=WHITE, anchor="mm")
        if home_era is not None:
            draw.text((950, 830), f"{home_era:.2f} ERA", font=fonts["s"], fill=MUTED, anchor="mm")

    # Inline stats row (y=910)
    stats_parts = []
    if away_l10 and home_l10:
        stats_parts.append(f"L10: {away_l10} / {home_l10}")
    if away_ats and home_ats:
        stats_parts.append(f"ATS: {away_ats} / {home_ats}")
    if away_ou and home_ou:
        stats_parts.append(f"O/U: {away_ou} / {home_ou}")
    if stats_parts:
        stats_line = "    ".join(stats_parts)
        draw.text((W // 2, 930), stats_line, font=fonts["xs"], fill=MUTED, anchor="mm")

    # Footer
    draw.text((60, H - 50), "truline.app", font=fonts["footer"], fill=ACCENT)
    draw.text((W - 60, H - 50), "@trulineapp", font=fonts["footer"], fill=ACCENT, anchor="ra")

    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return buf.getvalue()
```

- [ ] **Step 2: Generate test card and verify visually**

```bash
cd /Applications/XAMPP/xamppfiles/htdocs/Sites/Truline/backend && python3 -c "
from src.services.social.image_generator import generate_nrfi_card

# Full stats
png = generate_nrfi_card(
    away_team='NYY', home_team='LAD',
    away_name='Yankees', home_name='Dodgers',
    nrfi_pct=72.0,
    away_pitcher='Cole', away_era=3.12,
    home_pitcher='Yamamoto', home_era=2.89,
    game_time='7:05 PM ET',
    away_record='12-5', home_record='14-3',
    away_l10='7-3', home_l10='8-2',
    away_div_rank='1st AL East', home_div_rank='1st NL West',
    away_ats='10-7', home_ats='12-5',
    away_ou='9-8', home_ou='11-6',
)
with open('/tmp/nrfi_v2_full.png', 'wb') as f:
    f.write(png)

# No stats (fallback)
png2 = generate_nrfi_card(
    away_team='NYY', home_team='LAD',
    away_name='Yankees', home_name='Dodgers',
    nrfi_pct=72.0,
    away_pitcher='Cole', away_era=3.12,
    home_pitcher='Yamamoto', home_era=2.89,
)
with open('/tmp/nrfi_v2_nostats.png', 'wb') as f:
    f.write(png2)

# Long team names
png3 = generate_nrfi_card(
    away_team='MIN', home_team='ARI',
    away_name='Timberwolves', home_name='D-backs',
    nrfi_pct=65.0,
    away_pitcher='Paddack', away_era=4.12,
    home_pitcher='Gallen', home_era=3.22,
    away_record='8-9', home_record='10-7',
    away_l10='4-6', home_l10='6-4',
)
with open('/tmp/nrfi_v2_long.png', 'wb') as f:
    f.write(png3)

print('All 3 cards generated in /tmp/')
"
```

Open `/tmp/nrfi_v2_full.png` and verify:
- Team color bars visible at top/bottom
- Gradient background (not flat)
- NRFI % is large and readable
- Stats line shows at bottom
- Text readable at 50% zoom

- [ ] **Step 3: Commit**

```bash
git add src/services/social/image_generator.py
git commit -m "feat: redesign NRFI card — 1200x1200, team colors, bigger fonts"
```

---

### Task 4: Redesign final score card (1200x1200)

**Files:**
- Modify: `src/services/social/image_generator.py` — replace `generate_final_card()` (lines 438-560)

- [ ] **Step 1: Replace `generate_final_card` with the redesigned version**

Replace the entire function with:

```python
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
    draw = ImageDraw.Draw(img)
    fonts = _get_fonts()

    _draw_gradient_bg_fast(img, W, H)
    draw = ImageDraw.Draw(img)
    _draw_team_bars(draw, W, H, away_team, home_team)

    # Top label
    draw.text((60, 30), "FINAL", font=fonts["s"], fill=ACCENT)

    # Logos
    away_logo = _fetch_logo(away_team)
    home_logo = _fetch_logo(home_team)
    _paste_logo(img, away_logo, 200, 190, max_size=220)
    _paste_logo(img, home_logo, 1000, 190, max_size=220)

    draw = ImageDraw.Draw(img)

    # Team names
    draw.text((200, 330), away_name.upper(), font=fonts["l"], fill=WHITE, anchor="mm")
    draw.text((1000, 330), home_name.upper(), font=fonts["l"], fill=WHITE, anchor="mm")

    # Records
    if away_record:
        draw.text((200, 385), away_record, font=fonts["xs"], fill=MUTED, anchor="mm")
    if home_record:
        draw.text((1000, 385), home_record, font=fonts["xs"], fill=MUTED, anchor="mm")

    # Center: final score
    score_text = f"{away_score}  —  {home_score}"
    draw.text((W // 2, 490), score_text, font=fonts["hero"], fill=WHITE, anchor="mm")

    # 1st inning result
    y_mid = 610
    if away_first is not None and home_first is not None:
        is_nrfi = (away_first + home_first) == 0
        tag = "NRFI" if is_nrfi else "YRFI"
        tag_color = GREEN if is_nrfi else AMBER
        fi_text = f"1st Inning: {away_first}-{home_first} ({tag})"
        draw.text((W // 2, y_mid), fi_text, font=fonts["m"], fill=tag_color, anchor="mm")
        y_mid += 80

    # Pick result card
    if pick_team and pick_result:
        result_color = GREEN if pick_result == "win" else (RED if pick_result == "loss" else MUTED)
        result_label = "W" if pick_result == "win" else ("L" if pick_result == "loss" else "P")
        pick_label = pick_type.upper() if pick_type else ""
        line_str = f" {pick_line:+g}" if pick_line is not None else ""
        pick_text = f"Our Pick: {pick_team} {pick_label}{line_str} — {result_label}"
        _draw_rounded_rect(draw, [200, y_mid - 10, 1000, y_mid + 60], fill=CARD, outline=SURFACE, radius=12)
        draw.text((W // 2, y_mid + 25), pick_text, font=fonts["s"], fill=result_color, anchor="mm")
        y_mid += 90

    # Winner highlight
    if away_score > home_score:
        winner_name = away_name
    elif home_score > away_score:
        winner_name = home_name
    else:
        winner_name = None
    if winner_name:
        draw.text((W // 2, y_mid + 10), f"{winner_name.upper()} WINS", font=fonts["l"], fill=WHITE, anchor="mm")

    # Inline stats
    stats_parts = []
    if away_l10 and home_l10:
        stats_parts.append(f"L10: {away_l10} / {home_l10}")
    if away_ats and home_ats:
        stats_parts.append(f"ATS: {away_ats} / {home_ats}")
    if away_ou and home_ou:
        stats_parts.append(f"O/U: {away_ou} / {home_ou}")
    if stats_parts:
        stats_line = "    ".join(stats_parts)
        draw.text((W // 2, H - 110), stats_line, font=fonts["xs"], fill=MUTED, anchor="mm")

    # Footer
    draw.text((60, H - 50), "truline.app", font=fonts["footer"], fill=ACCENT)
    draw.text((W - 60, H - 50), "@trulineapp", font=fonts["footer"], fill=ACCENT, anchor="ra")

    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return buf.getvalue()
```

- [ ] **Step 2: Generate test card and verify**

```bash
cd /Applications/XAMPP/xamppfiles/htdocs/Sites/Truline/backend && python3 -c "
from src.services.social.image_generator import generate_final_card
png = generate_final_card(
    away_team='NYY', home_team='LAD',
    away_name='Yankees', home_name='Dodgers',
    away_score=4, home_score=7,
    away_first=0, home_first=1,
    pick_team='LAD', pick_type='moneyline', pick_line=None, pick_result='win',
    away_record='12-5', home_record='14-3',
    away_l10='7-3', home_l10='8-2',
    away_ats='10-7', home_ats='12-5',
    away_ou='9-8', home_ou='11-6',
)
with open('/tmp/final_v2.png', 'wb') as f:
    f.write(png)
print('Final card generated')
"
```

Open `/tmp/final_v2.png` and verify score is large, pick result visible, team colors at edges.

- [ ] **Step 3: Commit**

```bash
git add src/services/social/image_generator.py
git commit -m "feat: redesign final score card — 1200x1200, team colors, bigger fonts"
```

---

### Task 5: Redesign 1st inning recap card (1200x1200)

**Files:**
- Modify: `src/services/social/image_generator.py` — replace `generate_recap_card()` (lines 266-343)

- [ ] **Step 1: Replace `generate_recap_card` with the redesigned version**

```python
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
    draw = ImageDraw.Draw(img)
    fonts = _get_fonts()

    _draw_gradient_bg_fast(img, W, H)
    draw = ImageDraw.Draw(img)
    _draw_team_bars(draw, W, H, away_team, home_team)

    # Top label
    draw.text((60, 30), "1ST INNING RECAP", font=fonts["s"], fill=ACCENT)

    # Logos
    away_logo = _fetch_logo(away_team)
    home_logo = _fetch_logo(home_team)
    _paste_logo(img, away_logo, 200, 200, max_size=220)
    _paste_logo(img, home_logo, 1000, 200, max_size=220)

    draw = ImageDraw.Draw(img)

    # Team names
    draw.text((200, 340), away_name.upper(), font=fonts["l"], fill=WHITE, anchor="mm")
    draw.text((1000, 340), home_name.upper(), font=fonts["l"], fill=WHITE, anchor="mm")

    # Records
    away_sub = _build_team_sub(away_record, away_div_rank)
    home_sub = _build_team_sub(home_record, home_div_rank)
    if away_sub:
        draw.text((200, 400), away_sub, font=fonts["xs"], fill=MUTED, anchor="mm")
    if home_sub:
        draw.text((1000, 400), home_sub, font=fonts["xs"], fill=MUTED, anchor="mm")

    # Center: 1st inning score
    score_text = f"{away_first}  —  {home_first}"
    draw.text((W // 2, 520), score_text, font=fonts["hero"], fill=WHITE, anchor="mm")
    draw.text((W // 2, 630), "1ST INNING", font=fonts["s"], fill=MUTED, anchor="mm")

    # Result tag
    if is_nrfi:
        draw.text((W // 2, 740), "NRFI", font=fonts["xl"], fill=GREEN, anchor="mm")
        draw.text((W // 2, 840), "MODEL CALLED IT", font=fonts["s"], fill=GREEN, anchor="mm")
    else:
        draw.text((W // 2, 740), "YRFI", font=fonts["xl"], fill=AMBER, anchor="mm")
        draw.text((W // 2, 840), "RUNS IN THE 1ST", font=fonts["s"], fill=AMBER, anchor="mm")

    if predicted_nrfi_pct is not None:
        draw.text(
            (W // 2, 920),
            f"Model predicted {predicted_nrfi_pct:.0f}% NRFI",
            font=fonts["s"],
            fill=MUTED,
            anchor="mm",
        )

    # Footer
    draw.text((60, H - 50), "truline.app", font=fonts["footer"], fill=ACCENT)
    draw.text((W - 60, H - 50), "@trulineapp", font=fonts["footer"], fill=ACCENT, anchor="ra")

    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return buf.getvalue()
```

- [ ] **Step 2: Generate test card and verify**

```bash
cd /Applications/XAMPP/xamppfiles/htdocs/Sites/Truline/backend && python3 -c "
from src.services.social.image_generator import generate_recap_card
png = generate_recap_card(
    away_team='BOS', home_team='NYY',
    away_name='Red Sox', home_name='Yankees',
    away_first=0, home_first=0,
    is_nrfi=True, predicted_nrfi_pct=71.0,
    away_record='9-8', home_record='12-5',
    away_div_rank='3rd AL East', home_div_rank='1st AL East',
)
with open('/tmp/recap_v2.png', 'wb') as f:
    f.write(png)
print('Recap card generated')
"
```

- [ ] **Step 3: Commit**

```bash
git add src/services/social/image_generator.py
git commit -m "feat: redesign 1st inning recap card — 1200x1200, team colors, bigger fonts"
```

---

### Task 6: Add context helper functions for richer tweet copy

**Files:**
- Modify: `src/services/social/content.py` — add new functions after `_get_team_first_inning_pct` (after line 250)

- [ ] **Step 1: Add `_get_pitcher_first_inning_record`**

Add after line 250 in content.py:

```python
async def _get_pitcher_first_inning_record(
    session: AsyncSession, pitcher_id: int | None
) -> tuple[int, int] | None:
    """Return (scoreless_starts, total_starts) for a pitcher's 1st-inning record.

    Checks games where this pitcher was the starter and looks at whether
    the opposing team scored in the 1st inning.
    """
    if not pitcher_id:
        return None

    result = await session.execute(
        text("""
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN opp_first = 0 THEN 1 ELSE 0 END) AS scoreless
            FROM (
                SELECT COALESCE(away_first_inning_runs, 0) AS opp_first
                FROM mlb_games
                WHERE home_starter_id = :pid AND status = 'final'
                  AND home_first_inning_runs IS NOT NULL
                UNION ALL
                SELECT COALESCE(home_first_inning_runs, 0) AS opp_first
                FROM mlb_games
                WHERE away_starter_id = :pid AND status = 'final'
                  AND away_first_inning_runs IS NOT NULL
            ) t
        """),
        {"pid": pitcher_id},
    )
    row = result.fetchone()
    if not row or not row.total or int(row.total) == 0:
        return None
    return (int(row.scoreless or 0), int(row.total))


async def _get_team_streak(session: AsyncSession, team_abbr: str) -> str | None:
    """Return current win/loss streak like 'W4' or 'L2'. None if no data."""
    result = await session.execute(
        text("""
            SELECT
                home_team, away_team, home_score, away_score
            FROM mlb_games
            WHERE (home_team = :team OR away_team = :team)
              AND status = 'final' AND game_type = 'R'
            ORDER BY game_date DESC, game_time DESC
            LIMIT 20
        """),
        {"team": team_abbr},
    )
    rows = result.fetchall()
    if not rows:
        return None

    streak_type = None
    streak_count = 0
    for row in rows:
        if row.home_score is None or row.away_score is None:
            continue
        is_home = row.home_team == team_abbr
        won = (is_home and row.home_score > row.away_score) or \
              (not is_home and row.away_score > row.home_score)
        current = "W" if won else "L"
        if streak_type is None:
            streak_type = current
            streak_count = 1
        elif current == streak_type:
            streak_count += 1
        else:
            break

    if streak_type and streak_count >= 2:
        return f"{streak_type}{streak_count}"
    return None


async def _get_bet_type_summary(
    session: AsyncSession, days: int = 7
) -> dict[str, dict[str, int]]:
    """Return recent bet performance by type. E.g. {'moneyline': {'wins': 3, 'losses': 1}}."""
    cutoff = date.today() - timedelta(days=days)
    result = await session.execute(
        select(MLBPredictionSnapshot).where(
            and_(
                MLBPredictionSnapshot.game_date >= cutoff,
                MLBPredictionSnapshot.best_bet_result.isnot(None),
            )
        )
    )
    snapshots = list(result.scalars().all())

    summary: dict[str, dict[str, int]] = {}
    for s in snapshots:
        bt = (s.best_bet_type or "unknown").lower()
        if bt not in summary:
            summary[bt] = {"wins": 0, "losses": 0, "pushes": 0}
        if s.best_bet_result == "win":
            summary[bt]["wins"] += 1
        elif s.best_bet_result == "loss":
            summary[bt]["losses"] += 1
        else:
            summary[bt]["pushes"] += 1

    return summary
```

- [ ] **Step 2: Test imports**

```bash
cd /Applications/XAMPP/xamppfiles/htdocs/Sites/Truline/backend && python3 -c "
from src.services.social.content import (
    _get_pitcher_first_inning_record,
    _get_team_streak,
    _get_bet_type_summary,
)
print('All helpers loaded OK')
"
```

- [ ] **Step 3: Commit**

```bash
git add src/services/social/content.py
git commit -m "feat: add context helpers — pitcher 1st inning record, streaks, bet type summary"
```

---

### Task 7: Rewrite daily picks thread copy

**Files:**
- Modify: `src/services/social/content.py` — replace `generate_daily_picks_thread()` (lines 260-429)

- [ ] **Step 1: Replace `generate_daily_picks_thread` with narrative version**

Replace the entire function:

```python
async def generate_daily_picks_thread(session: AsyncSession, game_date: date) -> list[str]:
    """Generate a Twitter thread with today's best MLB value bets.

    Uses narrative voice with data-backed analysis instead of raw stat dumps.
    Returns a list of tweet-length strings (max 280 chars each).
    """
    tweets = []

    # Today's scheduled games
    games_result = await session.execute(
        select(MLBGame).where(
            and_(
                MLBGame.game_date == game_date,
                MLBGame.status == "scheduled",
            )
        ).order_by(MLBGame.game_time)
    )
    games = list(games_result.scalars().all())
    if not games:
        return []

    game_ids = [g.game_id for g in games]

    # Predictions (moneyline)
    pred_result = await session.execute(
        select(MLBPrediction).where(
            and_(
                MLBPrediction.game_id.in_(game_ids),
                MLBPrediction.market_type == "moneyline",
            )
        )
    )
    pred_map = {p.game_id: p for p in pred_result.scalars().all()}

    # Markets (moneyline)
    mkt_result = await session.execute(
        select(MLBMarket).where(
            and_(
                MLBMarket.game_id.in_(game_ids),
                MLBMarket.market_type == "moneyline",
            )
        ).order_by(desc(MLBMarket.updated_at))
    )
    mkt_map: dict[str, MLBMarket] = {}
    for m in mkt_result.scalars().all():
        if m.game_id not in mkt_map:
            mkt_map[m.game_id] = m

    # Build value plays
    value_plays = []
    for game in games:
        pred = pred_map.get(game.game_id)
        mkt = mkt_map.get(game.game_id)
        if not pred or not mkt:
            continue
        if pred.p_home_win is None or pred.p_away_win is None:
            continue
        if mkt.home_odds is None or mkt.away_odds is None:
            continue

        p_home = float(pred.p_home_win)
        p_away = float(pred.p_away_win)
        home_odds = float(mkt.home_odds)
        away_odds = float(mkt.away_odds)
        mk_home = _implied_prob_from_decimal(home_odds)
        mk_away = _implied_prob_from_decimal(away_odds)

        home_edge = p_home - mk_home
        away_edge = p_away - mk_away

        if home_edge >= away_edge:
            team = game.home_team
            model_p = p_home
            market_p = mk_home
            odds = home_odds
            edge = home_edge
            is_underdog = home_odds >= 2.0
        else:
            team = game.away_team
            model_p = p_away
            market_p = mk_away
            odds = away_odds
            edge = away_edge
            is_underdog = away_odds >= 2.0

        if edge <= 0.05:
            continue
        confidence = (pred.confidence or "").lower()
        if confidence and confidence not in ("high", "medium"):
            continue

        value_plays.append({
            "game": game,
            "team": team,
            "model_p": model_p,
            "market_p": market_p,
            "odds": odds,
            "edge": edge,
            "is_underdog": is_underdog,
        })

    value_plays.sort(key=lambda x: x["edge"], reverse=True)

    # Header tweet — natural voice
    day_name = game_date.strftime("%A")
    dogs = sum(1 for p in value_plays[:5] if p["is_underdog"])
    favs = min(len(value_plays), 5) - dogs

    pick_desc_parts = []
    if dogs:
        pick_desc_parts.append(f"{dogs} underdog{'s' if dogs != 1 else ''}")
    if favs:
        pick_desc_parts.append(f"{favs} favorite{'s' if favs != 1 else ''}")
    pick_desc = " and ".join(pick_desc_parts) if pick_desc_parts else "no value plays"

    if value_plays:
        header = (
            f"MLB picks for {day_name} — {len(games)} games, "
            f"{min(len(value_plays), 5)} cleared the model.\n\n"
            f"Today's card: {pick_desc}.\n\n"
            f"truline.app\n\n"
            f"#MLB #GamblingX"
        )
    else:
        header = (
            f"MLB picks for {day_name} — {len(games)} games analyzed, "
            f"nothing cleared the model today.\n\n"
            f"truline.app\n\n"
            f"#MLB #GamblingX"
        )
    tweets.append(header)

    if not value_plays:
        return tweets

    for play in value_plays[:5]:
        game = play["game"]
        away = game.away_team
        home = game.home_team
        team = play["team"]
        model_pct = round(play["model_p"] * 100)
        market_pct = round(play["market_p"] * 100)
        edge_pct = play["edge"] * 100
        odds_str = _fmt_odds(play["odds"])

        game_time = ""
        if game.game_time:
            et = game.game_time - timedelta(hours=4)
            try:
                game_time = et.strftime("%-I:%M %p")
            except Exception:
                game_time = et.strftime("%I:%M %p").lstrip("0")

        away_name = TEAM_NAMES.get(away, away)
        home_name = TEAM_NAMES.get(home, home)
        team_name = TEAM_NAMES.get(team, team)
        hashtag = TEAM_HASHTAGS.get(team, "")

        # Fetch context for narrative
        l10 = None
        stat_row = await session.execute(
            select(MLBTeamStats).where(
                MLBTeamStats.team_abbr == team
            ).order_by(desc(MLBTeamStats.stat_date)).limit(1)
        )
        stat = stat_row.scalar_one_or_none()
        if stat:
            l10 = stat.last_10_record or (
                f"{stat.last_10_wins}-{stat.last_10_losses}"
                if stat.last_10_wins is not None and stat.last_10_losses is not None
                else None
            )

        streak = await _get_team_streak(session, team)

        # Pitcher context
        starter_id = game.home_starter_id if team == home else game.away_starter_id
        pitcher_last, pitcher_era = await _get_pitcher_era(session, starter_id)
        fi_record = await _get_pitcher_first_inning_record(session, starter_id)

        # Build narrative
        matchup_line = f"{away_name} @ {home_name}"
        if game_time:
            matchup_line += f" — {game_time}"

        context_bits = []
        if l10:
            context_bits.append(f"{l10} in their last 10")
        if streak:
            context_bits.append(f"on a {streak[1:]}-game {'win' if streak[0] == 'W' else 'losing'} streak")

        pitcher_bit = ""
        if pitcher_last and fi_record and fi_record[1] >= 3:
            pitcher_bit = f" {pitcher_last} ({pitcher_era:.2f} ERA) has held opponents scoreless in the 1st in {fi_record[0]} of {fi_record[1]} starts."

        context_str = ""
        if context_bits:
            context_str = f" They're {', '.join(context_bits)}."

        dog_or_fav = "underdog" if play["is_underdog"] else "favorite"

        tweet = (
            f"{matchup_line}\n\n"
            f"Model likes {team_name} ML at {odds_str} ({dog_or_fav}).{context_str}{pitcher_bit}\n\n"
            f"{model_pct}% model vs {market_pct}% market — {edge_pct:.1f}% edge.\n\n"
            f"{hashtag} #MLB"
        ).strip()

        if len(tweet) > 280:
            # Trim pitcher bit first, then context
            tweet = (
                f"{matchup_line}\n\n"
                f"Model likes {team_name} ML at {odds_str}.{context_str}\n\n"
                f"{model_pct}% vs {market_pct}% — {edge_pct:.1f}% edge.\n\n"
                f"{hashtag} #MLB"
            ).strip()
        if len(tweet) > 280:
            tweet = (
                f"{matchup_line}\n\n"
                f"Model likes {team_name} ML at {odds_str}.\n\n"
                f"{model_pct}% vs {market_pct}% — {edge_pct:.1f}% edge.\n\n"
                f"{hashtag} #MLB"
            ).strip()
        if len(tweet) > 280:
            tweet = tweet[:277] + "..."
        tweets.append(tweet)

    return tweets
```

- [ ] **Step 2: Test import**

```bash
cd /Applications/XAMPP/xamppfiles/htdocs/Sites/Truline/backend && python3 -c "
from src.services.social.content import generate_daily_picks_thread
print('Loaded OK')
"
```

- [ ] **Step 3: Commit**

```bash
git add src/services/social/content.py
git commit -m "feat: rewrite daily picks thread with narrative voice and context"
```

---

### Task 8: Rewrite pregame NRFI tweet copy

**Files:**
- Modify: `src/services/social/content.py` — replace `generate_pregame_nrfi_tweet()` (lines 682-744)

- [ ] **Step 1: Replace `generate_pregame_nrfi_tweet` with narrative version**

```python
async def generate_pregame_nrfi_tweet(session: AsyncSession, game: MLBGame) -> str | None:
    """Generate a single-game NRFI pregame pick tweet with pitcher context."""
    home_off, home_def = await _get_team_first_inning_pct(session, game.home_team)
    away_off, away_def = await _get_team_first_inning_pct(session, game.away_team)
    if home_off is None or away_off is None or home_def is None or away_def is None:
        return None

    away_last, away_era = await _get_pitcher_era(session, game.away_starter_id)
    home_last, home_era = await _get_pitcher_era(session, game.home_starter_id)
    if not away_last or not home_last:
        return None

    p_away_scores = (away_off + home_def) / 2.0
    p_home_scores = (home_off + away_def) / 2.0
    nrfi_pct = (1.0 - p_away_scores) * (1.0 - p_home_scores) * 100.0
    nrfi_pct_rounded = round(nrfi_pct)

    away_name = TEAM_NAMES.get(game.away_team, game.away_team)
    home_name = TEAM_NAMES.get(game.home_team, game.home_team)
    away_handle = TEAM_HANDLES.get(game.away_team, game.away_team)
    home_handle = TEAM_HANDLES.get(game.home_team, game.home_team)

    game_time_str = _fmt_game_time_et(game.game_time)

    # Pitcher 1st-inning records for context
    away_fi = await _get_pitcher_first_inning_record(session, game.away_starter_id)
    home_fi = await _get_pitcher_first_inning_record(session, game.home_starter_id)

    pitcher_context = ""
    if away_fi and home_fi and away_fi[1] >= 3 and home_fi[1] >= 3:
        pitcher_context = (
            f"{home_last} has blanked the 1st in {home_fi[0]} of {home_fi[1]} starts, "
            f"{away_last} in {away_fi[0]} of {away_fi[1]}."
        )
    elif home_fi and home_fi[1] >= 3:
        pitcher_context = f"{home_last} has blanked the 1st in {home_fi[0]} of {home_fi[1]} starts."
    elif away_fi and away_fi[1] >= 3:
        pitcher_context = f"{away_last} has blanked the 1st in {away_fi[0]} of {away_fi[1]} starts."

    matchup = f"{away_name} @ {home_name}"
    if game_time_str:
        matchup += f", {game_time_str}"

    if nrfi_pct_rounded >= 70:
        confidence = f"NRFI at {nrfi_pct_rounded}% — strong lean."
    elif nrfi_pct_rounded >= 60:
        confidence = f"NRFI at {nrfi_pct_rounded}%."
    else:
        confidence = f"NRFI at {nrfi_pct_rounded}% — slight lean."

    parts = [matchup, "", confidence]
    if pitcher_context:
        parts.append(pitcher_context)
    parts.extend(["", f"{away_handle} vs {home_handle}", "", "#NRFI #MLB"])

    tweet = "\n".join(parts)
    if len(tweet) > 280:
        # Drop pitcher context
        parts = [matchup, "", confidence, "", f"{away_handle} vs {home_handle}", "", "#NRFI #MLB"]
        tweet = "\n".join(parts)
    if len(tweet) > 280:
        tweet = tweet[:277] + "..."
    return tweet
```

- [ ] **Step 2: Test import**

```bash
cd /Applications/XAMPP/xamppfiles/htdocs/Sites/Truline/backend && python3 -c "
from src.services.social.content import generate_pregame_nrfi_tweet
print('Loaded OK')
"
```

- [ ] **Step 3: Commit**

```bash
git add src/services/social/content.py
git commit -m "feat: rewrite pregame NRFI tweet with pitcher 1st-inning context"
```

---

### Task 9: Rewrite results recap tweet copy

**Files:**
- Modify: `src/services/social/content.py` — replace `generate_results_tweet()` (lines 511-593)

- [ ] **Step 1: Replace `generate_results_tweet` with narrative version**

```python
async def generate_results_tweet(session: AsyncSession, game_date: date) -> str | None:
    """Generate a results recap tweet with narrative context."""
    stmt = select(MLBPredictionSnapshot).where(
        and_(
            MLBPredictionSnapshot.game_date == game_date,
            MLBPredictionSnapshot.best_bet_result.isnot(None),
        )
    )
    result = await session.execute(stmt)
    snapshots = list(result.scalars().all())

    if not snapshots:
        return None

    wins = sum(1 for s in snapshots if s.best_bet_result == "win")
    losses = sum(1 for s in snapshots if s.best_bet_result == "loss")
    pushes = sum(1 for s in snapshots if s.best_bet_result == "push")
    profit = sum(float(s.best_bet_profit or 0) for s in snapshots)

    total = wins + losses
    win_rate = round(wins / total * 100, 1) if total > 0 else 0

    profit_str = f"+{profit:.2f}" if profit >= 0 else f"{profit:.2f}"

    # NRFI results
    nrfi_result = await session.execute(
        select(MLBGame).where(
            and_(
                MLBGame.game_date == game_date,
                MLBGame.status == "final",
                MLBGame.home_first_inning_runs.isnot(None),
            )
        )
    )
    nrfi_games = list(nrfi_result.scalars().all())
    nrfi_count = sum(1 for g in nrfi_games
                     if (g.home_first_inning_runs or 0) + (g.away_first_inning_runs or 0) == 0)
    nrfi_pct = round(nrfi_count / len(nrfi_games) * 100) if nrfi_games else 0

    # Best / worst
    graded = [s for s in snapshots if s.best_bet_profit is not None]
    best_line = ""
    worst_line = ""
    if graded:
        sorted_by_profit = sorted(graded, key=lambda s: float(s.best_bet_profit or 0), reverse=True)
        best = sorted_by_profit[0]
        worst = sorted_by_profit[-1]

        def _fmt_snap_name(s):
            team = TEAM_NAMES.get(s.best_bet_team, s.best_bet_team or "?")
            btype = (s.best_bet_type or "").lower()
            label = "ML" if btype == "moneyline" else ("runline" if btype == "runline" else ("O/U" if btype == "total" else ""))
            return f"{team} {label}".strip()

        if float(best.best_bet_profit or 0) > 0:
            best_line = f"Best: {_fmt_snap_name(best)} W\n"
        if float(worst.best_bet_profit or 0) < 0:
            worst_line = f"Worst: {_fmt_snap_name(worst)} L\n"

    # Bet type context
    type_summary = await _get_bet_type_summary(session, days=7)
    type_context = ""
    ml_data = type_summary.get("moneyline")
    if ml_data and (ml_data["wins"] + ml_data["losses"]) >= 3:
        ml_total = ml_data["wins"] + ml_data["losses"]
        type_context = f" ML picks are {ml_data['wins']}-{ml_data['losses']} this week."

    # Build tweet
    record_str = f"{wins}-{losses}"
    if pushes:
        record_str += f"-{pushes}"

    tweet = f"Yesterday: {record_str}, {profit_str}u.{type_context}\n"
    if nrfi_games:
        tweet += f"NRFI: {nrfi_count}/{len(nrfi_games)} ({nrfi_pct}%).\n"
    if best_line or worst_line:
        tweet += "\n" + best_line + worst_line

    tweet += "\ntruline.app\n\n#MLB #SportsBetting"

    if len(tweet) > 280:
        # Drop type context
        tweet = f"Yesterday: {record_str}, {profit_str}u.\n"
        if nrfi_games:
            tweet += f"NRFI: {nrfi_count}/{len(nrfi_games)} ({nrfi_pct}%).\n"
        if best_line or worst_line:
            tweet += "\n" + best_line + worst_line
        tweet += "\ntruline.app\n\n#MLB #SportsBetting"

    if len(tweet) > 280:
        tweet = tweet[:277] + "..."

    return tweet
```

- [ ] **Step 2: Test import**

```bash
cd /Applications/XAMPP/xamppfiles/htdocs/Sites/Truline/backend && python3 -c "
from src.services.social.content import generate_results_tweet
print('Loaded OK')
"
```

- [ ] **Step 3: Commit**

```bash
git add src/services/social/content.py
git commit -m "feat: rewrite results recap with narrative voice and bet type context"
```

---

### Task 10: Visual verification — generate all card types and review

**Files:** None (testing only)

- [ ] **Step 1: Generate all 3 card types with test data**

```bash
cd /Applications/XAMPP/xamppfiles/htdocs/Sites/Truline/backend && python3 -c "
from src.services.social.image_generator import generate_nrfi_card, generate_final_card, generate_recap_card

# 1. NRFI card — full stats
png = generate_nrfi_card(
    away_team='HOU', home_team='NYY',
    away_name='Astros', home_name='Yankees',
    nrfi_pct=68.0,
    away_pitcher='Verlander', away_era=3.22,
    home_pitcher='Cole', home_era=2.95,
    game_time='7:05 PM ET',
    away_record='11-6', home_record='13-4',
    away_l10='6-4', home_l10='8-2',
    away_div_rank='1st AL West', home_div_rank='1st AL East',
    away_ats='9-8', home_ats='11-6',
    away_ou='10-7', home_ou='8-9',
)
with open('/tmp/card_nrfi_final.png', 'wb') as f:
    f.write(png)

# 2. Final score card
png2 = generate_final_card(
    away_team='BOS', home_team='TB',
    away_name='Red Sox', home_name='Rays',
    away_score=3, home_score=5,
    away_first=0, home_first=0,
    pick_team='TB', pick_type='moneyline', pick_result='win',
    away_record='8-9', home_record='10-7',
    away_l10='4-6', home_l10='7-3',
    away_ats='6-11', home_ats='9-8',
    away_ou='8-9', home_ou='10-7',
)
with open('/tmp/card_final_final.png', 'wb') as f:
    f.write(png2)

# 3. Recap card
png3 = generate_recap_card(
    away_team='LAD', home_team='SF',
    away_name='Dodgers', home_name='Giants',
    away_first=2, home_first=0,
    is_nrfi=False, predicted_nrfi_pct=55.0,
    away_record='14-3', home_record='7-10',
    away_div_rank='1st NL West', home_div_rank='4th NL West',
)
with open('/tmp/card_recap_final.png', 'wb') as f:
    f.write(png3)

# 4. NRFI card — no stats (fallback test)
png4 = generate_nrfi_card(
    away_team='CWS', home_team='DET',
    away_name='White Sox', home_name='Tigers',
    nrfi_pct=61.0,
    away_pitcher='Cease', away_era=4.55,
    home_pitcher='Skubal', home_era=2.80,
)
with open('/tmp/card_nrfi_nostats.png', 'wb') as f:
    f.write(png4)

print('All 4 test cards generated in /tmp/')
print('Open them and verify readability at 50% zoom.')
"
```

- [ ] **Step 2: Open and verify each card**

Check each card at 50% zoom for:
- Team color bars at top/bottom
- Gradient background (not flat)
- All text readable
- Stats showing (or graceful fallback when missing)
- No text overlapping

- [ ] **Step 3: Adjust any layout issues found**

If text overlaps or is too small, adjust Y-coordinates or font sizes in the card generators and re-run step 1.

- [ ] **Step 4: Final commit if adjustments were made**

```bash
git add src/services/social/image_generator.py
git commit -m "fix: adjust card layouts after visual verification"
```

---

### Task 11: Deploy and verify on Railway

**Files:** None (deployment)

- [ ] **Step 1: Push to main**

```bash
git push origin main
```

Railway auto-deploys from main.

- [ ] **Step 2: Check Railway logs for stats loading**

Look for `get_team_card_stats` log lines to verify stats are populating:
- `get_team_card_stats: stats found` — good
- `get_team_card_stats: no MLBTeamStats row` — investigate
- `get_team_card_stats: no MLBTeam row` — means `mlb_teams` table may need seeding

- [ ] **Step 3: Verify next scheduled post**

Wait for the next scheduled post cycle and check Blotato/Twitter for:
- New 1200x1200 card format
- Team stats showing on cards
- Narrative copy in tweet text

If dry-run mode is on (`twitter_posting_enabled = False`), check Railway logs for `[BLOTATO-DRY-RUN]` output showing the new tweet text.
