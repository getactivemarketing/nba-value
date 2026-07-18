# SMS Pick Alerts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Text the founder every frozen MLB best_bet pick (score ≥ 40) via Twilio, using the proven flag-and-poll pattern.

**Architecture:** A dependency-free Twilio transport (`send_sms`) and a pure message formatter (`format_pick_alert`) live in a new `src/services/notifications/` package. A `run_pick_alerts` job on the existing social scheduler polls `mlb_prediction_snapshots` every 10 minutes for un-alerted best_bets and flips a new `sms_alert_sent` column only on successful send.

**Tech Stack:** Python 3.11, FastAPI backend on Railway, SQLAlchemy async, httpx (already a dependency — do NOT add the twilio SDK), structlog, pytest.

**Spec:** `docs/superpowers/specs/2026-07-18-sms-pick-alerts-design.md`

## Global Constraints

- Branch: create `sms-pick-alerts` off `main` (NOT off `nfl-phase4-live`); cherry-pick spec commit `12e36e6` onto it first. The main checkout is on `nfl-phase4-live` — use a worktree or stash-safe checkout so NFL work is untouched.
- Env vars (exact names): `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_FROM_NUMBER`, `PICK_ALERT_TO_NUMBER`. If any is unset, `send_sms` logs a warning and returns False. Never hardcode values — the destination number is supplied by the operator at deploy time and must not appear in the repo.
- Alert filter (exact): `game_date == today ET`, `best_bet_team IS NOT NULL`, `best_bet_value_score >= 40`, `sms_alert_sent == FALSE`.
- Log via `structlog` at WARNING for failures (Railway only shows WARN+; INFO is invisible in prod logs).
- All pytest runs from `backend/`: `python3 -m pytest tests/unit/<file> -v`.
- Migrations are applied manually to prod (alembic is stalled at 004 — do not add an alembic revision).
- Commit messages end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task 1: Twilio SMS transport

**Files:**
- Create: `backend/src/services/notifications/__init__.py` (empty)
- Create: `backend/src/services/notifications/sms.py`
- Test: `backend/tests/unit/test_sms_transport.py`

**Interfaces:**
- Consumes: nothing (env vars + httpx only)
- Produces: `send_sms(body: str) -> bool` — True only on Twilio 2xx; False on missing config, HTTP error, or non-2xx.

- [ ] **Step 0: Branch setup**

```bash
cd /Applications/XAMPP/xamppfiles/htdocs/Sites/Truline
git worktree add ../Truline-sms-alerts -b sms-pick-alerts main
cd ../Truline-sms-alerts
# Spec + plan currently live only on nfl-phase4-live — copy them over (hash-independent):
git checkout nfl-phase4-live -- \
  docs/superpowers/specs/2026-07-18-sms-pick-alerts-design.md \
  docs/superpowers/plans/2026-07-18-sms-pick-alerts.md
git commit -m "docs: SMS pick-alerts spec and plan

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

All subsequent work happens in the `Truline-sms-alerts` worktree.

- [ ] **Step 1: Write the failing tests**

`backend/tests/unit/test_sms_transport.py`:

```python
"""Twilio SMS transport: env gating, payload shape, error handling."""

import httpx

import src.services.notifications.sms as sms


class FakeResponse:
    def __init__(self, status_code, text=""):
        self.status_code = status_code
        self.text = text


def _set_env(monkeypatch):
    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "ACtest")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "tok")
    monkeypatch.setenv("TWILIO_FROM_NUMBER", "+15550001111")
    monkeypatch.setenv("PICK_ALERT_TO_NUMBER", "+15552223333")


def test_missing_env_no_ops(monkeypatch):
    for var in ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN",
                "TWILIO_FROM_NUMBER", "PICK_ALERT_TO_NUMBER"):
        monkeypatch.delenv(var, raising=False)
    called = []
    monkeypatch.setattr(sms.httpx, "post", lambda *a, **k: called.append(1))
    assert sms.send_sms("hi") is False
    assert not called


def test_sends_correct_payload_on_201(monkeypatch):
    _set_env(monkeypatch)
    captured = {}

    def fake_post(url, auth=None, data=None, timeout=None):
        captured.update(url=url, auth=auth, data=data)
        return FakeResponse(201)

    monkeypatch.setattr(sms.httpx, "post", fake_post)
    assert sms.send_sms("test body") is True
    assert captured["url"] == "https://api.twilio.com/2010-04-01/Accounts/ACtest/Messages.json"
    assert captured["auth"] == ("ACtest", "tok")
    assert captured["data"] == {"From": "+15550001111", "To": "+15552223333", "Body": "test body"}


def test_non_2xx_returns_false(monkeypatch):
    _set_env(monkeypatch)
    monkeypatch.setattr(sms.httpx, "post", lambda *a, **k: FakeResponse(401, "auth error"))
    assert sms.send_sms("hi") is False


def test_httpx_error_returns_false(monkeypatch):
    _set_env(monkeypatch)

    def boom(*a, **k):
        raise httpx.ConnectError("no network")

    monkeypatch.setattr(sms.httpx, "post", boom)
    assert sms.send_sms("hi") is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && python3 -m pytest tests/unit/test_sms_transport.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.services.notifications'`

- [ ] **Step 3: Write the implementation**

`backend/src/services/notifications/__init__.py`: empty file.

`backend/src/services/notifications/sms.py`:

```python
"""Twilio SMS transport for founder notifications (no Twilio SDK — plain httpx)."""

import os

import httpx
import structlog

logger = structlog.get_logger()

TWILIO_MESSAGES_URL = "https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"


def send_sms(body: str) -> bool:
    """Send an SMS to the founder. Returns True only on a Twilio 2xx.

    No-ops (returns False) when any required env var is unset so local
    dev and misconfigured prod never attempt a send.
    """
    sid = os.getenv("TWILIO_ACCOUNT_SID")
    token = os.getenv("TWILIO_AUTH_TOKEN")
    from_number = os.getenv("TWILIO_FROM_NUMBER")
    to_number = os.getenv("PICK_ALERT_TO_NUMBER")
    if not all([sid, token, from_number, to_number]):
        logger.warning("sms_not_configured")
        return False

    try:
        resp = httpx.post(
            TWILIO_MESSAGES_URL.format(sid=sid),
            auth=(sid, token),
            data={"From": from_number, "To": to_number, "Body": body},
            timeout=15.0,
        )
    except httpx.HTTPError as e:
        logger.warning("sms_send_failed", error=str(e))
        return False

    if resp.status_code // 100 != 2:
        logger.warning("sms_send_failed", status=resp.status_code, detail=resp.text[:200])
        return False
    return True
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && python3 -m pytest tests/unit/test_sms_transport.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add backend/src/services/notifications/ backend/tests/unit/test_sms_transport.py
git commit -m "feat: Twilio SMS transport for pick alerts

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Pick-alert message formatter

**Files:**
- Create: `backend/src/services/notifications/pick_alerts.py`
- Test: `backend/tests/unit/test_pick_alert_format.py`

**Interfaces:**
- Consumes: `_fmt_american(odds_decimal: float) -> str` from `src.services.social.content` (returns e.g. `"+150"`, `"-125"`, `"-"` for invalid).
- Produces: `format_pick_alert(*, away_team: str, home_team: str, bet_type: str, team: str | None, line: float | None, odds_decimal: float | None, value_score: int, edge: float | None, game_time: datetime | None, away_starter: str | None = None, home_starter: str | None = None) -> str`

- [ ] **Step 1: Write the failing tests**

`backend/tests/unit/test_pick_alert_format.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && python3 -m pytest tests/unit/test_pick_alert_format.py -v`
Expected: FAIL — `ModuleNotFoundError` / `ImportError: cannot import name 'format_pick_alert'`

- [ ] **Step 3: Write the implementation**

`backend/src/services/notifications/pick_alerts.py`:

```python
"""Formats frozen best_bet picks as founder SMS alerts."""

from datetime import datetime, timedelta

from src.services.social.content import _fmt_american


def _last_name(full_name: str | None) -> str | None:
    if not full_name or not full_name.strip():
        return None
    return full_name.strip().split()[-1]


def format_pick_alert(
    *,
    away_team: str,
    home_team: str,
    bet_type: str,
    team: str | None,
    line: float | None,
    odds_decimal: float | None,
    value_score: int,
    edge: float | None,
    game_time: datetime | None,
    away_starter: str | None = None,
    home_starter: str | None = None,
) -> str:
    if bet_type == "moneyline":
        label = f"{team} ML"
    elif bet_type == "runline":
        sign = "+" if (line or 0) > 0 else ""
        label = f"{team} {sign}{line:g}"
    else:  # totals — not in best_bet while the re-entry gate is closed
        label = f"O/U {line:g}" if line is not None else "O/U"

    if team is None:
        opponent = f"{away_team} @ {home_team}"
    elif team == home_team:
        opponent = f"vs {away_team}"
    else:
        opponent = f"@ {home_team}"

    odds_str = ""
    if odds_decimal:
        am = _fmt_american(float(odds_decimal))
        if am != "-":
            odds_str = f" ({am})"

    time_str = ""
    if game_time:
        # EDT display, same convention as content.py picks thread
        et = game_time - timedelta(hours=4)
        time_str = ", " + et.strftime("%I:%M %p").lstrip("0") + " ET"

    parts = [f"Score {int(value_score)}"]
    if edge is not None:
        parts.append(f"Edge {round(float(edge) * 100)}%")
    away_p, home_p = _last_name(away_starter), _last_name(home_starter)
    if away_p and home_p:
        parts.append(f"{away_p} vs {home_p}")

    return f"TruLine pick: {label}{odds_str} {opponent}{time_str}\n" + " | ".join(parts)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && python3 -m pytest tests/unit/test_pick_alert_format.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add backend/src/services/notifications/pick_alerts.py backend/tests/unit/test_pick_alert_format.py
git commit -m "feat: pick-alert SMS message formatter

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: `sms_alert_sent` column — model + prod migration

**Files:**
- Modify: `backend/src/models/mlb_prediction_snapshot.py` (after `celebration_tweet_posted`, ~line 99)
- Create: `backend/scripts/migrations/2026-07-18_add_sms_alert_sent.sql` (record of what was run)

**Interfaces:**
- Consumes: nothing
- Produces: `MLBPredictionSnapshot.sms_alert_sent: Mapped[bool]` — Task 4's query and UPDATE depend on this exact name.

- [ ] **Step 1: Add the model column**

In `backend/src/models/mlb_prediction_snapshot.py`, directly below `celebration_tweet_posted`:

```python
    sms_alert_sent: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
```

- [ ] **Step 2: Write the migration SQL file**

`backend/scripts/migrations/2026-07-18_add_sms_alert_sent.sql`:

```sql
-- Founder SMS pick alerts (spec: docs/superpowers/specs/2026-07-18-sms-pick-alerts-design.md)
-- Applied manually to prod 2026-07-18 (alembic stalled at 004; manual ALTER is current practice).
ALTER TABLE mlb_prediction_snapshots
    ADD COLUMN IF NOT EXISTS sms_alert_sent BOOLEAN NOT NULL DEFAULT FALSE;
-- Backfill: never alert on historical picks.
UPDATE mlb_prediction_snapshots SET sms_alert_sent = TRUE;
```

- [ ] **Step 3: Apply to prod and verify**

Run from `backend/`:

```bash
python3 - <<'EOF'
import re, psycopg2
from pathlib import Path
url = re.search(r"postgresql://[^\"'\s]+", Path("src/tasks/prediction_tracker.py").read_text()).group(0)
conn = psycopg2.connect(url)
cur = conn.cursor()
cur.execute(Path("scripts/migrations/2026-07-18_add_sms_alert_sent.sql").read_text())
conn.commit()
cur.execute("SELECT COUNT(*) FILTER (WHERE sms_alert_sent), COUNT(*) FROM mlb_prediction_snapshots")
print(cur.fetchone())
EOF
```

Expected: both counts equal (all rows backfilled TRUE). The ALTER is additive — safe to apply before the code deploys.

- [ ] **Step 4: Commit**

```bash
git add backend/src/models/mlb_prediction_snapshot.py backend/scripts/migrations/2026-07-18_add_sms_alert_sent.sql
git commit -m "feat: sms_alert_sent flag on mlb_prediction_snapshots

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: `run_pick_alerts` job on the social scheduler

**Files:**
- Modify: `backend/src/tasks/social_scheduler.py` — constant near `CELEBRATION_MIN_DECIMAL_ODDS` (~line 522), async job + wrapper near `run_post_celebrations` (~line 684), registration in `start_scheduler()` (~line 835)

**Interfaces:**
- Consumes: `send_sms(body) -> bool` (Task 1), `format_pick_alert(...)` (Task 2), `MLBPredictionSnapshot.sms_alert_sent` (Task 3), existing `_today_et()`, `_social_session_factory`, `_run_async`, `log_task`.
- Produces: `run_pick_alerts()` registered every 10 minutes.

- [ ] **Step 1: Add the constant**

Below `CELEBRATION_MIN_DECIMAL_ODDS`:

```python
# Minimum best_bet value score to text the founder — matches the site's
# "Top Value Picks" threshold (post-tanh rescale; old 65 ≈ new 40).
PICK_ALERT_MIN_SCORE = 40
```

- [ ] **Step 2: Add the async job and wrapper**

After `run_post_celebrations()`:

```python
async def _post_pick_alerts_async() -> dict:
    """Text the founder each frozen best_bet pick (score >= 40), once per snapshot."""
    from sqlalchemy import select, and_, text
    from src.models.mlb_prediction_snapshot import MLBPredictionSnapshot
    from src.services.notifications.pick_alerts import format_pick_alert
    from src.services.notifications.sms import send_sms

    sent = 0
    skipped = 0
    async with _social_session_factory() as session:
        stmt = select(MLBPredictionSnapshot).where(
            and_(
                MLBPredictionSnapshot.game_date == _today_et(),
                MLBPredictionSnapshot.best_bet_team.isnot(None),
                MLBPredictionSnapshot.best_bet_value_score >= PICK_ALERT_MIN_SCORE,
                MLBPredictionSnapshot.sms_alert_sent == False,  # noqa: E712
            )
        )
        snaps = list((await session.execute(stmt)).scalars().all())
        for snap in snaps:
            body = format_pick_alert(
                away_team=snap.away_team,
                home_team=snap.home_team,
                bet_type=snap.best_bet_type,
                team=snap.best_bet_team,
                line=float(snap.best_bet_line) if snap.best_bet_line is not None else None,
                odds_decimal=float(snap.best_bet_odds) if snap.best_bet_odds is not None else None,
                value_score=snap.best_bet_value_score,
                edge=float(snap.best_bet_edge) if snap.best_bet_edge is not None else None,
                game_time=snap.game_time,
                away_starter=snap.away_starter_name,
                home_starter=snap.home_starter_name,
            )
            if send_sms(body):
                await session.execute(
                    text("UPDATE mlb_prediction_snapshots SET sms_alert_sent = TRUE WHERE id = :sid"),
                    {"sid": snap.id},
                )
                sent += 1
            else:
                skipped += 1
        await session.commit()
    return {"sent": sent, "skipped": skipped, "type": "pick_alerts"}


def run_pick_alerts():
    log_task("Sending best-bet pick alert SMS...")
    try:
        result = _run_async(_post_pick_alerts_async())
        log_task("Pick alerts complete", **{k: str(v) for k, v in result.items()})
        return result
    except Exception as e:
        log_task(f"Pick alerts FAILED: {e}")
        return {"status": "failed", "error": str(e)}
```

- [ ] **Step 3: Register the job**

In `start_scheduler()`, next to the celebrations registration:

```python
    # Founder SMS alerts for frozen best_bet picks — every 10 min
    social_scheduler.every(10).minutes.do(run_pick_alerts)
```

- [ ] **Step 4: Sanity-check imports and full unit suite**

Run: `cd backend && python3 -c "from src.tasks.social_scheduler import run_pick_alerts, PICK_ALERT_MIN_SCORE; print('ok')"`
Expected: `ok`

Run: `cd backend && python3 -m pytest tests/unit -v`
Expected: all pass (test_sms_transport 4, test_pick_alert_format 4, plus pre-existing suite green).

- [ ] **Step 5: Commit**

```bash
git add backend/src/tasks/social_scheduler.py
git commit -m "feat: run_pick_alerts SMS job every 10 min on social scheduler

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: End-to-end verification and deploy

**Files:** none created — operational task.

**Interfaces:**
- Consumes: everything above; Twilio credentials from the operator (same account as AfterLine — check `Sites/afterline/.env` locally for `TWILIO_ACCOUNT_SID` / `TWILIO_AUTH_TOKEN` / a from-number). `PICK_ALERT_TO_NUMBER` is supplied by the operator in-session — never commit it.

- [ ] **Step 1: Live local send against prod DB**

Inspect `_init_engine()` in `social_scheduler.py` to mirror its engine setup, then run the job once locally with real env vars (DATABASE_URL from the `prediction_tracker.py` grep trick, Twilio vars from the operator). Example harness:

```bash
cd backend && python3 - <<'EOF'
import asyncio, os, re
from pathlib import Path
os.environ["DATABASE_URL"] = re.search(
    r"postgresql://[^\"'\s]+",
    Path("src/tasks/prediction_tracker.py").read_text()).group(0)
# TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN / TWILIO_FROM_NUMBER / PICK_ALERT_TO_NUMBER
# must already be exported in the shell (operator-provided).
from src.tasks import social_scheduler as ss
ss._init_engine()
print(asyncio.run(ss._post_pick_alerts_async()))
EOF
```

Expected: `{'sent': N, 'skipped': 0, 'type': 'pick_alerts'}` where N = today's un-alerted picks (may be 0 before snapshots run — if 0, temporarily verify with `send_sms("TruLine SMS test")` from a REPL instead), and a text arrives at the founder's phone. Note: any pick alerted here is flagged and will NOT re-text after deploy — that's correct, no double-sends.

- [ ] **Step 2: Merge and push**

```bash
git checkout main && git merge --no-ff sms-pick-alerts -m "Merge sms-pick-alerts: founder SMS for best_bet picks

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
git push origin main
```

Railway auto-deploys from main (`backend/src/**` matches watchPatterns since `2b88639`).

- [ ] **Step 3: Set prod env vars**

Set on the **nba-value** Railway service: `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_FROM_NUMBER`, `PICK_ALERT_TO_NUMBER`. Try `railway variables --set "KEY=value"` first; if the CLI isn't logged in, the founder sets them in the Railway dashboard. Env changes trigger a restart automatically.

Then guard against stale-pick texts on first boot (picks snapshotted between the Task 3 backfill and this deploy whose games already started):

```sql
UPDATE mlb_prediction_snapshots SET sms_alert_sent = TRUE
WHERE sms_alert_sent = FALSE AND game_time < NOW();
```

- [ ] **Step 4: Verify the deploy**

- `railway status --json` — commitHash matches the merge commit (health endpoint alone proves nothing; it stays green on stale containers).
- After the next 10-min cycle: `railway logs <deployment-id> --lines 200` shows `Pick alerts complete` (or `sms_not_configured` if env vars are missing — fix and recheck).
- Tonight's slate: founder receives one text per qualifying pick ~30–45 min before each first pitch; cross-check against the site's Top Value Picks panel.

- [ ] **Step 5: Clean up worktree**

```bash
cd /Applications/XAMPP/xamppfiles/htdocs/Sites/Truline
git worktree remove ../Truline-sms-alerts
```
