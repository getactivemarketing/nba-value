# SMS Pick Alerts — Design

**Date:** 2026-07-18
**Status:** Approved

## Purpose

The founder wants a text message every time a top value pick is made (they missed all 4 winning picks on Jul 17 because there is no direct-to-owner channel — picks only surface on the website and public socials). Deliver each frozen MLB best_bet pick to the founder's phone via SMS, in time to bet it.

## Decisions (confirmed with founder)

- **Delivery:** Twilio SMS (existing Twilio account from AfterLine; TruLine gets its own env vars).
- **Timing:** Per pick, at snapshot time (~30–45 min before first pitch). The snapshot is the lock-in point — what gets texted is exactly what gets graded, so the founder's personal record matches the tracker. No early/provisional alerts (possible later addition: morning digest).
- **Scope of picks:** All best_bet picks with `best_bet_value_score >= 40` — the same set shown in the site's "Top Value Picks" panel and graded by the retune tracker (~3–6/night).
- **Sport:** MLB now; sender is sport-agnostic so NFL (live ~Sep 2026) and NBA can reuse it.

## Architecture

**Approach: flag-and-poll** (same proven pattern as celebration tweets / NRFI cards — idempotent, auto-retrying, isolated from the prediction pipeline).

### 1. Notification service — `backend/src/services/notifications/sms.py`

- `send_sms(body: str) -> bool` — POST to Twilio Messages API (`https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json`) with HTTP basic auth using plain `requests`. No Twilio SDK dependency.
- Config via env vars: `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_FROM_NUMBER`, `PICK_ALERT_TO_NUMBER`.
- If any env var is unset → log once and return False (no-op on local dev / misconfigured prod).
- Returns True only on Twilio 2xx.

### 2. Alert job — `run_pick_alerts` in `backend/src/tasks/social_scheduler.py`

- Registered `every(10).minutes` on the existing social scheduler.
- Query: today's (ET) `mlb_prediction_snapshots` where `best_bet_team IS NOT NULL AND best_bet_value_score >= 40 AND sms_alert_sent = FALSE`.
- One SMS per pick; set `sms_alert_sent = TRUE` only when `send_sms` returns True (failures retry next cycle).
- Message format (≤160 chars target):

```
TruLine pick: DET +1.5 (+150) @ LAA, 9:38 PM ET
Score 90 | Edge 23% | Skubal vs Soriano
```

- Runline label `TEAM +1.5 / -1.5`, moneyline `TEAM ML`, totals `O/U <line>` (future-proofing — totals not currently in best_bet).
- Odds displayed American (convert from stored decimal).
- Game time displayed ET.

### 3. Migration

```sql
ALTER TABLE mlb_prediction_snapshots ADD COLUMN sms_alert_sent BOOLEAN DEFAULT FALSE;
UPDATE mlb_prediction_snapshots SET sms_alert_sent = TRUE;  -- backfill: never alert historical rows
```

Applied to prod manually (consistent with existing migration practice). SQLAlchemy model `MLBPredictionSnapshot` gains the column.

## Error handling

- Twilio failure → flag stays FALSE, retried every 10 min; log warning (WARN+ so it shows in Railway logs).
- Missing env vars → job no-ops with a single log line per run.
- Snapshot pipeline is untouched — alerts cannot affect prediction/grading.

## Testing

- Unit tests: message formatter (runline/ML/totals labels, American odds, ET time, length) and the query filter (score threshold, flag, date).
- End-to-end: run the job locally against prod DB with real Twilio creds; verify a text arrives on the founder's phone before deploying.
- Deploy verification: `railway status --json` commitHash (health endpoint stays green on stale containers).

## Config to be provided by founder

- `PICK_ALERT_TO_NUMBER` — founder adds directly to Railway (not shared in repo/chat).
- `TWILIO_ACCOUNT_SID` / `TWILIO_AUTH_TOKEN` / `TWILIO_FROM_NUMBER` — from the existing Twilio account (AfterLine numbers).

## Out of scope

- Morning provisional digest (possible follow-up).
- NBA/NFL wiring (service is reusable; each sport adds its own flag column + job when live).
- Two-way SMS / reply handling.
