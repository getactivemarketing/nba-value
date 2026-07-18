-- Founder SMS pick alerts (spec: docs/superpowers/specs/2026-07-18-sms-pick-alerts-design.md)
-- Applied manually to prod 2026-07-18 (alembic stalled at 004; manual ALTER is current practice).
ALTER TABLE mlb_prediction_snapshots
    ADD COLUMN IF NOT EXISTS sms_alert_sent BOOLEAN NOT NULL DEFAULT FALSE;
-- Backfill: never alert on historical picks.
UPDATE mlb_prediction_snapshots SET sms_alert_sent = TRUE;
