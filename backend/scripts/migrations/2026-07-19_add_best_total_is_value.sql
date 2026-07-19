-- Totals shadow widened to all games (2026-07-19): best_total_* now records
-- every game's best total; this flag preserves the strict gate-passing cut
-- that the re-entry decision and the UI badge use.
ALTER TABLE mlb_prediction_snapshots
    ADD COLUMN IF NOT EXISTS best_total_is_value BOOLEAN;
-- Backfill: every pre-change shadow row was gate-passing by construction.
UPDATE mlb_prediction_snapshots
    SET best_total_is_value = TRUE
    WHERE best_total_direction IS NOT NULL AND best_total_is_value IS NULL;
