# TruLine Engineering Documentation

The operating manual for TruLine as it grows from a model into an analytics platform.

Written 2026-07-31, grounded in the system as it actually exists — every claim here was verified against code or the production database, not aspirational.

---

## The documents

| # | Document | Answers |
|---|---|---|
| 01 | [System Architecture](01-system-architecture.md) | What runs where, what talks to what, what state is durable |
| 02 | [Feature Engineering Playbook](02-feature-engineering-playbook.md) | How a feature is proposed, validated, versioned, retired |
| 03 | [Model Governance Handbook](03-model-governance.md) | How models and constants are versioned, promoted, rolled back |
| 04 | [Research & Experimentation](04-research-experimentation.md) | How a hypothesis becomes evidence, and evidence becomes a change |
| 05 | [API & Data Contracts](05-api-data-contracts.md) | What each component publishes and consumes, and which guarantees hold |
| 06 | [Roadmap 2026 H2](06-roadmap-2026H2.md) | What we build next, in what order, and why |

Read 06 first for direction, then 03 and 04 — they govern decisions being made right now. 01 and 05 are reference. 02 sits between.

---

## The five things a new reader most needs to know

1. **`mlb_prediction_snapshots` is the only source of truth for what was bet.** `mlb_markets` is overwritten in place and has no history. Grading from it is always a bug.

2. **The live MLB run-diff model was trained 2026-02-09 and has never been retrained.** Six months stale, mid-season. See `03-model-governance.md` §1.

3. **A market is disabled by config, not by deletion.** Runline and totals are still scored and stored while excluded from `best_bet`, which is what makes evidence-based re-entry possible.

4. **P&L over a few hundred bets is weak evidence.** The moneyline record is +1.18 standard errors from zero. Calibration studies and closing-line value are the primary instruments; P&L is lagging confirmation. See `04-research-experimentation.md` §1.

5. **Pushing `main` is a production deploy.** There is no staging environment. The frontend does *not* auto-deploy and needs a manual `vercel --prod`.

---

## Consolidated open items

Sequenced in [06-roadmap-2026H2.md](06-roadmap-2026H2.md). Listed here with source docs.

| # | Item | Doc |
|---|---|---|
| 1 | **Build closing-line-value capture.** Resolves in ~50 bets what win rate needs thousands for | 04 §5 |
| 2 | **Retrain the MLB run-diff model** — and fix the feature-vector contract in the same change | 03 §9, 02 §2 |
| 3 | **Wire up already-collected features** — `wind_factor` (never computed), `temperature`, `is_dome`, `last_10` | 02 §6 |
| 4 | Expose model version + `trained_at` on `/health`; alert on fallback engagement | 03 §7 |
| 5 | Distinguish "no edge" from "no data" in ingest failures | 01 §6 |
| 6 | Move `DEFAULT_RUN_DIFF_MODEL` into config for rollback parity | 03 §9 |
| 7 | Process stats (FIP/xERA) over outcome stats (ERA/AVG) | 02 §6 |
| 8 | Bullpen features — ~40% of innings, currently invisible to the model | 02 §6 |
| 9 | Retain candidate odds sets so selection changes become testable | 04 §8 |
| 10 | Runline empirical cover curve; stays paused until forward-validated | 03 §6 |

---

## Maintenance

- Each document carries a **"Last verified against code"** date. Update it when you check, not when you edit.
- Update a document in the **same PR** as the change it describes.
- `03-model-governance.md` §8 is an append-only change log — add a row for every model or constant change that reaches production.
- These documents describe the system as it **is**, including its defects. Aspirations belong in `docs/superpowers/specs/`.
