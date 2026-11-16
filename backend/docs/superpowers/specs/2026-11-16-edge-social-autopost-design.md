# TruLine Edge social autopost: blind plays, cheat sheets, value plays, receipts, final recaps

Date: 2026-11-16. Owner: Samir. Status: design, awaiting approval.

## 1. Why

TruLine costs real money each month and earns nothing. The product works: NBA Edge averages +1.5% CLV and
beats the close 78% of the time. Nobody sees it. On 11-15..16 the API logs show 2-4 human visitors in 41 hours.
@trulineapp has 116 followers and has been silent since MLB ended on 10-31. Its last posts were text-only and
drew 13-96 views each.

The fix Samir chose is social first: get the X account posting again, automatically, so it drives people to
truline.app, where the email signup (shipped 11-16, `POST /edge/signup`) and Vercel Web Analytics are waiting.
Paid tiers (Stripe) are on hold until traffic exists.

Success: real visitors arriving from X (Vercel referrer `x.com`/`t.co`), email signups, and progress toward
Samir's October goal of **1,000 X followers**. Followers should come away saying the account gives
**"trusted insights, value plays, winning plays."**

## 2. What Samir decided

- **Platform:** X first. TikTok later (out of scope here).
- **Fully automatic.** No approval step, no daily post cap (October decision, unchanged).
- **Never** post a wrong score, or a pick after its game has started or after it has lost.
- **Style:** branded image cards with real numbers, modeled on @MySpariEdge (cheat sheets from their own data),
  @ItsParlayPete (polished daily picks cards; a "10-0 SWEEP" receipt was their top post at 74K views) and
  @UpsideTools (closest product, weakest engagement: avoid bare tool screenshots).
- **The hook: Blind Plays**, Samir's BlindDFS idea, run **under the TruLine name** (not a separate brand).
  A top play is shown with the player and team hidden, to take fan bias out of the bet. People click
  through to truline.app to reveal it.
- **Blind card content** (mockup approved, "version 2"): position + market, side + line + price, edge, books
  priced, last-10 hits, last-10 average, and a **matchup panel**: opponent vs position, opponent defensive
  rating rank, opponent pace, the player's usage rate (L10, with change), recent minutes. A "FAVORABLE" chip.
- **Only post blind plays whose matchup is favorable** ("if we keep posting bad matchups and losses we lose
  credibility"). Value alone is not enough.
- **Real ESPN team logos and player headshots** on every card except where they would give a blind play away.
- **Post types per day, NBA:** Cheat Sheet, Value Plays, Blind Plays, Final Recap per game, Receipt on wins.
- **NFL:** Cheat Sheets, Final Recaps and Blind Plays. Samir's explicit decision on 2026-11-16 to post NFL
  blind plays, although NFL Edge is tracking-only. Record it in the **Operational log** of
  `docs/engineering/07-nfl-promotion-gates.md`, **not** §7 Amendments: those gates cover the NFL game model,
  not Edge props, and an amendment written mid-window would void this season's pre-registered evaluation.
- **Receipts only when we win.** Losses still appear in Final Recaps and on the results page.
- **Final recaps** modeled on the MLB final card Samir liked (status 2096855166045589837: logos, score,
  records, our pick marked W/L, L10).

## 3. Goals, non-goals, anti-goals

Goals
1. Five NBA post types and three NFL post types, rendered and posted automatically on schedule.
2. A public reveal page per blind play on truline.app that shows the full breakdown and the email signup.
3. A favorable-matchup filter for blind plays, validated on this season's graded plays before launch.
4. Matchup stats (opponent vs position, defensive rating rank, pace, usage, minutes) computed from data we
   already pay for.

Non-goals
- TikTok, video, or the Remotion pipeline.
- Changing how the Edge board picks or prices bets.
- Paid tiers, Stripe, gating.
- LLM-written copy. Copy comes from templates over real numbers (keeps "never wrong" auditable).
- Engagement-only posts (polls, "who's ready for SNF?"). Maybe later.

Anti-goals (a failure even if it technically works)
- Any post with a wrong score, wrong result, wrong player, or a pick posted after tip.
- A blind card that names or shows the player or team (logo, headshot, team colors, matchup string).
- Bragging posts that misstate the record: no "+805 units" (that figure counts the same bet many times),
  no cherry-picked win rates. Receipts say what happened to that one play, nothing more.
- Legacy MLB or legacy NBA model posts going out by accident when posting is switched on.
- A runaway loop posting the same card many times.
- Leaking secrets or the cron key in a post, a log, or a reveal URL.

## 4. Current state (what exists)

- **Posting:** Blotato (`src/services/social/blotato.py`: `upload_media`, `post_tweet`, `post_thread`), account
  id in `settings.blotato_twitter_account_id`, master switch `settings.twitter_posting_enabled`
  (`TWITTER_POSTING_ENABLED`). Typefully client exists but is the old path.
- **Scheduler:** `src/tasks/social_scheduler.py`, a `schedule` loop run as a daemon thread inside the
  `nba-value` API process (`src/main.py` `_start_social_scheduler_thread`). It still schedules MLB posts
  (picks, NRFI, first-inning, finals, celebrations, founder SMS alerts) and legacy NBA model posts (gated by
  `nba_posting_allowed()` → `settings.nba_picks_public`, currently false). MLB jobs are idle only because
  there are no MLB games; they would fire again in spring.
- **Images:** `src/services/social/image_generator.py` (PIL). `generate_final_card` drew the MLB final card;
  team logos come from `https://a.espncdn.com/i/teamlogos/{sport}/500/{abbr}.png`.
- **Edge data:** `edge_opportunities` (every +EV row, graded, CLV), `edge_nightly_legs` + `src/edge/nightly.py`
  (Tonight's Top 5 locked an hour before first tip, `GET /edge/nightly`), `edge_player_games` (one box-score line
  per player per game: minutes, pts, reb, ast, fg3m, stl, blk, tov; Final games only), `edge_players` (odds name
  → BallDontLie id, team). NFL: `edge_nfl_*` tables, `src/edge/matchups.py` + `tagging.py` (matchup tags; only
  the `def_soft` tailwind is proven).
- **Frontend:** React/Vite on Vercel; Edge pages under `/edge`; `EmailSignup` component (source enum in
  `src/edge/signups.py` `SOURCES`, add new sources there).
- **Evidence on matchup stats:** NFL: only `def_soft` held up. NBA: untested; recent form helped only as a
  warning (≥15 pts below market lost money).

## 5. The posts

All cards: 1080×1350 PNG, dark TruLine style (see approved mockups in
`.superpowers/brainstorm/71005-1794854652/content/post-lineup-v2.html`, not committed), `truline.app` +
`@trulineapp` footer. Times are ET and must handle the EST/EDT switch (the old scheduler hard-codes UTC for
EDT; don't repeat that).

### 5.1 NBA

| Post | When | Content |
|---|---|---|
| Cheat Sheet | 11:00 ET on days with games | Tonight's props with the best last-10 hit rates **among props that are +EV on the board right now** (min 6 rows, max 10), each with headshot, line, L10 hits, market fair %. One card, two if long. Text: date + one line. If fewer than 6 qualify, skip the post. *(My call: hit rate alone would post props the market prices against us.)* Cheat Sheet rows are **not** "posted plays": they don't appear in Final Recaps or Receipts. |
| Value Plays | When `nightly` locks (an hour before first tip) | Tonight's Top 5 from `edge_nightly_legs`, best book + price + edge per row, headshots. Same plays the site shows, at their **locked** prices ("locked 6:10 PM ET"). *(My call:)* Value Plays are exempt from the 60-minute buffer and the live-EV check in §9; they only need every row's own tip to still be in the future. If the lock happened late and any row's game has started, skip the whole post. |
| Blind Plays | Staggered, each ≥ 60 min before its own tip | 1-3 per night (see §6). Thread: post 1 = card + short text ("No names. No teams. No bias. Who is it? 👇"); post 2 (reply) = reveal link. |
| Final Recap | After each game is Final, only games where TruLine had a posted play | Logos, score, team records, every TruLine *posted* play in that game (blind + value) with W/L, units for the game. |
| Receipt | After a posted blind play or Value Play wins | Headshot, name, team logo, bet → stat line, price, units, "CASHED". Must link the original post (quote) where the posting API allows; otherwise reply to it. Blind receipts say "Blind Play #N was <Name>". |

### 5.2 NFL

| Post | When | Content |
|---|---|---|
| Cheat Sheet | Thursday, Sunday 10:00 ET and Monday morning for MNF | Props by hit rate + matchup tags (`def_soft` etc.), headshots. |
| Blind Plays | Before kickoff (≥ 60 min) | Same card as NBA with NFL fields (see §7). Matchup must include a proven tailwind (`def_soft`). |
| Final Recap | After each game with a posted play | Same layout as NBA. |

Receipts for NFL blind plays follow the NBA rule (wins only).

## 6. Blind Plays

**Selection** (per sport, per slate):
1. Start from current Edge opportunities that would qualify for Tonight's Top 5 (same eligibility as
   `nightly.eligible`: 5+ books, not "verify", not cold), props only (player markets).
   **Exclude any player in tonight's Top 5** *(my call)*, so the named Value Plays card never gives a blind play
   away. Blind plays are therefore distinct bets from Value Plays; both count as "posted plays".
2. Keep only **favorable matchups** (§7 defines the score per sport).
3. One per game and per player; rank by EV; take up to 3.
4. Must be ≥ 60 min before tip when posted. If a play's price moves so EV ≤ 0 before posting, drop it.
5. Zero qualifying plays is fine: post nothing. Never relax the filter to fill a slot.

**Numbering:** "Blind Play #N" counts up per sport for the season (stable, shown on the reveal page).

**What may never be on the card:** name, team, opponent, logos, headshot, team colors, matchup string, venue,
exact tip time if only one game tips then (round to "tonight" / "early" / "late" when a tip time would
identify the game). Position + market + line + price + edge + books + L10 + matchup ranks are allowed.
The ranks do narrow it down; that guessing is the game.

**Reveal page** (`truline.app/blind/<sport>/<n>`, public, no login):
- Before tip and after: name, headshot, team + opponent logos, the full card numbers, a one-paragraph plain
  explanation built from the numbers ("9 books price this at 49%; one pays +116, which needs only 46%"), link
  to the odds screen and player page, `EmailSignup` (new source `blind_reveal`).
- After grading: result, stat line, units.
- A list page `truline.app/blind` with all blind plays and the running record (wins and losses).

**Results honesty:** blind-play record (W-L, units) is public on the list page and never shown on X as a
running total unless it is positive *and* n ≥ 30 (my call; avoids bragging on noise).

## 7. Matchup data

### NBA (new, computed from stored box scores)
- **Store more box-score fields:** add `fga`, `fta`, `oreb`, `fgm` (if needed) to `edge_player_games` via
  `EDGE_COLUMN_MIGRATIONS`; BallDontLie `/stats` already returns them. Backfill this season (raise
  `EDGE_BOX_RELOAD_DAYS` briefly, or a one-off script).
- **Positions:** BallDontLie player position (G/F/C, combos like G-F). Map to G, F, C (first letter of a combo,
  my call; builder may refine). Store on `edge_players` or a small table.
- **Opponent vs position:** for each team, average of the stat the prop measures (pts, reb, ast, PRA, 3PM, …)
  allowed to that position per game, this season, last 15 games weighted or plain season (builder's call,
  document it). Rank 1-30 (30 = allows most).
- **Defensive rating rank:** points allowed per 100 possessions, possessions = FGA + 0.44·FTA − OREB + TOV
  from opponent team totals. Rank 1-30.
- **Pace:** possessions per game. Rank 1-30.
- **Usage rate (player):** standard formula from player and team totals; minutes from box score. L10 vs season.
- **Favorable (NBA), initial definition:** for an *over*, opponent rank vs position in the bottom third
  (21-30 allows most) **and** opponent defensive rating rank not top-10; mirrored for *unders*. Minutes L5 not
  down more than 15% vs season. Usage not down more than 3 points.

### NFL (reuse)
- Use `matchups.py` / `tagging.py` tags. **Favorable (NFL)** = the bet has the proven `def_soft` tailwind and no
  headwind tag. Card fields: opponent rank vs position (yards/receptions allowed), snap share, target or carry
  share L4, `def_soft` chip.

### Validation gate (before any blind play posts)
Run the NBA favorable filter over every graded NBA prop opportunity since 2026-10-20 (dedupe to one row per
bet, earliest qualifying sighting). Compare favorable vs not on CLV, win rate and units with 95% CIs.
**Pre-registered launch rule** *(my call, fixed before looking)*: launch only if (a) the favorable group's mean
CLV is ≥ the rest's mean CLV minus 0.25 percentage points, **and** (b) the favorable group's 95% CI for mean
CLV is entirely above 0, **and** (c) n ≥ 200 favorable bets. Win rate and units are reported, not gated (too
noisy at this n). Otherwise stop and bring the numbers to Samir. Don't change the rule after seeing results.
Write the result to `docs/backtests/nba-matchup-filter-2026-11.md`. Same check for NFL on 2026 graded plays,
lighter (sample is small): just confirm `def_soft` still isn't negative.

## 8. Images, faces, logos

- **Logos:** ESPN CDN as today.
- **Headshots:** ESPN `https://a.espncdn.com/i/headshots/{nba|nfl}/players/full/{espn_id}.png`. ESPN ids come
  from the public roster feed `site.api.espn.com/apis/site/v2/sports/{basketball/nba|football/nfl}/teams/{abbr}/roster`
  (gives name, id, position). Match to our players by normalized name + team (reuse the name-matching in
  `players.py`, `NAME_ALIASES`). Cache ids in a table; refresh weekly. Missing headshot → neutral silhouette,
  never a wrong face.
- Logos and headshots are NBA/NFL/ESPN property. Accepted risk (Samir, 11-16); a takedown request means
  switching to silhouettes + team colors, so keep that fallback working.
- Rendering: extend `image_generator.py` (PIL) or move cards to HTML→PNG; builder's call, but cards must match
  the approved mockup closely and render in < 5 s each on Railway.

## 9. Posting, scheduling, safety

- **Where it runs:** builder's call between the API's social thread and the `edge-worker`. Must run exactly
  once (the API may run more than one replica later; the worker holds an advisory lock).
- **Retire legacy jobs:** MLB and legacy NBA model jobs must not fire. Remove them from the scheduler or put
  them behind their own explicit flags defaulting off. Founder SMS pick alerts: off unless Samir asks.
- **Idempotency:** a `social_posts` table (sport, kind, key e.g. `blind:nba:12` / `final:<event_id>`,
  status, blotato submission id, public URL if known, rendered image URL, created_at). Unique on (kind, key).
  Never post a key twice.
- **Safety checks before each post** (skip and log, never post on failure):
  - Blind plays: `now < commence_time − 60 min`; EV still > 0 at current best price.
  - Value Plays: see §5.1 (every row's tip still in the future; locked prices).
  - Final/receipt: game status Final in our data **and** score cross-checked against a second source (BDL game
    + Odds API scores, or ESPN scoreboard); prop result from `edge_opportunities` grading only after the box
    score is Final.
  - Text and image built only from DB values; no free text.
- **Brakes:** max 25 posts per sport per day and max 6 per hour (my numbers; Samir said no cap, these are
  runaway guards, not content limits). Tripping a brake stops posting and logs loudly.
- **Switches:** keep `TWITTER_POSTING_ENABLED` as master. Add per-kind flags (`SOCIAL_BLIND`, `SOCIAL_CHEATSHEET`,
  …) so one can be turned off without a deploy of code. A **dry-run mode** renders and stores cards and text
  without posting, viewable on an admin page, used for the first days.
- **Links:** reveal and site links carry UTM tags (`utm_source=x&utm_medium=social&utm_campaign=<kind>`) so
  Vercel Analytics can attribute traffic.
- **Images to Blotato:** `upload_media` returns a URL. `post_thread` creates a post plus replies in one call
  (used for blind card + reveal-link reply). It returns only a `postSubmissionId`; replying to or quoting an
  *older* post isn't supported by our client today.
- **Receipts without reply/quote** *(my call)*: first check whether Blotato's API exposes the published post
  URL for a submission. If it does, store it in `social_posts` and put the original post's URL in the receipt
  text (X renders it as a quote). If it doesn't, post the receipt standalone with **two images**: the receipt
  card and the original blind card, plus the reveal link. Do not add a direct X API integration for this.

## 10. Rollout

1. Build with dry-run on. Samir reviews a few days of rendered cards on the admin page.
2. Validation gate (§7) passes.
3. Turn on NBA posting. One-week check: replies, clicks (Vercel referrers + UTM), signups, follower change,
   blind vs other post types.
4. NFL on once NBA has run cleanly for a week (or the next NFL Sunday if Samir wants it sooner).

## 11. Left to the builder

Rendering approach, where the scheduler lives, exact NBA rank windows, card copy wording within the templates
above, file layout, admin page shape. Follow `CLAUDE.md` (tests first, the throwaway Postgres for SQL tests,
check both Railway services after pushes, watch patterns under `backend/src/**`).
