# NFL Frontend + Evaluation API (P5) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the NFL vertical's UI — a totals-forward picks/slate page and a performance page, wired to live endpoints — plus the two backend evaluation endpoints the performance page needs. Build now on a branch; deploy at the September go-live.

**Architecture:** Mirror the existing MLB frontend + API. Backend: add `/nfl/evaluation/{summary,daily}` to `api/nfl.py` (aggregate graded `nfl_prediction_snapshots`). Frontend: `lib/nflApi.ts` + `lib/nflLogos.ts` data layer, `components/nfl/NFLGameCard.tsx`, `pages/NFLPicks.tsx` + `pages/NFLEvaluation.tsx`, wired into `Layout.tsx` nav + `App.tsx` routes. Totals-forward: best_bet (totals) is the headline; spread + ML are greyed SHADOW everywhere.

**Tech Stack:** Backend — FastAPI, SQLAlchemy 2.0 async, pytest. Frontend — React 18 + TypeScript + Vite, `@tanstack/react-query`, axios, Tailwind.

**Spec:** `backend/docs/superpowers/specs/2026-07-23-nfl-frontend-design.md`.

## Global Constraints

- **Totals-forward:** best_bet (totals) is the headline metric on picks AND evaluation; spread + ML are always rendered as clearly-secondary SHADOW (tracked, not bet) — never the primary number. Matches gating (`nfl_spread_in_best_bet=False`).
- **Mirror existing patterns:** copy the structure of the named MLB file for each frontend piece (`mlbApi.ts`, `mlbLogos.ts`, `MLBGameCard.tsx`, `MLBPicks.tsx`, `MLBEvaluation.tsx`, `Layout.tsx`, `App.tsx`); change mlb→nfl and adapt to the NFL API shapes below. Do NOT invent new conventions.
- **Design tokens (verbatim from the approved mockup / existing cards):** page `#0a0e17`, card `#191c22`, sub-card `#0b0e14`, muted `#32353c`, border `#1e293b`; accent sky-cyan `#a4e6ff` (+ `#00d1ff`), strong green `#66f796`, amber `#f59e0b`, loss red `#ef4444`; mono numerics. Value-score badge tiers: **≥70 strong (green), ≥60 moderate (cyan), else low**; **≥65** → left-edge glow.
- **No deploy in this build.** Commit locally on branch `nfl-frontend`. Do NOT `vercel --prod`, do NOT push. (Deploy is the Sept go-live, separate.)
- **Empty states are required**, not optional: no live NFL data exists until ~Sept, so every page renders a designed empty/"NFL season starts in September" state, never a perpetual spinner.
- **Git:** stage specific files only (never `git add -A`/`.`). Commit trailer `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. Do NOT push.
- **DB (backend test only):** unit tests mock the session — no prod DB needed. If a smoke read is wanted: `export DATABASE_URL=$(grep -oE "postgresql://[^\"']+" src/tasks/prediction_tracker.py | head -1)`, mask passwords.

## NFL API response shapes (already live in `api/nfl.py` — the frontend types must match these exactly)

- `GET /nfl/picks?min_value_score=&limit=` → `{picks: NFLPick[], total, min_value_score}`; `NFLPick = {game_id, home_team, away_team, kickoff_utc, best_bet_type, best_bet_team, best_bet_line, best_bet_odds, best_bet_value_score, best_bet_edge, predicted_margin, predicted_total}` (all `*` nullable).
- `GET /nfl/games?season=&week=` → `{games: NFLGameSummary[], total}`; `NFLGameSummary = {game_id, season, week, home_team, away_team, kickoff_utc, is_divisional, is_primetime, best_bet_type, best_bet_team, best_bet_line, best_bet_value_score}`.

## Canonical NFL team abbreviations (32) — our codebase uses these keys
`ARI ATL BAL BUF CAR CHI CIN CLE DAL DEN DET GB HOU IND JAX KC LA LAC LV MIA MIN NE NO NYG NYJ PHI PIT SEA SF TB TEN WAS`
(`LA` = Rams, `WAS` = Washington, `LV` = Raiders, `LAC` = Chargers, `JAX` = Jaguars.)

## File Structure

**Create:** `backend/tests/unit/test_nfl_api_evaluation.py`; `frontend/src/lib/nflApi.ts`; `frontend/src/lib/nflLogos.ts`; `frontend/src/components/nfl/NFLGameCard.tsx`; `frontend/src/pages/NFLPicks.tsx`; `frontend/src/pages/NFLEvaluation.tsx`.
**Modify:** `backend/src/api/nfl.py` (2 endpoints + 2 models); `frontend/src/App.tsx` (2 routes); `frontend/src/components/Layout.tsx` (2 nav items).

---

### Task 1: Backend — `/nfl/evaluation/{summary,daily}` endpoints

**Files:**
- Modify: `backend/src/api/nfl.py`
- Test: `backend/tests/unit/test_nfl_api_evaluation.py`

**Interfaces:**
- Produces:
  - `GET /nfl/evaluation/summary` → `NFLEvaluationSummary = {total_predictions, graded, wins, losses, pushes, win_rate, total_profit, by_market}` where `by_market` maps `"best_bet"|"spread"|"ml"` → `{wins, losses, pushes, profit, win_rate, count}`. best_bet = the live totals record (headline); spread/ml = shadow.
  - `GET /nfl/evaluation/daily?days=` (default 30, ge=1 le=60) → `NFLDailyPerformance[]` = `{date, predictions, wins, losses, pushes, win_rate, profit}` for the **best_bet** market, oldest→newest.
- Reads graded `NFLPredictionSnapshot` rows (`best_bet_result`/`best_bet_profit`, `best_spread_result`/`best_spread_profit`, `best_ml_result`/`best_ml_profit`, `best_bet_value_score`, `game_date`). "Graded" = `best_bet_result IS NOT NULL`.

Mirror `api/mlb.py`'s `/evaluation/daily` (lines 428-483) and `/evaluation/summary` (486-576) closely, but replace the MLB `by_value_tier`/`by_type` blocks with the NFL **`by_market`** breakout (best_bet/spread/ml) — that's the NFL-specific shape.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/unit/test_nfl_api_evaluation.py
from contextlib import asynccontextmanager
from datetime import date
from unittest.mock import AsyncMock, MagicMock
from fastapi.testclient import TestClient
from src.api import nfl as nfl_api
from src.config import settings
from src.main import app
from src.models import NFLPredictionSnapshot


def _snap(gid, bb_result, bb_profit, vs, gday, sp_result=None, sp_profit=None):
    return NFLPredictionSnapshot(
        game_id=gid, home_team="KC", away_team="CIN", game_date=gday,
        best_bet_type="total", best_bet_value_score=vs,
        best_bet_result=bb_result, best_bet_profit=bb_profit,
        best_spread_result=sp_result, best_spread_profit=sp_profit,
        best_ml_result=None, best_ml_profit=None,
    )


def _scalars(items):
    res = MagicMock(); res.scalars.return_value.all.return_value = list(items); return res


def _patch(monkeypatch, rows):
    session = MagicMock(); session.execute = AsyncMock(return_value=_scalars(rows))
    @asynccontextmanager
    async def _factory():
        yield session
    monkeypatch.setattr(nfl_api, "async_session", _factory)


def test_evaluation_summary_aggregates_best_bet_and_shadow(monkeypatch):
    rows = [
        _snap("g1", "win", 90.9, 55.0, date(2026, 9, 13), sp_result="loss", sp_profit=-100.0),
        _snap("g2", "loss", -100.0, 48.0, date(2026, 9, 13), sp_result="win", sp_profit=90.9),
        _snap("g3", "push", 0.0, 44.0, date(2026, 9, 14)),
    ]
    _patch(monkeypatch, rows)
    body = TestClient(app).get(f"{settings.api_v1_prefix}/nfl/evaluation/summary").json()
    assert body["total_predictions"] == 3 and body["wins"] == 1 and body["losses"] == 1 and body["pushes"] == 1
    assert round(body["total_profit"], 1) == -9.1           # 90.9 - 100 + 0
    assert body["by_market"]["best_bet"]["wins"] == 1
    assert body["by_market"]["spread"]["wins"] == 1 and body["by_market"]["spread"]["losses"] == 1


def test_evaluation_daily_groups_by_date(monkeypatch):
    rows = [
        _snap("g1", "win", 90.9, 55.0, date(2026, 9, 13)),
        _snap("g2", "loss", -100.0, 48.0, date(2026, 9, 13)),
        _snap("g3", "win", 90.9, 60.0, date(2026, 9, 14)),
    ]
    _patch(monkeypatch, rows)
    body = TestClient(app).get(f"{settings.api_v1_prefix}/nfl/evaluation/daily?days=30").json()
    assert len(body) == 2
    d0 = next(d for d in body if d["date"] == "2026-09-13")
    assert d0["predictions"] == 2 and d0["wins"] == 1 and d0["losses"] == 1


def test_evaluation_endpoints_in_openapi():
    schema = TestClient(app).get("/openapi.json").json()
    assert f"{settings.api_v1_prefix}/nfl/evaluation/summary" in schema["paths"]
    assert f"{settings.api_v1_prefix}/nfl/evaluation/daily" in schema["paths"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python3 -m pytest tests/unit/test_nfl_api_evaluation.py -q`
Expected: FAIL (endpoints 404 / not in schema).

- [ ] **Step 3: Implement** — add to `backend/src/api/nfl.py` (after the existing `/debug/odds` route; import `date, timedelta` from datetime at top if not present):

```python
class NFLDailyPerformance(BaseModel):
    date: str
    predictions: int
    wins: int
    losses: int
    pushes: int
    win_rate: float | None
    profit: float


class NFLEvaluationSummary(BaseModel):
    total_predictions: int
    graded: int
    wins: int
    losses: int
    pushes: int
    win_rate: float | None
    total_profit: float
    by_market: dict


def _tally(result: str | None, profit, acc: dict) -> None:
    if result == "win":
        acc["wins"] += 1
    elif result == "loss":
        acc["losses"] += 1
    elif result is not None:
        acc["pushes"] += 1
    acc["profit"] += float(profit or 0)


def _finish(acc: dict) -> dict:
    decided = acc["wins"] + acc["losses"]
    acc["win_rate"] = round(acc["wins"] / decided, 3) if decided else None
    acc["count"] = acc["wins"] + acc["losses"] + acc["pushes"]
    acc["profit"] = round(acc["profit"], 2)
    return acc


@router.get("/evaluation/summary", response_model=NFLEvaluationSummary)
async def get_evaluation_summary() -> NFLEvaluationSummary:
    """Graded best_bet (totals, LIVE) record + spread/ML as SHADOW."""
    async with async_session() as session:
        rows = (await session.execute(
            select(NFLPredictionSnapshot).where(NFLPredictionSnapshot.best_bet_result.isnot(None))
        )).scalars().all()

    markets = {m: {"wins": 0, "losses": 0, "pushes": 0, "profit": 0.0}
               for m in ("best_bet", "spread", "ml")}
    for s in rows:
        _tally(s.best_bet_result, s.best_bet_profit, markets["best_bet"])
        _tally(s.best_spread_result, s.best_spread_profit, markets["spread"])
        _tally(s.best_ml_result, s.best_ml_profit, markets["ml"])
    bb = markets["best_bet"]
    decided = bb["wins"] + bb["losses"]
    return NFLEvaluationSummary(
        total_predictions=len(rows), graded=len(rows),
        wins=bb["wins"], losses=bb["losses"], pushes=bb["pushes"],
        win_rate=round(bb["wins"] / decided, 3) if decided else None,
        total_profit=round(bb["profit"], 2),
        by_market={m: _finish(a) for m, a in markets.items()},
    )


@router.get("/evaluation/daily", response_model=list[NFLDailyPerformance])
async def get_daily_evaluation(
    days: int = Query(30, ge=1, le=60, description="Days to include"),
) -> list[NFLDailyPerformance]:
    """Per-day best_bet (totals) performance, oldest first."""
    async with async_session() as session:
        start = date.today() - timedelta(days=days)
        rows = (await session.execute(
            select(NFLPredictionSnapshot).where(
                NFLPredictionSnapshot.game_date >= start,
                NFLPredictionSnapshot.best_bet_result.isnot(None),
            ).order_by(NFLPredictionSnapshot.game_date)
        )).scalars().all()

    by_date: dict[str, dict] = {}
    for s in rows:
        d = s.game_date.isoformat() if s.game_date else "unknown"
        acc = by_date.setdefault(d, {"wins": 0, "losses": 0, "pushes": 0, "profit": 0.0})
        _tally(s.best_bet_result, s.best_bet_profit, acc)
    out = []
    for d, a in sorted(by_date.items()):
        decided = a["wins"] + a["losses"]
        out.append(NFLDailyPerformance(
            date=d, predictions=a["wins"] + a["losses"] + a["pushes"],
            wins=a["wins"], losses=a["losses"], pushes=a["pushes"],
            win_rate=round(a["wins"] / decided, 3) if decided else None,
            profit=round(a["profit"], 2)))
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && python3 -m pytest tests/unit/test_nfl_api_evaluation.py -q && python3 -c "import src.main"`
Expected: PASS (3 tests) + clean import.

- [ ] **Step 5: Full NFL slice green + commit**

Run: `python3 -m pytest tests/unit/ -k nfl -q`
```bash
git add backend/src/api/nfl.py backend/tests/unit/test_nfl_api_evaluation.py
git commit -m "feat(nfl): /nfl/evaluation/{summary,daily} endpoints (best_bet live + spread/ML shadow)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Frontend data layer — `nflApi.ts` + `nflLogos.ts`

**Files:**
- Create: `frontend/src/lib/nflApi.ts`, `frontend/src/lib/nflLogos.ts`

**Interfaces:**
- Produces (`nflApi.ts`): TS interfaces `NFLPick`, `NFLPicksResponse`, `NFLGameSummary`, `NFLGamesResponse`, `NFLDailyPerformance`, `NFLEvaluationSummary` (match the API shapes above + Task 1); a `nflApi` object with `getPicks(minValueScore=40, limit=20)`, `getGames(season?, week?)`, `getDailyEvaluation(days=30)`, `getEvaluationSummary()`; `NFL_TEAMS` colour map + `getTeamInfo(abbr)` + `formatOdds(decimal)`.
- Produces (`nflLogos.ts`): `getTeamLogo(abbr) -> string` (ESPN CDN URL).

- [ ] **Step 1: Write `nflApi.ts`** — copy `frontend/src/lib/mlbApi.ts` lines 1-19 (axios `client` + auth interceptor) verbatim, then:

```ts
export interface NFLPick {
  game_id: string; home_team: string; away_team: string; kickoff_utc: string | null;
  best_bet_type: string | null; best_bet_team: string | null; best_bet_line: number | null;
  best_bet_odds: number | null; best_bet_value_score: number | null; best_bet_edge: number | null;
  predicted_margin: number | null; predicted_total: number | null;
}
export interface NFLPicksResponse { picks: NFLPick[]; total: number; min_value_score: number; }

export interface NFLGameSummary {
  game_id: string; season: number; week: number; home_team: string; away_team: string;
  kickoff_utc: string | null; is_divisional: boolean | null; is_primetime: boolean | null;
  best_bet_type: string | null; best_bet_team: string | null;
  best_bet_line: number | null; best_bet_value_score: number | null;
}
export interface NFLGamesResponse { games: NFLGameSummary[]; total: number; }

export interface NFLDailyPerformance {
  date: string; predictions: number; wins: number; losses: number; pushes: number;
  win_rate: number | null; profit: number;
}
export interface NFLMarketRecord {
  wins: number; losses: number; pushes: number; profit: number; win_rate: number | null; count: number;
}
export interface NFLEvaluationSummary {
  total_predictions: number; graded: number; wins: number; losses: number; pushes: number;
  win_rate: number | null; total_profit: number;
  by_market: Record<'best_bet' | 'spread' | 'ml', NFLMarketRecord>;
}

export const nflApi = {
  async getPicks(minValueScore = 40, limit = 20): Promise<NFLPicksResponse> {
    const r = await client.get<NFLPicksResponse>(`/nfl/picks?min_value_score=${minValueScore}&limit=${limit}`);
    return r.data;
  },
  async getGames(season?: number, week?: number): Promise<NFLGamesResponse> {
    const p = new URLSearchParams();
    if (season != null) p.set('season', String(season));
    if (week != null) p.set('week', String(week));
    const q = p.toString();
    const r = await client.get<NFLGamesResponse>(`/nfl/games${q ? `?${q}` : ''}`);
    return r.data;
  },
  async getDailyEvaluation(days = 30): Promise<NFLDailyPerformance[]> {
    const r = await client.get<NFLDailyPerformance[]>(`/nfl/evaluation/daily?days=${days}`);
    return r.data;
  },
  async getEvaluationSummary(): Promise<NFLEvaluationSummary> {
    const r = await client.get<NFLEvaluationSummary>('/nfl/evaluation/summary');
    return r.data;
  },
};

// Team colours (primary, secondary). Abbrs = our canonical NFL keys.
export const NFL_TEAMS: Record<string, { name: string; primary: string; secondary: string }> = {
  ARI:{name:'Cardinals',primary:'#97233F',secondary:'#000000'}, ATL:{name:'Falcons',primary:'#A71930',secondary:'#000000'},
  BAL:{name:'Ravens',primary:'#241773',secondary:'#9E7C0C'}, BUF:{name:'Bills',primary:'#00338D',secondary:'#C60C30'},
  CAR:{name:'Panthers',primary:'#0085CA',secondary:'#101820'}, CHI:{name:'Bears',primary:'#0B162A',secondary:'#C83803'},
  CIN:{name:'Bengals',primary:'#FB4F14',secondary:'#000000'}, CLE:{name:'Browns',primary:'#311D00',secondary:'#FF3C00'},
  DAL:{name:'Cowboys',primary:'#003594',secondary:'#869397'}, DEN:{name:'Broncos',primary:'#FB4F14',secondary:'#002244'},
  DET:{name:'Lions',primary:'#0076B6',secondary:'#B0B7BC'}, GB:{name:'Packers',primary:'#203731',secondary:'#FFB81C'},
  HOU:{name:'Texans',primary:'#03202F',secondary:'#A71930'}, IND:{name:'Colts',primary:'#002C5F',secondary:'#A2AAAD'},
  JAX:{name:'Jaguars',primary:'#101820',secondary:'#D7A22A'}, KC:{name:'Chiefs',primary:'#E31837',secondary:'#FFB81C'},
  LA:{name:'Rams',primary:'#003594',secondary:'#FFA300'}, LAC:{name:'Chargers',primary:'#0080C6',secondary:'#FFC20E'},
  LV:{name:'Raiders',primary:'#000000',secondary:'#A5ACAF'}, MIA:{name:'Dolphins',primary:'#008E97',secondary:'#FC4C02'},
  MIN:{name:'Vikings',primary:'#4F2683',secondary:'#FFC62F'}, NE:{name:'Patriots',primary:'#002244',secondary:'#C60C30'},
  NO:{name:'Saints',primary:'#D3BC8D',secondary:'#101820'}, NYG:{name:'Giants',primary:'#0B2265',secondary:'#A71930'},
  NYJ:{name:'Jets',primary:'#125740',secondary:'#000000'}, PHI:{name:'Eagles',primary:'#004C54',secondary:'#A5ACAF'},
  PIT:{name:'Steelers',primary:'#FFB612',secondary:'#101820'}, SEA:{name:'Seahawks',primary:'#002244',secondary:'#69BE28'},
  SF:{name:'49ers',primary:'#AA0000',secondary:'#B3995D'}, TB:{name:'Buccaneers',primary:'#D50A0A',secondary:'#34302B'},
  TEN:{name:'Titans',primary:'#0C2340',secondary:'#4B92DB'}, WAS:{name:'Commanders',primary:'#5A1414',secondary:'#FFB612'},
};
export function getTeamInfo(abbr: string) {
  return NFL_TEAMS[abbr] || { name: abbr, primary: '#333', secondary: '#777' };
}
export function formatOdds(decimal: number): string {
  return decimal >= 2.0 ? `+${Math.round((decimal - 1) * 100)}` : `${Math.round(-100 / (decimal - 1))}`;
}
```

- [ ] **Step 2: Write `nflLogos.ts`** — mirror `frontend/src/lib/mlbLogos.ts`'s pattern (ESPN CDN, our-abbr→espn-abbr map). ESPN NFL uses lowercase of our abbr EXCEPT `LA→'lar'` and `WAS→'wsh'`:

```ts
/** NFL team logos via ESPN CDN: https://a.espncdn.com/i/teamlogos/nfl/500/{abbr}.png */
const NFL_ESPN_ABBR: Record<string, string> = {
  ARI:'ari',ATL:'atl',BAL:'bal',BUF:'buf',CAR:'car',CHI:'chi',CIN:'cin',CLE:'cle',DAL:'dal',DEN:'den',
  DET:'det',GB:'gb',HOU:'hou',IND:'ind',JAX:'jax',KC:'kc',LA:'lar',LAC:'lac',LV:'lv',MIA:'mia',
  MIN:'min',NE:'ne',NO:'no',NYG:'nyg',NYJ:'nyj',PHI:'phi',PIT:'pit',SEA:'sea',SF:'sf',TB:'tb',TEN:'ten',WAS:'wsh',
};
export function getTeamLogo(abbr: string): string {
  const e = NFL_ESPN_ABBR[abbr] || abbr.toLowerCase();
  return `https://a.espncdn.com/i/teamlogos/nfl/500/${e}.png`;
}
```

- [ ] **Step 3: Type-check** — Run: `cd frontend && npx tsc --noEmit` (expect no new errors from these files). If the repo has no `tsc` script, run `npm run build` and confirm it compiles.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/lib/nflApi.ts frontend/src/lib/nflLogos.ts
git commit -m "feat(nfl-fe): nflApi client + nflLogos (ESPN CDN)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: `NFLGameCard.tsx` — the totals-forward card

**Files:**
- Create: `frontend/src/components/nfl/NFLGameCard.tsx`

**Interfaces:**
- Consumes: `NFLGameSummary` (and optionally the matching `NFLPick` fields) from `nflApi.ts`; `getTeamLogo` from `nflLogos.ts`; `getTeamInfo`, `formatOdds` from `nflApi.ts`.
- Produces: `export function NFLGameCard({ game }: { game: NFLGameSummary })`.

The component is a React/TSX port of the approved mockup card (`scratchpad/nfl-ui-mockup.html`). Read `frontend/src/components/mlb/MLBGameCard.tsx` for the project's card idioms (Tailwind arbitrary-value hex classes, logo `<img>` with `onError`, badge tiering) and mirror them. Requirements:
- **Header:** `AWY @ HOM` label + kickoff time; pills for `is_primetime` (amber "PRIME"), `is_divisional` (slate "DIV").
- **Teams:** for each team a logo `<img src={getTeamLogo(abbr)}>` with `onError` swapping to a two-tone monogram crest (a `<span>` with `background: linear-gradient(135deg, primary 50%, secondary 50%)` + the abbr), name from `getTeamInfo`.
- **Value badge (round):** `best_bet_value_score` with tier classes — `>=70` green `#66f796`, `>=60` cyan `#a4e6ff`, else slate; label STRONG/MODERATE/LOW. `>=65` adds a left-edge glow bar on the card.
- **Best-bet row (highlighted):** when `best_bet_type === 'total'` and `best_bet_value_score != null`, show `OVER/UNDER {best_bet_line}` (direction from `best_bet_team`) with the value score; cyan/green highlight. When no qualifying best_bet, a muted "No value pick" line.
- **SHADOW strip:** a greyed bottom strip labeled "SHADOW" noting spread & ML are tracked, not bet (the game summary doesn't carry spread/ML odds, so this is a static contextual strip — do NOT fabricate lines).
- Use the design tokens from Global Constraints. Card: `rounded-xl bg-[#191c22] border border-[#1e293b] hover:border-[#a4e6ff]/30`.

- [ ] **Step 1: Implement the component** per the above, mirroring MLBGameCard's Tailwind idioms. (No unit-test harness for cards in this repo; the gate is type-check + build + the page render in Task 4.)
- [ ] **Step 2: Type-check** — `cd frontend && npx tsc --noEmit` (or `npm run build`); expect clean.
- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/nfl/NFLGameCard.tsx
git commit -m "feat(nfl-fe): NFLGameCard (totals-forward, real logos + crest fallback)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: `NFLPicks.tsx` page + nav + route

**Files:**
- Create: `frontend/src/pages/NFLPicks.tsx`
- Modify: `frontend/src/App.tsx`, `frontend/src/components/Layout.tsx`

**Interfaces:**
- Consumes: `nflApi.getPicks`, `nflApi.getGames`, `NFLGameCard`.
- Produces: `export function NFLPicks()` (route `/nfl`).

Mirror `frontend/src/pages/MLBPicks.tsx`'s structure (React Query via `@tanstack/react-query`, `useState`/`useMemo`, layout container `max-w-7xl`). Requirements:
- Two React Query calls: `queryKey ['nfl-picks', minValueScore]` → `nflApi.getPicks(minValueScore)`, and `['nfl-games']` → `nflApi.getGames()`.
- **Best Bets / Full Slate** segmented toggle + a `min_value_score` slider (default 40). Best Bets → render games/picks with a qualifying `best_bet` (from `/nfl/picks`); Full Slate → all upcoming games (from `/nfl/games`), each as `NFLGameCard`.
- Grid `grid gap-6 lg:grid-cols-2`. Loading → skeleton cards (`animate-pulse`, mirror MLBPicks). **Empty →** a designed panel: "🏈 NFL best bets return in September" + one line that the model is totals-forward (spread & ML shadow-tracked until they beat the market). Never a bare spinner.
- Header "NFL Best Bets" + a totals-forward one-liner.

- [ ] **Step 1: Implement `NFLPicks.tsx`** per above.
- [ ] **Step 2: Wire route + nav**
  - `App.tsx`: add `import { NFLPicks } from '@/pages/NFLPicks';` and `<Route path="/nfl" element={<NFLPicks />} />`.
  - `Layout.tsx` `navItems`: add `{ path: '/nfl', label: 'NFL', icon: '🏈' }` (after the MLB entries).
- [ ] **Step 3: Build** — `cd frontend && npm run build` (tsc + vite build) must succeed with the new route wired.
- [ ] **Step 4: Commit**

```bash
git add frontend/src/pages/NFLPicks.tsx frontend/src/App.tsx frontend/src/components/Layout.tsx
git commit -m "feat(nfl-fe): NFLPicks page (Best Bets/Full Slate) + NFL nav/route

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: `NFLEvaluation.tsx` page + nav + route

**Files:**
- Create: `frontend/src/pages/NFLEvaluation.tsx`
- Modify: `frontend/src/App.tsx`, `frontend/src/components/Layout.tsx`

**Interfaces:**
- Consumes: `nflApi.getEvaluationSummary`, `nflApi.getDailyEvaluation`.
- Produces: `export function NFLEvaluation()` (route `/nfl/performance`).

Mirror `frontend/src/pages/MLBEvaluation.tsx`. Requirements:
- React Query: `['nfl-eval-summary']` → `getEvaluationSummary()`, `['nfl-eval-daily', days]` → `getDailyEvaluation(days)`.
- **Summary tiles:** headline the **best_bet (totals)** record — win%, units (`total_profit`/100 or profit as-is per MLB's unit convention; match MLBEvaluation's display), W-L-P. Then **spread** and **ml** as clearly-secondary SHADOW tiles (smaller/greyed, labeled "shadow — tracked, not bet") from `by_market`.
- **Daily** table or simple bar (mirror MLBEvaluation) from `getDailyEvaluation`.
- **Empty state** when `graded === 0`: "No graded NFL games yet — results appear once the season starts." Never a bare spinner.

- [ ] **Step 1: Implement `NFLEvaluation.tsx`** per above.
- [ ] **Step 2: Wire route + nav**
  - `App.tsx`: `import { NFLEvaluation } from '@/pages/NFLEvaluation';` + `<Route path="/nfl/performance" element={<NFLEvaluation />} />`.
  - `Layout.tsx` `navItems`: add `{ path: '/nfl/performance', label: 'NFL Results' }`.
- [ ] **Step 3: Build** — `cd frontend && npm run build` must succeed.
- [ ] **Step 4: Commit**

```bash
git add frontend/src/pages/NFLEvaluation.tsx frontend/src/App.tsx frontend/src/components/Layout.tsx
git commit -m "feat(nfl-fe): NFLEvaluation page (totals headline, spread/ML shadow) + nav/route

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Phase P5 Exit Criteria

1. `pytest tests/unit -k nfl` green (adds the evaluation-endpoint tests); `/nfl/evaluation/{summary,daily}` in the OpenAPI schema; `import src.main` clean.
2. `cd frontend && npm run build` succeeds with all NFL files + the two routes wired.
3. NFL nav tabs (`🏈 NFL`, `NFL Results`) present in `Layout.tsx` (desktop + mobile rows); routes `/nfl` and `/nfl/performance` resolve.
4. Totals-forward honored: best_bet is the headline on both pages; spread/ML render only as labeled SHADOW.
5. Every page renders a designed empty/"season starts September" state (no perpetual spinner) — since there is no live NFL data until go-live.
6. **NOT deployed, NOT pushed.** Committed on `nfl-frontend`. Deploy (`vercel --prod` + backend push) is the deliberate September go-live, out of scope here.
