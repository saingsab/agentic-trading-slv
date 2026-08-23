# agentic-trading-slv — build plan

A decision-support system for discretionary swing trading in XAG/USD.
It reads, computes, verifies, and proposes. It never executes.

---

## Locked decisions

These are settled. Revisiting them mid-build is the main way this project dies.

| Decision | Choice | Why |
|---|---|---|
| Instrument | XAG/USD only | Depth beats breadth. One instrument you know deeply. |
| Language | Python 3.11+ | Ecosystem, and matches the learning goal. |
| Storage | SQLite, single file | Zero ops, trivially backed up, queryable. |
| Scheduling | cron | Boring and reliable. Not a workflow engine. |
| Agent framework | smolagents | ~1k lines, readable, model-agnostic. Phase 5, not before. |
| Local model | Ollama, 8B class | Extraction and classification only. Never synthesis. |
| Paid API | Anthropic, final prose only | ~$0.50/month at this volume. |
| Execution | Not implemented | Not in v1. Not in v2. |

**Hard scope lock:** no other instruments, no broker integration, no auto-execution,
no fine-tuning, no price prediction models. If a feature isn't on this plan,
it goes in `IDEAS.md` and waits.

---

## Non-goals

- Predicting price
- Beating the market on speed
- Replacing the daily briefing skill (this feeds it)
- Being a product for anyone else, yet

---

## Repo layout

```
agentic-trading-slv/
├── slv/
│   ├── config.py          # paths, API keys from env, constants
│   ├── db.py              # schema, migrations, connection
│   ├── fetch/
│   │   ├── prices.py      # yfinance
│   │   ├── fred.py        # FRED API
│   │   ├── cot.py         # CFTC weekly CSV
│   │   └── calendar.py    # seeded CSV in v1
│   ├── compute/
│   │   ├── indicators.py  # ATR, EMA, RSI, realized vol
│   │   ├── regime.py      # regime state classification
│   │   └── eventrisk.py   # sizing math at leverage
│   ├── journal.py         # append-only theses + process scoring
│   ├── brief.py           # assembles facts object
│   └── cli.py             # command entry points
├── data/
│   ├── slv.db              # SQLite
│   └── calendar_seed.csv  # FOMC/CPI/NFP dates
├── tests/
├── briefs/                # dated markdown output, gitignored (regenerable via `slv brief`)
└── PLAN.md
```

---

## Schema

```sql
-- raw, append-only, idempotent on (date, source)
prices(date PK, open, high, low, close, volume, symbol, source)
macro(date, series_id, value, PRIMARY KEY(date, series_id))
cot(report_date PK, mm_net, mm_long, mm_short, open_interest)
events(date, name, actual, consensus, prior, surprise_z, PRIMARY KEY(date, name))

-- derived, always recomputable from raw
indicators(date PK, atr14, ema20, ema50, sma200, rsi14, rvol20,
           gsr, dist_ema20_atr, range_pctile_60d)

-- the keystone
theses(id PK, created_at, direction, entry_zone, invalidation, target,
       size_pct_equity, leverage, rationale, claim_types,
       provenance, status)
thesis_outcomes(thesis_id PK, closed_at, exit_price, r_multiple,
                process_score, process_detail_json)

briefs(date PK, path, facts_json)
backtest_runs(id PK, spec_hash, params_json, train_metrics_json, created_at)
```

Rule: **raw tables are append-only.** Derived tables can be dropped and rebuilt.
`theses` rows are immutable after creation — corrections create a new row.

---

## Data sources (all free in v1)

| Need | Source | Notes |
|---|---|---|
| Silver / gold OHLCV | `yfinance` — `SI=F`, `GC=F` | Free, adequate for daily |
| DXY | `yfinance` — `DX-Y.NYB` | |
| 10y real yield | FRED `DFII10` | Free API key, official |
| Breakevens | FRED `T10YIE` | |
| Positioning | CFTC COT legacy CSV, weekly | Free download, Friday |
| Event calendar | Hand-seeded CSV | FOMC/CPI/NFP dates published a year ahead |

The seeded calendar is deliberate — it removes a paid dependency for the whole
of v1 and takes twenty minutes to fill in for a year.

---

## Phases

Each phase has a **done-when** that is testable. Don't start the next phase
until the current one passes.

### Phase 0 — Skeleton (½ day)
Repo, venv, config from env vars, SQLite schema, `pytest` running.

**Done when:** `pytest` passes with one trivial test and `slv.db` exists with all tables.

---

### Phase 1 — Data layer (2–3 evenings)
Four fetchers. Each is idempotent: running twice writes the same rows.
Handle failure by logging and exiting non-zero — never by writing partial data.

**Done when:** `slv ingest` run twice in a row produces identical DB state,
and you have ≥5 years of daily silver history stored.

---

### Phase 2 — Compute layer (2–3 evenings)
Pure functions, no I/O, unit-tested against hand-computed values.

- `indicators.py` — ATR14, EMA20/50, SMA200, RSI14, 20d realized vol, GSR,
  distance from EMA20 in ATR units, position in 60-day range
- `regime.py` — classifies: real yield trend, DXY trend, GSR percentile,
  COT percentile → a small regime state object
- `eventrisk.py` — given an upcoming event and your leverage, returns median
  and 90th-percentile historical silver move, and the resulting equity impact

**Done when:** every function has a unit test with a hand-verified expected value,
and `slv compute` rebuilds the indicators table from scratch in under a second.

---

### Phase 3 — Journal (2 evenings) ⭐ **the keystone**

This is the phase that makes everything else meaningful. Build it before anything
that could be called an agent.

```
slv thesis open   --direction long --entry 72.50-73.20 \
                 --invalidation 70.80 --target 79.00 --size 8
slv thesis close  <id> --exit 70.60
slv journal score
```

**Process rubric — 5 binary checks, scored independent of P&L:**

1. Falsifiable — a specific invalidation price was stated before entry
2. Sized — position size respected event risk in the holding window
3. Honored — exit happened at or before the stated invalidation
4. In-condition — entry occurred inside the stated entry zone
5. Pre-committed — the thesis row was written before the trade, not after

Rows are immutable. `slv thesis open` refuses to overwrite. This is enforced in
code, not discipline.

**Done when:** you can open, close, and score a thesis, and the score is
computed with no reference to whether the trade made money.

---

### Phase 4 — Brief assembly (2 evenings)
`slv brief` builds a facts object from computed state and writes dated markdown
to `briefs/`. **No LLM in this phase** — templated output only.

Then feed the facts object into your existing silver-analyst skill so it stops
scraping indicator values from articles and starts reading real numbers.

**Done when:** `slv brief` produces a dated file every morning via cron, and the
skill consumes the facts JSON instead of running web searches for technicals.

---

### Phase 5 — Agent loop (1–2 weekends) — *the learning phase*
smolagents with three custom tools: `get_regime`, `get_indicators`,
`search_journal`. Local model first, then swap to Claude and compare.

Sandbox `CodeAgent` — it executes generated Python and this machine holds your
API keys.

**Done when:** the agent answers "what did the last three pullback-to-EMA20
theses look like, and how did they score?" by calling tools, not by guessing.

---

### Phase 6 — Backtest engine (multiple weekends) — *the hard part*

**Sealed holdout is enforced at the tool layer.** The agent has no tool that
reaches the holdout period. Unlocking it is a manual CLI action by you, logged,
and rare.

- Walk-forward, not a single train/test split
- Every run logged to `backtest_runs` for multiple-comparison accounting
- Minimum trade count before any result is reportable
- Report expectancy and R-multiple distribution — **never win rate**

**Done when:** you can express a rule, backtest it on train data, and the system
refuses to touch holdout without an explicit unlock.

---

### Phase 7 — MCP server (1 weekend)
Expose the read tools over MCP so Claude Code queries your computed state
directly. Cheaper (compact facts instead of bloated search results) and better
(real numbers).

---

## Cadence

**Daily (cron, 06:00 ICT):** `slv ingest && slv compute && slv brief`
**Weekly (Saturday):** `slv journal score`, review open theses, log a note
**Monthly:** review process scores; check whether any regime input has actually
predicted anything
**Quarterly:** re-read this plan; delete features that aren't earning their place

---

## Kill conditions

Written now, while you're unbiased, because you won't want to write them later.

- **6 months in:** if median process score isn't improving, the tool isn't
  changing your behaviour. Stop building features and ask why.
- **Any phase that stalls 3+ weeks:** the phase is too big. Cut its scope in half.
- **If you start wanting more instruments:** re-read the non-goals. That urge is
  action bias, not opportunity.
- **If the briefing becomes something you skim:** it's too long. Cut it.

---

## First session checklist

1. `mkdir agentic-trading-slv && cd agentic-trading-slv && git init`
2. venv, install `pandas numpy yfinance requests pytest`
3. Get a FRED API key (free, instant)
4. Write `db.py` with the schema above, run it, confirm `slv.db` exists
5. Write `fetch/prices.py`, pull 5 years of `SI=F`, store it
6. Commit

That's Phase 0 and half of Phase 1 in one evening.

---

## The honest risk

The failure mode is not losing money. It's building an elegant machine that
generates confident narratives and never discovering whether they were any good.

Phase 3 is the only defence. It is also the least interesting phase to build.
Build it anyway, and build it before Phase 5.
