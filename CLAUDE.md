# agentic-trading-slv

Decision-support for discretionary XAG/USD swing trading. Reads, computes,
verifies, proposes. **Never executes.**

Full roadmap: @PLAN.md

## Stack

Python 3.11+ · SQLite (`data/slv.db`) · pandas/numpy · pytest · cron
Phase 5+: smolagents, Ollama. Not before.

## Commands

```
pytest                  # all tests
slv ingest               # fetch raw data (idempotent)
slv compute              # rebuild derived indicators
slv brief                # write today's brief to briefs/
```

## Non-negotiable constraints

- **XAG/USD only.** Never add another instrument without me asking explicitly.
- **No execution.** No broker APIs, no order placement, not even stubs.
- **Raw tables are append-only.** `prices`, `macro`, `cot`, `events`.
  Derived tables (`indicators`) can be dropped and rebuilt.
- **`theses` rows are immutable.** Corrections create a new row. Enforce in code.
- **Fetchers must be idempotent.** Running twice writes identical state.
  On failure: log, exit non-zero, write nothing partial.
- **No LLM in the compute layer.** Indicators, regime, and event risk are
  deterministic Python with unit tests. The model never produces a number.
- **Report expectancy and R-multiple, never win rate.**
- Secrets from env vars only. Never commit keys or `data/slv.db`.

## Working agreement

- Follow PLAN.md phase order. Don't build ahead — if a later phase seems needed
  now, say so and wait for my answer.
- One phase per session. Stop at the phase's done-when and let me verify.
- Small commits, one logical change each.
- Every compute function needs a unit test with a hand-verified expected value.
  Not a snapshot of its own output.
- If a decision isn't in PLAN.md, ask instead of assuming. Don't invent
  thresholds, lookback windows, or scoring weights silently.
- Prefer boring and readable over clever. This is a learning project — explain
  non-obvious choices in a comment or in your reply.
- New feature ideas go in IDEAS.md, not into the code.

## Style

- Standard library and pandas/numpy first. Justify any new dependency.
- Type hints on public functions.
- Pure functions in `compute/` — no I/O, no DB access, no network.
- Fail loud. No silent `except: pass`, no default values papering over missing data.

## Domain notes

- `SI=F` is silver futures, used as the XAG/USD proxy. Note the basis; don't
  treat it as spot.
- The CFTC COT CSV changes column names between years. Parse defensively.
- Silver's daily ATR runs ~2%; CPI and FOMC days regularly hit 3–5%.
  At 3x leverage that is 6–15% of equity. Event risk sizing matters more than
  direction calls.
- Weekend gap risk is real — spot closes Friday, reopens Sunday, stops don't
  exist in between.
