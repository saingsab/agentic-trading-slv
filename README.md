# agentic-trading-slv

A decision-support and process-discipline system for discretionary swing trading
in silver (XAG/USD).

It reads market data, computes state deterministically, scores my own decisions
against a fixed rubric, and writes a daily brief.

**It does not predict price, generate signals, or place orders.**

---

## Why this exists

Agentic coding works because verification is fast, objective, and cheap. Tests
pass or fail in seconds, and a bad change costs a `git revert`.

Trading inverts all of that. Feedback takes weeks, outcomes are noisy, the
environment is adversarial and non-stationary, and a losing trade is not a stack
trace — it's ambiguous evidence. A trade can lose money after a good decision and
make money after a reckless one.

So this project doesn't try to automate the decision. It automates everything
around the decision that *is* verifiable, and it builds the missing piece:
a scoring loop that grades **process, not P&L**.

---

## What it does

**Deterministic market state.** Indicators are computed from stored OHLCV, not
scraped from articles. Same method, same source, every day, reproducible.

**Event risk sizing.** Given an upcoming release and a leverage level, it reports
the historical distribution of silver's move and the resulting equity impact —
so the question becomes "is my stop inside the event range?" rather than "what
will CPI be?"

**A prediction journal.** Every thesis is written down *before* entry with an
explicit invalidation level. Rows are immutable. Corrections create new rows.

**Process scoring.** Five binary checks per closed thesis, all independent of
whether it made money:

1. Falsifiable — a specific invalidation price was stated before entry
2. Sized — position size respected event risk in the holding window
3. Honored — exit happened at or before the stated invalidation
4. In-condition — entry occurred inside the stated entry zone
5. Pre-committed — the thesis was logged before the trade, not after

**Backtesting with a sealed holdout.** An agent that can run backtests will
p-hack at industrial scale. The holdout period is unreachable at the tool layer,
every run is logged for multiple-comparison accounting, and results are reported
as expectancy and R-multiple distribution — never win rate.

---

## What it is not

- Not a signal generator
- Not a trading bot — there is no broker integration, not even a stub
- Not a price prediction model
- Not multi-asset. Silver only, by design.

---

## Architecture

Permissions tighten as you move up the stack, the same way `git push --force`
is gated while reading files is not.

```
Data tools        prices, FRED, COT, calendar        always allowed
Compute layer     deterministic, unit-tested         always allowed
Verification      backtest, journal scoring          logged and counted
Agent loop        proposes, never decides            read + propose only
Execution         order placement                    human only
Sealed holdout    no tool reaches it                 unlocked manually, rarely
```

The compute layer contains no LLM. The model never produces a number — it reads
finished facts and writes prose.

---

## Stack

Python 3.11+ · SQLite · pandas/numpy · pytest · cron
Later phases: smolagents, Ollama for local extraction, MCP server

All data sources are free: `yfinance` for OHLCV, FRED for real yields and DXY,
the CFTC weekly COT report for positioning, and a hand-seeded calendar of
scheduled releases.

---

## Commands

```bash
slv ingest               # fetch raw data (idempotent)
slv compute              # rebuild derived indicators
slv brief                # write today's brief
slv thesis open ...      # log a thesis before entry
slv thesis close <id>    # record the exit
slv journal score        # grade closed theses on process
```

---

## Status

| Phase | Status |
|---|--------|
| 0 — Skeleton |        |
| 1 — Data layer |        |
| 2 — Compute layer |     |
| 3 — Journal |        |
| 4 — Brief assembly |  |
| 5 — Agent loop | ⬜     |
| 6 — Backtest engine | ⬜     |
| 7 — MCP server | ⬜     |

See [PLAN.md](PLAN.md) for the full roadmap, locked decisions, and kill conditions.

---

## Disclaimer

Personal project. Not investment advice, not a recommendation to trade, and no
claim of profitability. Leveraged trading in silver can lose more than you put in.
The journal and brief outputs are excluded from version control.
