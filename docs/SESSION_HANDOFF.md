# AprilAlgo — Session Handoff Document

## Today's Handoff (2026-04-17)

### Executive status
- v0.3 and v0.4 sprint scope is functionally complete in code/tests/docs; **information bars are wired into the ML train/predict/SHAP/walk-forward path** (see `CHANGELOG.md` Unreleased and `data/loader.py`).
- Cursor plan-file YAML todos are not authoritative; use `CHANGELOG.md`, green pytest, and this handoff for “what’s done.”
- Current validation status: `uv run pytest tests/ -q` -> **128 passed, 1 skipped** (~129 collected; see `tests.md`).

### What changed in this session
- **Information bars in ML:** `load_ohlcv_for_ml`, `meta.json` recipe, `ml_xgboost` + backtest loop parity, walk-forward uses the same bar series as ML when YAML enables bars.
- Completed v0.4.1 completion pass across explainability, bars, and walk-forward polish:
  - SHAP module + CLI exports + report section
  - Information-driven bars module + CLI command
  - Walk-forward CLI summary/per-fold return + Streamlit enhancements
- Also closed remaining integration/test gaps across v0.3/v0.4 workstreams:
  - ML evaluator outputs and fold leakage checks
  - Importance table schema/ranking checks
  - Logger full-schema validation helpers
  - ML strategy + position sizing/log event path coverage
  - CLI (`train/evaluate/predict/walk-forward/shap/bars`) smoke coverage
  - Streamlit page import smoke for ML/metrics/walk-forward views
  - Portfolio runner pooled-capital/risk-cap behavior tests
  - Reporting section-id coverage for HTML output
- Aligned documentation and inventory:
  - `tests.md` regenerated to **113 collected tests**
  - `AGENTS.md` test-count reference updated
  - `ARCHITECTURE.md` roadmap text updated for shipped v0.3/v0.4 baseline + v0.4.1 additions
  - `DATA_SCHEMA.md` updated for SHAP artifacts, information-driven bars, and walk-forward output schema

### Important implementation notes
- `hmmlearn` support remains optional:
  - Regime HMM path exists behind a feature flag.
  - Test suite intentionally skips one test when `hmmlearn` is unavailable.
- Plan file statuses are not authoritative; rely on:
  - `CHANGELOG.md` release sections (`0.3.0`, `0.4.0`, `0.4.1`)
  - green pytest run
  - feature files/tests present in `src/` and `tests/`

### Known caveats / follow-ups
- SHAP commentary narrative UX can still be polished.
- Information-driven bars tuning presets/research still open.
- Walk-forward UX deep analytics can still be expanded in Streamlit.
### Next sprint (active)
- Polish: SHAP commentary UX, information-bar presets/research helpers, richer walk-forward Streamlit analytics.
- Optional hardening: shared portfolio risk caps, deeper walk-forward tuner UI (see `ARCHITECTURE.md` v0.5).

### Recommended next action
- Run `uv run pytest tests/ -v` after substantive changes; keep `CHANGELOG.md` Unreleased in sync until you cut **0.4.2** (or merge Unreleased into a dated release).

> **Date:** April 17, 2026  
> **For:** Joshua (project owner)  
> **Purpose:** Everything you need to pick up where we left off, explain the project to someone else, or start a new AI session.

---

## What Is AprilAlgo?

AprilAlgo is a stock trading backtesting system you're building. The idea is simple: take a bunch of technical indicators, figure out when they agree with each other (called "confluence"), and calculate the probability that a trade setup actually works. If most indicators say "buy" at the same time, across multiple timeframes, that's a high-confidence signal.

The stack now includes **XGBoost + triple-barrier labels + purged CV**, optional **information-driven bars** in the ML path, **SHAP** exports, and a **Streamlit ML lab**; rule-based indicators and confluence remain first-class.

---

## What's Built (Current State)

Think of the project like building a house. Here's where we are:

### The Foundation (v0.1) — Done
- Load stock price data from CSV files (73 symbols, daily + 5-minute bars)
- Basic indicators: RSI, SMA, Bollinger Bands, Volume Trend
- A backtesting engine that simulates trading bar-by-bar (no cheating with future data)
- Trade tracking, portfolio management, performance metrics (Sharpe ratio, win rate, drawdown, etc.)
- A simple RSI + SMA strategy as a proof of concept
- Command-line interface to run backtests

### The Walls (v0.2) — Done
- **6 new indicators**: DeMark Sequential, Hurst Exponent, Ehlers cycle filters (Super Smoother, Roofing Filter, Decycler), Turn Measurement Index (TMI), Price-Volume Sequences
- **Dual-signal system**: every indicator now says both "this is bullish" AND "this is bearish" at the same time — the confluence engine figures out which interpretation wins
- **Confluence scoring**: counts how many indicators agree, produces a score from -1.0 (all bearish) to +1.0 (all bullish)
- **Multi-timeframe alignment**: merge signals from daily + intraday into one view
- **Parameter tuner**: automatically test thousands of indicator parameter combinations to find what works best
- **Position sizing**: Fractional Kelly Criterion, fixed percentage, and ATR-based methods
- **Data fetcher**: download fresh data from Massive.com API (the company that used to be Polygon.io)
- **DeMark Confluence strategy**: a real strategy that uses DeMark exhaustion signals confirmed by multi-indicator confluence

### The Interior (v0.2 → v0.4+) — Done
- **Indicator descriptor system**: every indicator self-describes its parameters, display name, and category. The UI and tuner auto-generate controls from this — adding a new indicator requires zero UI code changes.
- **Configurable strategy**: pick indicators from a catalog and trade on confluence without new Python for each combo.
- **Streamlit UI**: multi-page app (charts, signals, dashboard, tuner, plus **ML / regime / portfolio / walk-forward** labs).
- **Machine learning path**: triple-barrier targets, feature matrix from indicators, XGBoost train/load, purged CV, gain/permutation/SHAP importance, JSONL signal logging, **`ml_xgboost`** strategy.
- **Meta / reporting**: meta-label helpers, sampling weights, vol regime buckets, walk-forward splits, HTML report sections, multi-symbol portfolio runner.
- **Information-driven bars**: CLI CSV builder **and** optional YAML-driven aggregation inside the ML pipeline (same recipe in `meta.json` for predict and backtest parity).
- **Test suite**: ~129 collected tests (`tests.md`; one HMM-related test may skip without ``uv sync --extra hmm``).

---

## How to Run It

Open a terminal in the `AprilAlgo` folder and run:

| What you want to do | Command |
|---------------------|---------|
| Install everything | `uv sync` |
| Open the web dashboard | `uv run streamlit run src/aprilalgo/ui/app.py` |
| Run a backtest from command line | `uv run python main.py` |
| Run a specific strategy + symbol | `uv run python main.py --symbol NVDA --strategy demark_confluence` |
| Run the test suite | `uv run pytest tests/ -v` |
| Download fresh market data | `uv run python scripts/fetch_data.py --symbols AAPL,NVDA --timeframe daily` |

---

## Project File Map

```
AprilAlgo/
├── src/aprilalgo/           # The actual code
│   ├── data/                # Loading, fetching, storing price data
│   ├── indicators/          # 11 technical indicators + descriptor catalog
│   │   └── descriptor.py    # Single source of truth for indicator metadata
│   ├── confluence/          # Multi-timeframe signal scoring
│   ├── tuner/               # Parameter optimization engine
│   ├── backtest/            # Trade simulation engine
│   ├── strategies/          # Trading strategies (3 built-in)
│   └── ui/                  # Streamlit (Charts, Signals, Dashboard, Tuner, ML/Regime/Portfolio labs)
├── tests/                   # pytest + fixtures
├── data/                    # CSV price files (not in git)
├── configs/                 # YAML configuration
├── docs/                    # Reference docs (HANDOFF, LEARNING, REPO_ANALYSIS)
├── ARCHITECTURE.md          # Full system design through v0.4
├── CHANGELOG.md             # What changed in each version
├── CLAUDE.md                # Context file for Claude AI
└── AGENTS.md                # Rules for AI coding agents
```

---

## The 11 Indicators

| Indicator | What it measures | Category |
|-----------|-----------------|----------|
| **RSI** | Momentum (overbought/oversold) | Momentum |
| **SMA** | Trend direction (price above/below average) | Trend |
| **Bollinger Bands** | Volatility (price at band extremes) | Volatility |
| **Volume Trend** | Is volume confirming the price move? | Volume |
| **DeMark Sequential** | Exhaustion patterns (setup 9, countdown 13) | Exhaustion |
| **Hurst Exponent** | Is the market trending or mean-reverting? | Regime |
| **Super Smoother** | Low-lag trend filter | Cycle |
| **Roofing Filter** | Isolates the dominant market cycle | Cycle |
| **Decycler** | Extracts the trend by removing cycles | Trend |
| **TMI** | Curvature — detects trend turns | Momentum |
| **PV Sequences** | Price-Volume state transitions (conviction) | Pattern |

All indicators produce parameterized column names (e.g., `rsi_14_bull`, `sma_20_bear`) so you can run the same indicator with different settings without columns overwriting each other.

---

## Core strategies

| Strategy | How it works |
|----------|--------------|
| **RSI + SMA** | Buy when RSI is oversold AND price is above the SMA. Sell when RSI is overbought OR price drops below SMA. |
| **DeMark Confluence** | Buy when DeMark signals exhaustion AND confluence score confirms. Sell on reversal signal, confluence flip, or stop loss. |
| **Configurable** | You pick which indicators to use from a list, set parameters, and it trades based on the total confluence score. No code changes needed. |
| **ml_xgboost** | Loads a saved model bundle; enters when predicted P(take-profit) clears a threshold. Use the **same raw OHLCV** as training when the model used `information_bars`. |

---

## Key Design Decisions Worth Knowing

1. **Dual-signal principle**: RSI < 30 is simultaneously "bullish" (mean reversion: bounce likely) AND "bearish" (momentum: downtrend continuing). The confluence engine resolves which interpretation wins by looking at what all the other indicators say.

2. **Parameterized columns**: `rsi_14_bull` not `rsi_bull`. This was a bug fix — calling RSI with period 14 then period 7 used to silently overwrite the first result. Now both coexist.

3. **Descriptor catalog**: instead of hardcoding indicator lists in 5 places (UI charts, UI signals, UI dashboard, UI tuner, strategies), there's one registry in `descriptor.py` that everything reads from.

4. **No look-ahead bias**: the backtester processes one bar at a time. Strategies can only see the current bar and past bars, never future bars.

---

## What's NOT Built Yet (The Roadmap)

### Shipped baselines (v0.3 / v0.4 / v0.4.1)

| Area | Status |
|------|--------|
| ML pipeline (triple-barrier, features, XGBoost, purged CV, CLI) | Done |
| SHAP values + importance CSV + report/UI hooks | Done (baseline) |
| Information-driven bars (CLI + **ML YAML / meta.json / `ml_xgboost`**) | Done (baseline) |
| Walk-forward CLI + Streamlit lab + summary JSON | Done (baseline) |
| Meta-label / sampling / regime / portfolio runner / HTML report | Done (baseline) |

### Still open (polish / research)

| Item | Notes |
|------|--------|
| SHAP commentary UX | Narrative explanations alongside plots |
| Bar presets / research mode | Threshold helpers, batch experiments |
| Walk-forward UI depth | Richer fold analytics in Streamlit |
| Shared portfolio risk | Advanced cross-symbol constraints (beyond current runner) |

---

## External Resources We've Researched

Full details in `docs/REPO_ANALYSIS.md`. Quick summary:

| Resource | How we'll use it |
|----------|-----------------|
| **massive-com/client-python** | Already integrated — data fetcher uses this |
| **shap/shap** | Direct dependency for v0.3 (TreeExplainer for XGBoost) |
| **aticio/legitindicators** | Already used as formula reference for Hurst and Ehlers (ported math, not code) |
| **hudson-and-thames/mlfinlab** | **Cannot use code** (proprietary license) — but concepts like triple-barrier labeling and purged CV are in published papers and we'll implement manually |
| **Rachnog/Deep-Trading** | Educational notebooks about ML trading pitfalls — read before building v0.3 |

---

## Suggested Next Steps (Pick One)

### Path A: Research bars + ML
Tune `information_bars` thresholds per symbol; compare purged CV metrics vs calendar bars; document findings in `configs/ml/`.

### Path B: Operational polish
Improve SHAP narrative UX, walk-forward lab charts, and Streamlit export flows.

### Path C: Scale data + backtests
Use Massive.com to extend histories; run portfolio runner and tuner across the full universe.

### Path D: Release hygiene
Fold `CHANGELOG.md` Unreleased into the next semver tag; regenerate `tests.md` whenever tests change.

---

## How to Start a New AI Session

Copy-paste something like this to give a new AI session full context:

> I'm building AprilAlgo, a modular stock trading backtester in Python. Read CLAUDE.md for project context, AGENTS.md for coding rules, and ARCHITECTURE.md for the full system design. The project is at v0.2 with 11 indicators, confluence scoring, a parameter tuner, 3 strategies, a Streamlit UI, and 39 passing tests. I want to work on [YOUR GOAL HERE].

The AI will read those files and have everything it needs.

---

## Quick Technical Reference

| Topic | Where to look |
|-------|---------------|
| How indicators work | `src/aprilalgo/indicators/` — each file is small and self-contained |
| How the backtester works | `src/aprilalgo/backtest/engine.py` — 50 lines, very readable |
| How confluence scoring works | `src/aprilalgo/confluence/scorer.py` — auto-detects `*_bull`/`*_bear` columns |
| How to add a new indicator | `CLAUDE.md` → "Adding New Components" section |
| How to add a new strategy | `CLAUDE.md` → "Adding New Components" section |
| Full system architecture | `ARCHITECTURE.md` — includes ASCII diagrams and data flow |
| External repo research | `docs/REPO_ANALYSIS.md` |
| Beginner explanations of Git, packaging, etc. | `docs/LEARNING.md` |
| Version history | `CHANGELOG.md` |
| ML + information bars contracts | `docs/DATA_SCHEMA.md` §9 |

---

## Cursor Handoff Protocol (v0.5+)

Starting with the v0.5 (ML depth) backlog, every 10-task Sprint in [`BACKLOG.md`](../BACKLOG.md) ends with a **handoff entry** appended below. These entries are the canonical way a finishing agent hands context to the next one.

### Standing rules

1. **No sprint may end with failing tests or with a red BACKLOG checkbox for items the agent claims as completed.** If a task cannot be finished in the sprint, leave its box unchecked and record why in the handoff entry.
2. **Any change to `meta.json` keys (new, renamed, removed, or changed types) must update [`docs/DATA_SCHEMA.md`](DATA_SCHEMA.md) in the same sprint.** A sprint that bumps the bundle schema without touching DATA_SCHEMA is invalid regardless of passing tests.

### Post-Sprint Handoff Template

Copy this block verbatim and fill it in at the end of every sprint.

```
### Sprint <N> handoff — <YYYY-MM-DD>
- Sprint title: <from BACKLOG.md>
- Tasks completed (ids): S<N>-T01, S<N>-T02, ...
- Tasks deferred (ids + reason): S<N>-T0X — <why>
- Commits / diffs: <SHAs or file list>
- Test command run + result: `uv run pytest tests/ -q` -> X passed / Y skipped / 0 failed
- New public symbols: <module.symbol>, ...
- meta.json / bundle schema changes: <new keys + types>, DATA_SCHEMA section updated: <§N>
- Docs updated: BACKLOG rows checked, CHANGELOG Unreleased row(s), DATA_SCHEMA §, README row, other
- Warning Zone deltas: new items added / items closed (reference PROJECT_STATE.md §3)
- Downstream impact observed: <tests/modules touched beyond the sprint's listed files>
- Next sprint gate: approved / blocked (reason)
```

### Start-of-sprint gate

Before starting Sprint `N`, the agent must:

1. Re-read the latest handoff entry above.
2. Confirm `uv run pytest tests/ -q` is green on the current branch.
3. Confirm the Sprint `N-1` handoff marked `Next sprint gate: approved`.
4. Only then begin Sprint `N` task T01.

### Sprint handoff entries

<!-- Append Sprint 1..12 handoff entries below this marker, newest last. -->

### Sprint 1 handoff — 2026-04-17
- Sprint title: Sample-weight plumbing (trainer signature only)
- Tasks completed (ids): S1-T01 … S1-T10 (all)
- Tasks deferred (ids + reason): none
- Commits / file diffs: `src/aprilalgo/ml/trainer.py`, `src/aprilalgo/cli.py`, `configs/ml/default.yaml`, `tests/test_trainer_sample_weight.py`, `tests/test_cli_ml.py`, `docs/DATA_SCHEMA.md` §11, `CHANGELOG.md`, `AGENTS.md`, `BACKLOG.md`, `tests.md`
- Test command run + result: `uv run pytest tests/ -q` → **119 passed / 1 skipped / 0 failed**; `uv run pytest tests/ --collect-only -q` → **120 tests collected**
- New public symbols: `aprilalgo.cli._weights_for_training`, `aprilalgo.cli._sampling_meta` (underscore = internal CLI helpers; importable for smoke checks)
- meta.json / bundle schema changes: new key **`sampling`** (object: `strategy` str default `none`; optional `random_state` int from YAML). **DATA_SCHEMA** updated: **§11 Model bundle — sampling metadata**
- Docs updated: BACKLOG Sprint 1 checkboxes; CHANGELOG Unreleased; DATA_SCHEMA §11; AGENTS.md test count; `tests.md` regenerated
- Warning Zone deltas: none (Sprint 1 only)
- Downstream impact observed: `cmd_importance` now uses `_weights_for_training` for parity with `train`
- Next sprint gate: **approved** (proceed to Sprint 2 — uniqueness / bootstrap wiring)

### Sprint 2 handoff — 2026-04-18
- Sprint title: Sequential bootstrap + uniqueness weights in training
- Tasks completed (ids): S2-T01 … S2-T10 (all)
- Tasks deferred (ids + reason): none
- Commits / file diffs: `src/aprilalgo/cli.py`, `configs/ml/default.yaml`, `tests/test_sampling.py`, `tests/test_cli_ml.py`, `docs/DATA_SCHEMA.md` §11 (expanded), `README.md`, `CHANGELOG.md`, `BACKLOG.md`, `tests.md`, `AGENTS.md` / `PROJECT_STATE.md` / header counts
- Test command run + result: `uv run pytest tests/ -q` → **123 passed / 1 skipped / 0 failed**; `uv run pytest tests/ --collect-only -q` → **124 tests collected**
- New public symbols: none (uses existing `uniqueness_weights`, `sequential_bootstrap_sample`)
- meta.json / bundle schema changes: **`sampling`** object extended for `bootstrap`: always includes resolved `random_state` and `n_draw` (`null` or int). Uniqueness leaves only `strategy`. **DATA_SCHEMA** §11 expanded accordingly.
- Docs updated: BACKLOG Sprint 2; CHANGELOG; DATA_SCHEMA §11; README table + paragraph
- Warning Zone deltas: none
- Downstream impact observed: none beyond listed files
- Next sprint gate: **approved** (Sprint 3 — primary OOF capture)

### Sprint 3 handoff — 2026-04-18
- Sprint title: Primary OOF capture (+ optional `hmm` extra)
- Tasks completed (ids): S3-T01 … S3-T10 (all), plus **`hmm` optional dependency** in `pyproject.toml` and docs (HMM regime path)
- Commits / file diffs: `src/aprilalgo/ml/oof.py`, `src/aprilalgo/ml/__init__.py`, `src/aprilalgo/cli.py` (`oof`, `_xgb_estimator_factory`, `cmd_oof`), `tests/test_oof.py`, `tests/test_cli_ml.py`, `docs/DATA_SCHEMA.md` §12, `README.md`, `CLAUDE.md`, `CHANGELOG.md`, `BACKLOG.md`, `pyproject.toml`, `tests.md`, `AGENTS.md`, `PROJECT_STATE.md`, `docs/SESSION_HANDOFF.md`
- Test command run + result: `uv run pytest tests/ -q` → **128 passed / 1 skipped / 0 failed**; collect → **129 tests**
- New public symbols: `aprilalgo.ml.oof.compute_primary_oof` (also re-exported from `aprilalgo.ml`)
- meta.json / bundle schema changes: **`oof`** key `{"path": "oof_primary.csv"}` merged when `oof` runs and `meta.json` exists (**DATA_SCHEMA** §12)
- Docs updated: BACKLOG Sprint 3; CHANGELOG; DATA_SCHEMA §11 note for `hmm` extra + §12 OOF; README install + quick command; CLAUDE ML CLI list
- Warning Zone deltas: HMM item **partially mitigated** — optional `[hmm]` extra documented; default envs without `hmmlearn` still skip HMM smoke test
- Downstream impact observed: `evaluate` refactored to shared `_xgb_estimator_factory` (behavior unchanged)
- Problems / follow-ups: **`hmmlearn` cannot be a default dev dependency** on some stacks (e.g. Python 3.14 + no MSVC) because it often builds from source — use `--extra hmm` when wheels exist
- Next sprint gate: **approved** (Sprint 4 — meta-label bundle)

