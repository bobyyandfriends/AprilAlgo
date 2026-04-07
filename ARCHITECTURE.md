# AprilAlgo — Full System Architecture

> **Version:** Design Doc v1.0  
> **Date:** March 29, 2026  
> **Status:** Proposed — review before implementation  
> **Scope:** End-state vision (v0.1 → v0.4+)

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Versioned Roadmap](#2-versioned-roadmap)
3. [Full Folder Structure](#3-full-folder-structure)
4. [Module Descriptions](#4-module-descriptions)
5. [Data Flow Diagrams](#5-data-flow-diagrams)
6. [Dependency Map](#6-dependency-map)
7. [External Library Integration Plan](#7-external-library-integration-plan)
8. [Design Principles](#8-design-principles)

---

## 1. System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        AprilAlgo System                             │
│                                                                     │
│  ┌──────────┐  ┌────────────┐  ┌───────────┐  ┌────────────────┐  │
│  │  DATA    │→ │ INDICATORS │→ │ STRATEGY  │→ │  BACKTESTER    │  │
│  │  LAYER   │  │  ENGINE    │  │  + ML     │  │  + METRICS     │  │
│  └──────────┘  └────────────┘  └───────────┘  └────────────────┘  │
│       ↑                              ↑               │             │
│  ┌──────────┐  ┌────────────┐  ┌───────────┐  ┌─────▼──────────┐  │
│  │ FETCHER  │  │ CONFLUENCE │  │   SHAP    │  │  STREAMLIT UI  │  │
│  │(Massive) │  │  SCORING   │  │ EXPLAINER │  │  + DASHBOARD   │  │
│  └──────────┘  └────────────┘  └───────────┘  └────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

**Core Pipeline (left to right):**
1. **Data Layer** — Fetch OHLCV from Massive API or load from CSV; resample to multiple timeframes
2. **Indicator Engine** — Apply technical indicators (DeMark, RSI, Hurst, Ehlers, etc.)
3. **Confluence Scoring** — Evaluate multi-timeframe, multi-indicator agreement → probability score
4. **Strategy + ML** — Rule-based or ML-driven signal generation
5. **Backtester** — Bar-by-bar simulation with portfolio management
6. **Metrics + Explainability** — Performance stats, SHAP explanations, trade logging
7. **UI** — Streamlit dashboard for charts, signals, SHAP plots, performance

---

## 2. Versioned Roadmap

### v0.1 — Basic Backtester ✅ COMPLETE
- Load OHLCV data from CSV
- RSI, SMA, Bollinger Bands, Volume Trend indicators
- Bar-by-bar backtest engine with Trade/Portfolio/Metrics
- RSI+SMA strategy
- CLI entry point

### v0.2 — Advanced Indicators + Multi-Timeframe + Data Fetcher
- **Update data fetcher** to Massive (formerly Polygon.io) package
- **Dual-signal indicators** — every indicator emits both bullish and bearish reinforcement signals
- **DeMark indicators** (Sequential, Setup, Countdown)
- **Hurst Exponent** (R/S method + rolling windows)
- **Ehlers cycle indicators** (Super Smoother, Roofing Filter, Decycler)
- **Turn Measurement Index (TMI)**
- **Price-Volume Sequences** (state transitions)
- **Multi-timeframe engine** — align and score indicators across timeframes
- **Confluence scoring** — probability of setup success based on indicator agreement
- **Parameter tuning engine** — grid search over indicator params (e.g. SMA 19 vs 20), find best combos per symbol/timeframe
- **Fractional Kelly** position sizing

### v0.3 — Machine Learning + Explainability + UI
- **Triple-barrier labeling** for supervised learning targets
- **XGBoost classifier** for signal generation
- **Purged k-fold CV** (no temporal leakage)
- **SHAP integration** — TreeExplainer on XGBoost
- **Intelligent Commentary Generator** — human-readable trade explanations
- **Streamlit UI** — charts, signal feed, SHAP plots, dashboard
- **Signal logging** — persist every signal with features, SHAP, outcome

### v0.4 — Meta-Modeling + Advanced Features
- **Meta-labeling** — secondary model predicts primary model accuracy
- **Information-driven bars** (volume, dollar, tick bars)
- **Feature importance** (MDI, MDA, SHAP-based)
- **Regime detection** (volatility clustering, HMM or similar)
- **Walk-forward optimization** framework
- **Multi-asset portfolio** — run strategies across all 73 symbols
- **Export/reporting** — PDF/HTML performance reports

---

## 3. Full Folder Structure

```
AprilAlgo/
├── main.py                         # CLI entry point for backtests
├── pyproject.toml                  # Project config + dependencies
├── uv.lock                         # Locked dependency versions
├── .gitignore
├── .python-version
│
├── configs/                        # Configuration files
│   ├── default.yaml                # Default backtest settings
│   ├── strategies/                 # Per-strategy config overrides
│   │   ├── rsi_sma.yaml
│   │   ├── demark.yaml
│   │   └── ml_xgboost.yaml
│   └── symbols.yaml                # Watchlist / universe definition
│
├── data/                           # Raw market data (gitignored)
│   ├── daily_data/                 # *_daily.csv (73 tickers)
│   ├── 5min_data/                  # *_5min.csv (73 tickers)
│   ├── 1min_data/                  # Future: minute bars
│   ├── hourly_data/                # Future: hourly bars
│   └── weekly_data/                # Future: weekly bars
│
├── outputs/                        # Backtest results (gitignored)
│   ├── trades/                     # Trade logs per run
│   ├── signals/                    # Signal logs with SHAP values
│   ├── equity/                     # Equity curves
│   └── reports/                    # HTML/PDF reports
│
├── models/                         # Saved ML models (gitignored)
│   ├── xgboost/                    # Trained XGBoost models
│   └── meta/                       # Meta-labeling models
│
├── notebooks/                      # Jupyter notebooks for research
│   ├── indicator_exploration.ipynb
│   ├── confluence_analysis.ipynb
│   └── ml_experiments.ipynb
│
├── src/aprilalgo/                  # Main Python package
│   ├── __init__.py                 # Version, author, top-level exports
│   ├── config.py                   # YAML config loader
│   │
│   ├── data/                       # === DATA LAYER ===
│   │   ├── __init__.py
│   │   ├── loader.py               # Load OHLCV from CSV by symbol+timeframe
│   │   ├── fetcher.py              # [v0.2] Fetch from Massive API → CSV
│   │   ├── store.py                # CSV/Pickle/Parquet I/O helpers
│   │   ├── resampler.py            # Resample OHLCV across timeframes
│   │   └── universe.py             # [v0.2] Symbol universe management
│   │
│   ├── indicators/                 # === INDICATOR ENGINE ===
│   │   ├── __init__.py
│   │   ├── registry.py             # Pluggable indicator pipeline
│   │   ├── rsi.py                  # Relative Strength Index
│   │   ├── sma.py                  # Simple Moving Average
│   │   ├── bollinger.py            # Bollinger Bands
│   │   ├── volume_trend.py         # Volume expansion confirmation
│   │   ├── demark.py               # [v0.2] DeMark Sequential/Setup/Countdown
│   │   ├── hurst.py                # [v0.2] Hurst Exponent (multiple windows)
│   │   ├── ehlers.py               # [v0.2] Super Smoother, Roofing Filter, Decycler
│   │   ├── tmi.py                  # [v0.2] Turn Measurement Index
│   │   └── pv_sequences.py         # [v0.2] Price-Volume state transitions
│   │
│   ├── confluence/                 # === MULTI-TIMEFRAME CONFLUENCE === [v0.2]
│   │   ├── __init__.py
│   │   ├── timeframe_aligner.py    # Align indicators across timeframes
│   │   ├── scorer.py               # Confluence score calculation
│   │   └── probability.py          # Historical win-rate by confluence level
│   │
│   ├── tuner/                      # === PARAMETER OPTIMIZATION === [v0.2]
│   │   ├── __init__.py
│   │   ├── grid.py                 # Define param ranges, generate combos
│   │   ├── runner.py               # Run backtest per combo, collect metrics
│   │   └── analyzer.py             # Rank combos, robustness checks
│   │
│   ├── backtest/                   # === BACKTESTING ENGINE ===
│   │   ├── __init__.py
│   │   ├── engine.py               # Bar-by-bar simulation loop
│   │   ├── trade.py                # Trade dataclass
│   │   ├── portfolio.py            # Cash, positions, equity tracking
│   │   ├── metrics.py              # Performance metrics
│   │   ├── position_sizer.py       # [v0.2] Fractional Kelly + other methods
│   │   └── logger.py               # [v0.3] Signal/trade persistence
│   │
│   ├── strategies/                 # === STRATEGY FRAMEWORK ===
│   │   ├── __init__.py
│   │   ├── base.py                 # Abstract BaseStrategy
│   │   ├── rsi_sma.py              # RSI + SMA crossover
│   │   ├── demark_confluence.py    # [v0.2] DeMark + multi-TF confluence
│   │   └── ml_strategy.py          # [v0.3] XGBoost-driven strategy
│   │
│   ├── labels/                     # === LABELING FOR ML === [v0.3]
│   │   ├── __init__.py
│   │   ├── triple_barrier.py       # Triple-barrier method
│   │   ├── meta_label.py           # [v0.4] Meta-labeling on primary signals
│   │   └── reinforcement.py        # Positive/negative indicator classification
│   │
│   ├── ml/                         # === MACHINE LEARNING === [v0.3]
│   │   ├── __init__.py
│   │   ├── features.py             # Feature matrix construction
│   │   ├── trainer.py              # XGBoost training pipeline
│   │   ├── cv.py                   # Purged k-fold cross-validation
│   │   ├── sampling.py             # [v0.4] Sequential bootstrap
│   │   └── importance.py           # [v0.4] Feature importance (MDI/MDA)
│   │
│   ├── explain/                    # === EXPLAINABILITY === [v0.3]
│   │   ├── __init__.py
│   │   ├── shap_explainer.py       # SHAP TreeExplainer wrapper
│   │   └── commentary.py           # Human-readable signal narratives
│   │
│   ├── meta/                       # === META-MODELING === [v0.4]
│   │   ├── __init__.py
│   │   ├── regime.py               # Volatility regime detection
│   │   └── model_monitor.py        # Track primary model degradation
│   │
│   └── ui/                         # === STREAMLIT UI === [v0.3]
│       ├── __init__.py
│       ├── app.py                  # Main Streamlit app entry point
│       ├── pages/
│       │   ├── charts.py           # Interactive price + indicator charts
│       │   ├── signals.py          # Signal feed with commentary
│       │   ├── shap_plots.py       # SHAP waterfall/force/beeswarm
│       │   └── dashboard.py        # Performance metrics dashboard
│       └── components/
│           ├── sidebar.py          # Symbol/timeframe/strategy selectors
│           └── metrics_cards.py    # KPI display cards
│
├── tests/                          # === TEST SUITE ===
│   ├── __init__.py
│   ├── test_data/                  # Small fixture CSVs for tests
│   ├── test_loader.py
│   ├── test_indicators.py
│   ├── test_backtest.py
│   ├── test_confluence.py
│   └── test_ml.py
│
├── scripts/                        # === UTILITY SCRIPTS ===
│   ├── fetch_data.py               # Bulk download via Massive API
│   └── generate_report.py          # Export backtest report
│
├── docs/                           # === DOCUMENTATION ===
│   ├── AGENTS.md                   # AI agent coding rules
│   ├── CHANGELOG.md                # Version history
│   ├── CLAUDE.md                   # AI context file
│   ├── HANDOFF.md                  # Non-technical project summary
│   ├── LEARNING.md                 # Beginner-friendly explanations
│   ├── REPO_ANALYSIS.md            # External repo research
│   └── ARCHITECTURE.md             # This file
│
└── LICENSE                         # Apache 2.0
```

---

## 4. Module Descriptions

### 4.1 Data Layer (`src/aprilalgo/data/`)

```
                     ┌──────────────┐
                     │  Massive API │ (REST)
                     └──────┬───────┘
                            │ fetcher.py
                            ▼
                     ┌──────────────┐
                     │  CSV Files   │ (data/*_data/)
                     └──────┬───────┘
                            │ loader.py
                            ▼
                     ┌──────────────┐
                     │  DataFrame   │ (datetime-indexed OHLCV)
                     └──────┬───────┘
                            │ resampler.py
                            ▼
                ┌───────────┴───────────┐
                │    Multi-Timeframe    │
                │   DataFrames Dict     │
                │  {"1min": df, ...}    │
                └───────────────────────┘
```

| File | Purpose |
|------|---------|
| `loader.py` | Load CSV → DataFrame. Already exists. |
| `fetcher.py` | **NEW v0.2.** Call `massive.RESTClient.list_aggs()` → save CSVs. |
| `store.py` | CSV/Pickle read/write. Already exists. Future: add Parquet. |
| `resampler.py` | Resample OHLCV bars. Already exists. |
| `universe.py` | **NEW v0.2.** Manage symbol watchlists from YAML. |

### 4.2 Indicator Engine (`src/aprilalgo/indicators/`)

```
  DataFrame (OHLCV)
       │
       ▼
  ┌─────────────────────────────────────┐
  │         IndicatorRegistry           │
  │                                     │
  │  ┌─────┐ ┌────┐ ┌────────┐ ┌────┐  │
  │  │ RSI │ │SMA │ │Bollinger│ │Vol │  │  ← v0.1 (exists)
  │  └─────┘ └────┘ └────────┘ └────┘  │
  │                                     │
  │  ┌──────┐ ┌─────┐ ┌──────┐ ┌────┐  │
  │  │DeMark│ │Hurst│ │Ehlers│ │TMI │  │  ← v0.2 (planned)
  │  └──────┘ └─────┘ └──────┘ └────┘  │
  │                                     │
  │  ┌──────────┐                       │
  │  │PV Sequen.│                       │  ← v0.2 (planned)
  │  └──────────┘                       │
  └─────────────────┬───────────────────┘
                    │
                    ▼
         DataFrame (OHLCV + indicator columns)
```

Each indicator is a **pure function**: `(DataFrame, **params) → DataFrame` (adds new columns).

**Dual-Signal Principle: Every indicator state carries BOTH positive and negative reinforcement simultaneously.**

No indicator reading is purely bullish or bearish. Every state has two competing
forces — and which one dominates depends on what the OTHER indicators are saying.

```
  Example: RSI(14) at current bar = 28

  NEGATIVE reinforcement (mean reversion):
    RSI < 30 means oversold — statistically likely to bounce back up.
    This argues AGAINST shorting and FOR buying the dip.

  POSITIVE reinforcement (momentum):
    RSI < 30 means strong downtrend — momentum is clearly bearish.
    This argues FOR shorting (trend continuation) and AGAINST buying.

  BOTH forces exist at the same time on the same bar.

  Which one "wins" is determined by CONFLUENCE with other indicators:
  - If Hurst says trending + DeMark countdown active + volume expanding down
    → momentum view wins → the downtrend is real, stay short
  - If Hurst says mean-reverting + DeMark 9 completed + volume fading
    → reversion view wins → the bounce is coming, go long

  This is why confluence matters: a single indicator can never tell you
  the answer alone. It always has competing interpretations.
```

Each indicator function adds signal columns in pairs:

| Column Pattern | Meaning |
|----------------|---------|
| `{name}_bull` | Momentum/trend reinforcement for upside (True/False or strength 0-1) |
| `{name}_bear` | Momentum/trend reinforcement for downside (True/False or strength 0-1) |

**Important:** `_bull` and `_bear` are NOT mutually exclusive. In ambiguous regimes,
both can be False (no strong signal) or the confluence engine weighs them with context
from other indicators. The raw bull/bear flags are the first pass; the confluence
scorer provides the final probabilistic verdict.

**New indicators planned for v0.2:**

| Indicator | Columns Added | Reference |
|-----------|---------------|-----------|
| DeMark Sequential | `td_setup`, `td_countdown`, `td_bull`, `td_bear` | DeMark literature |
| Hurst Exponent | `hurst_20`, `hurst_50`, `hurst_100`, `hurst_bull`, `hurst_bear` | R/S analysis + legitindicators ref |
| Super Smoother | `ss_{period}`, `ss_bull`, `ss_bear` | Ehlers / legitindicators ref |
| Roofing Filter | `roof_{lp}_{hp}`, `roof_bull`, `roof_bear` | Ehlers / legitindicators ref |
| Decycler | `decycler_{period}`, `decycler_bull`, `decycler_bear` | Ehlers / legitindicators ref |
| TMI | `tmi_{period}`, `tmi_bull`, `tmi_bear` | Curvature-based trend change |
| PV Sequences | `pv_state` (enum: PU_VU, PU_VD, PD_VU, PD_VD), `pv_bull`, `pv_bear` | State machine |

### 4.2b Parameter Tuning Engine (`src/aprilalgo/tuner/`) — v0.2

Indicator parameters are not one-size-fits-all. An SMA(19) might pair better with
RSI(12) on AAPL daily, while SMA(21) + RSI(14) works better on NVDA 5-min. The
tuner systematically finds which parameter combinations perform best together.

```
  ┌──────────────────────────────────────────────────────────┐
  │                   Parameter Tuner                         │
  │                                                           │
  │   Input: indicator list + param ranges + data + metric    │
  │                                                           │
  │   ┌─────────────────────────────────────────────────┐     │
  │   │  Param Grid (example)                           │     │
  │   │                                                 │     │
  │   │  SMA period:  [10, 15, 19, 20, 21, 25, 50]     │     │
  │   │  RSI period:  [10, 12, 14, 16, 20]             │     │
  │   │  RSI oversold: [25, 28, 30, 32, 35]            │     │
  │   │  BB period:   [15, 20, 25]                     │     │
  │   │  BB std_dev:  [1.5, 2.0, 2.5]                  │     │
  │   └─────────────────────────────────────────────────┘     │
  │                         │                                 │
  │                         ▼                                 │
  │   ┌─────────────────────────────────────────────────┐     │
  │   │  For each combination:                          │     │
  │   │    1. Apply indicators with these params        │     │
  │   │    2. Run backtest (or score confluence)         │     │
  │   │    3. Record metric (win rate, Sharpe, etc.)    │     │
  │   └─────────────────────────────────────────────────┘     │
  │                         │                                 │
  │                         ▼                                 │
  │   ┌─────────────────────────────────────────────────┐     │
  │   │  Results Matrix                                 │     │
  │   │                                                 │     │
  │   │  SMA(19) + RSI(12):  Sharpe 1.42, Win 64%      │     │
  │   │  SMA(20) + RSI(14):  Sharpe 1.38, Win 62%      │     │
  │   │  SMA(21) + RSI(14):  Sharpe 1.51, Win 66%  ★   │     │
  │   │  SMA(25) + RSI(16):  Sharpe 1.12, Win 58%      │     │
  │   │  ...                                            │     │
  │   │                                                 │     │
  │   │  Best per symbol:                               │     │
  │   │    AAPL daily:  SMA(21) + RSI(14) + BB(20,2.0) │     │
  │   │    NVDA 5min:   SMA(15) + RSI(12) + BB(15,2.5) │     │
  │   │    TSLA daily:  SMA(19) + RSI(16) + BB(25,2.0) │     │
  │   └─────────────────────────────────────────────────┘     │
  │                                                           │
  │   Safeguards:                                             │
  │   • Walk-forward validation (not just in-sample fit)      │
  │   • Minimum sample size per combination                   │
  │   • Penalize complexity (fewer params preferred)          │
  │   • Report robustness: does nearby param also work?       │
  └──────────────────────────────────────────────────────────┘
```

| File | Purpose |
|------|---------|
| `grid.py` | Define parameter grids per indicator, generate combinations |
| `runner.py` | Execute backtest for each param combo, collect results |
| `analyzer.py` | Rank combinations, find best per symbol/timeframe, check robustness |

### 4.3 Confluence Engine (`src/aprilalgo/confluence/`) — NEW v0.2

```
  ┌─────────────────────────────────────────────────┐
  │              Timeframe Aligner                   │
  │                                                  │
  │   5min_df ──┐                                    │
  │             │    ┌──────────────────┐             │
  │  daily_df ──┼───▶│ align timestamps │──▶ merged  │
  │             │    │ (forward-fill,   │   DataFrame │
  │ hourly_df ──┘    │  no look-ahead)  │             │
  │                  └──────────────────┘             │
  └──────────────────────┬──────────────────────────┘
                         │
                         ▼
  ┌─────────────────────────────────────────────────┐
  │              Confluence Scorer                   │
  │                                                  │
  │   For each bar, count how many indicators        │
  │   on how many timeframes agree (BOTH sides):     │
  │                                                  │
  │   ┌────────────────────────────────────────────┐ │
  │   │ Evaluating: LONG direction                 │ │
  │   │                                            │ │
  │   │ Indicator  │ 5min │ hourly │ daily │signal │ │
  │   │────────────│──────│────────│───────│───────│ │
  │   │ DeMark     │  +1  │   0   │  +1   │ BULL  │ │
  │   │ RSI        │  +1  │  +1   │   0   │ BULL  │ │
  │   │ Hurst      │  +1  │  +1   │  +1   │ BULL  │ │
  │   │ BB         │   0  │  +1   │   0   │ BULL  │ │
  │   │ Vol Trend  │  -1  │   0   │  +1   │ BEAR  │ │
  │   │════════════│══════│════════│═══════│═══════│ │
  │   │ Bull score │  3/5 │  3/5  │  3/5  │       │ │
  │   │ Bear score │  1/5 │  0/5  │  0/5  │       │ │
  │   │ Net        │  CONFLUENCE: +0.73 (LONG)     │ │
  │   └────────────────────────────────────────────┘ │
  │                                                  │
  │   Bearish reinforcement acts as a WARNING:       │
  │   "3 indicators say go, but 1 says don't"        │
  └──────────────────────┬──────────────────────────┘
                         │
                         ▼
  ┌─────────────────────────────────────────────────┐
  │            Probability Engine                    │
  │                                                  │
  │   Historical lookup:                             │
  │   "When confluence >= 0.75 in the past,          │
  │    win rate was 68%, avg win 2.1%, avg loss 1.3% │
  │    → suggested stop: 1.5%"                       │
  └─────────────────────────────────────────────────┘
```

**This is the core differentiator.** The confluence module takes multi-timeframe indicator signals and produces a single probability score for each bar.

### 4.4 Backtesting Engine (`src/aprilalgo/backtest/`)

```
  main.py (CLI)
       │
       ▼
  ┌──────────────────────────────────────────────┐
  │              run_backtest()                    │
  │                                               │
  │   for each bar:                               │
  │     1. strategy.on_bar(idx, row, portfolio)   │
  │     2. portfolio.record_equity()              │
  │     3. logger.log_signal() [v0.3]             │
  │                                               │
  │   ┌──────────┐  ┌───────────┐  ┌──────────┐  │
  │   │ Trade    │  │ Portfolio │  │ Metrics  │  │
  │   │ entry/   │  │ cash, pos │  │ Sharpe,  │  │
  │   │ exit,    │  │ equity    │  │ drawdown │  │
  │   │ P&L      │  │ curve     │  │ win rate │  │
  │   └──────────┘  └───────────┘  └──────────┘  │
  │                                               │
  │   ┌──────────────────┐  ┌──────────────────┐  │
  │   │ Position Sizer   │  │  Signal Logger   │  │
  │   │ [v0.2] Kelly,    │  │  [v0.3] persist  │  │
  │   │ fixed %, ATR     │  │  every signal    │  │
  │   └──────────────────┘  └──────────────────┘  │
  └──────────────────────────────────────────────┘
```

**New v0.2:** `position_sizer.py` — Fractional Kelly Criterion, fixed percentage, ATR-based sizing  
**New v0.3:** `logger.py` — Persist every signal with timestamp, probability, SHAP values, outcome

### 4.5 Strategy Framework (`src/aprilalgo/strategies/`)

```
              BaseStrategy (abstract)
              ┌──────────────────┐
              │  init(data)      │
              │  on_bar(idx,row) │
              └────────┬─────────┘
                       │
          ┌────────────┼────────────────┐
          │            │                │
  ┌───────▼──────┐ ┌──▼──────────┐ ┌───▼──────────┐
  │ RsiSmaStrat  │ │ DeMarkConfl │ │ MLStrategy   │
  │ (v0.1) ✅    │ │ (v0.2)      │ │ (v0.3)       │
  │              │ │             │ │              │
  │ RSI oversold │ │ DeMark +    │ │ XGBoost      │
  │ + SMA cross  │ │ Hurst +     │ │ predict() +  │
  │              │ │ multi-TF    │ │ SHAP explain │
  │              │ │ confluence  │ │ + Kelly size │
  └──────────────┘ └─────────────┘ └──────────────┘
```

### 4.6 ML Pipeline (`src/aprilalgo/ml/` + `src/aprilalgo/labels/`) — v0.3

```
  ┌──────────────┐
  │  Price Data  │
  │  + Indicator │
  │  Columns     │
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐    ┌──────────────────┐
  │ Triple       │    │ Feature Matrix   │
  │ Barrier      │    │ Construction     │
  │ Labeling     │    │ (features.py)    │
  │ (labels/)    │    │                  │
  └──────┬───────┘    └──────┬───────────┘
         │                   │
         │    ┌──────────────┘
         │    │
         ▼    ▼
  ┌──────────────┐
  │  Purged      │
  │  K-Fold CV   │
  │  (cv.py)     │
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐    ┌──────────────────┐
  │  XGBoost     │───▶│  SHAP            │
  │  Trainer     │    │  TreeExplainer   │
  │  (trainer.py)│    │  (explain/)      │
  └──────┬───────┘    └──────┬───────────┘
         │                   │
         ▼                   ▼
  ┌──────────────┐    ┌──────────────────┐
  │  Trained     │    │  Human-Readable  │
  │  Model       │    │  Commentary      │
  │  (.json)     │    │  "Buy: DeMark    │
  │              │    │   +0.4, Hurst    │
  │              │    │   +0.2"          │
  └──────────────┘    └──────────────────┘
```

**Key design decisions:**
- Labels come from triple-barrier method (profit target, stop loss, time limit)
- Features are the indicator columns from the registry (no raw OHLCV to avoid leakage)
- CV uses purged k-fold with embargo to prevent temporal leakage
- Model outputs probability → feed to Kelly for position sizing
- SHAP explains each prediction → commentary generator formats text

### 4.7 Explainability (`src/aprilalgo/explain/`) — v0.3

| File | Purpose |
|------|---------|
| `shap_explainer.py` | Wrap `shap.Explainer(model)`, compute SHAP for signals, cache results |
| `commentary.py` | Take SHAP vector + prediction prob → format: "Buy signal (Prob: 0.85). Driven by DeMark exhaustion (SHAP: +0.4)..." |

### 4.8 Meta-Modeling (`src/aprilalgo/meta/`) — v0.4

| File | Purpose |
|------|---------|
| `regime.py` | Detect market regime (trending, mean-reverting, volatile) using HMM or rule-based |
| `model_monitor.py` | Track rolling accuracy of primary XGBoost → reduce size when model degrades |

### 4.9 Streamlit UI (`src/aprilalgo/ui/`) — v0.3

```
  ┌──────────────────────────────────────────────────┐
  │                  Streamlit App                    │
  │                                                   │
  │  ┌────────────┐  ┌─────────────────────────────┐  │
  │  │  Sidebar   │  │    Main Content Area        │  │
  │  │            │  │                             │  │
  │  │ Symbol: ▼  │  │  ┌─────────────────────┐   │  │
  │  │ AAPL       │  │  │  Price Chart        │   │  │
  │  │            │  │  │  + Indicator Overlay │   │  │
  │  │ Timeframe: │  │  │  + Buy/Sell Markers  │   │  │
  │  │ ○ Daily    │  │  └─────────────────────┘   │  │
  │  │ ○ 5min     │  │                             │  │
  │  │            │  │  ┌─────────────────────┐   │  │
  │  │ Strategy:  │  │  │  Signal Feed        │   │  │
  │  │ ▼ DeMark   │  │  │  "Buy AAPL 0.85..."│   │  │
  │  │            │  │  └─────────────────────┘   │  │
  │  │ [Run]      │  │                             │  │
  │  │            │  │  ┌──────┐  ┌──────┐        │  │
  │  │ Metrics:   │  │  │ SHAP │  │Perf. │        │  │
  │  │ Win: 68%   │  │  │Waterf│  │Dash  │        │  │
  │  │ Sharpe:1.4 │  │  │      │  │      │        │  │
  │  │ DD: -8.2%  │  │  └──────┘  └──────┘        │  │
  │  └────────────┘  └─────────────────────────────┘  │
  └──────────────────────────────────────────────────┘
```

---

## 5. Data Flow Diagrams

### 5.1 Complete Data Flow (End State)

```
  Massive API                    CSV Files (data/)
       │                              │
       │  fetcher.py                  │  loader.py
       ▼                              ▼
  ┌───────────────────────────────────────────┐
  │         Raw OHLCV DataFrame               │
  │   (datetime, open, high, low, close, vol) │
  └─────────────────┬─────────────────────────┘
                    │
                    │  resampler.py
                    ▼
  ┌───────────────────────────────────────────┐
  │      Multi-Timeframe DataFrames           │
  │   {"1min": df, "5min": df, "daily": df}   │
  └─────────────────┬─────────────────────────┘
                    │
                    │  IndicatorRegistry.apply()
                    ▼
  ┌───────────────────────────────────────────┐
  │   Enriched DataFrames (per timeframe)     │
  │   OHLCV + rsi + sma + bb + demark +       │
  │   hurst + ehlers + tmi + pv_state + ...   │
  └──────────┬──────────────────┬─────────────┘
             │                  │
             │                  │  features.py
             │                  ▼
             │         ┌────────────────┐
             │         │ Feature Matrix │ (X)
             │         └───────┬────────┘
             │                 │
             │                 │  triple_barrier.py
             │                 ▼
             │         ┌────────────────┐
             │         │ Label Vector   │ (y)
             │         └───────┬────────┘
             │                 │
             │                 │  cv.py + trainer.py
             │                 ▼
             │         ┌────────────────┐
             │         │ XGBoost Model  │
             │         └───────┬────────┘
             │                 │
             │    ┌────────────┤
             │    │            │
             │    ▼            ▼
             │  ┌──────┐  ┌──────────┐
             │  │ SHAP │  │Prediction│
             │  │Values│  │Probability│
             │  └──┬───┘  └────┬─────┘
             │     │           │
             │     ▼           ▼
             │  ┌──────────────────┐
             │  │   Commentary     │
             │  │  Generator       │
             │  └────────┬─────────┘
             │           │
  ┌──────────▼───────────▼───────────────────┐
  │      timeframe_aligner.py                 │
  │      + scorer.py                          │
  │      + probability.py                     │
  └──────────────────┬────────────────────────┘
                     │
                     │  strategy.on_bar()
                     ▼
  ┌───────────────────────────────────────────┐
  │         Backtest Engine                    │
  │   Trade → Portfolio → Equity Curve         │
  └──────────────────┬────────────────────────┘
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
    ┌──────────┐ ┌────────┐ ┌──────────┐
    │ Metrics  │ │ Signal │ │ Streamlit│
    │ Report   │ │  Log   │ │   UI     │
    └──────────┘ └────────┘ └──────────┘
```

### 5.2 v0.2 Scope (What We Build Next)

```
  Massive API ──fetcher.py──▶ CSV
                                │
  CSV ──loader.py──▶ DataFrame ─┤
                                │
  resampler.py ◀────────────────┘
       │
       ▼
  Multi-TF DataFrames
       │
       │  NEW indicators:
       │  demark.py, hurst.py, ehlers.py, tmi.py, pv_sequences.py
       ▼
  Enriched DataFrames
       │
       │  NEW confluence module:
       │  timeframe_aligner.py + scorer.py + probability.py
       ▼
  Confluence Score + Suggested Stop
       │
       │  NEW strategy: demark_confluence.py
       │  NEW: position_sizer.py (Kelly)
       ▼
  Backtest Engine (existing, enhanced)
       │
       ▼
  Metrics (existing)
```

---

## 6. Dependency Map

### Current (v0.1)
```
pandas, numpy, matplotlib, jupyter, pyyaml
```

### v0.2 (add)
```
massive          # Data fetching (Polygon.io → Massive.com)
scipy            # Hurst exponent calculation (R/S analysis)
```

### v0.3 (add)
```
xgboost          # ML signal classifier
shap             # Explainability
scikit-learn     # Preprocessing, CV utilities
streamlit        # UI dashboard
plotly           # Interactive charts in Streamlit
```

### v0.4 (add)
```
hmmlearn         # Hidden Markov Model for regime detection (optional)
```

---

## 7. External Library Integration Plan

| Library | Version | How We Use It | Integration Point |
|---------|---------|---------------|-------------------|
| `massive` | v0.2 | `fetcher.py` calls `RESTClient.list_aggs()` | `src/aprilalgo/data/fetcher.py` |
| `scipy` | v0.2 | `stats` module for Hurst R/S calculation | `src/aprilalgo/indicators/hurst.py` |
| `xgboost` | v0.3 | `XGBClassifier` trained on feature matrix | `src/aprilalgo/ml/trainer.py` |
| `shap` | v0.3 | `shap.Explainer(model)` + plots | `src/aprilalgo/explain/shap_explainer.py` |
| `scikit-learn` | v0.3 | Preprocessing, custom CV splitter | `src/aprilalgo/ml/cv.py`, `features.py` |
| `streamlit` | v0.3 | UI framework | `src/aprilalgo/ui/app.py` |
| `plotly` | v0.3 | Interactive charts in Streamlit | `src/aprilalgo/ui/pages/charts.py` |
| `legitindicators` | REFERENCE | Math formulas for Hurst, Ehlers — port, don't import | `src/aprilalgo/indicators/*.py` |

---

## 8. Design Principles

### 8.1 No Look-Ahead Bias
- All indicators use only past and current data (rolling windows, no future peeking)
- Multi-timeframe alignment uses **forward-fill only** (higher TF value persists until next bar)
- ML features are computed on training window only; test window never touches training
- Purged CV ensures label overlap doesn't leak between folds

### 8.2 Pure Functions for Indicators
- Every indicator is `DataFrame → DataFrame` (adds columns, never removes)
- No side effects, no global state
- Registry applies them in sequence

### 8.2b Dual-Signal Reinforcement
- Every indicator state carries BOTH positive and negative reinforcement simultaneously
- Example: RSI < 30 is BOTH momentum-bearish (strong downtrend) AND mean-reversion-bullish (bounce likely)
- A single indicator can never give a definitive answer — it always has competing interpretations
- The confluence engine resolves the ambiguity by weighing what ALL indicators say together
- This prevents one-sided analysis — you always see the case FOR and AGAINST

### 8.2c Parameter Sensitivity
- Indicator parameters (e.g. SMA period, RSI threshold) are never assumed optimal
- The tuner tests parameter grids and ranks combinations by backtest performance
- Best parameters vary by symbol, timeframe, and market regime
- Walk-forward validation prevents overfitting to historical data
- Robustness check: if SMA(20) works but SMA(19) and SMA(21) don't, the result is fragile

### 8.3 Strategy as Event Handler
- Strategies implement `on_bar()` — called once per bar
- Strategy can read past bars (via DataFrame) but not future bars
- Strategy communicates with Portfolio through `open_position()` / `close_position()`

### 8.4 Separation of Concerns
```
  Data     →  Features  →  Signals  →  Execution  →  Analysis
  (data/)    (indicators/) (strategies/) (backtest/)   (metrics/)
                           (confluence/) (portfolio/)  (explain/)
                           (ml/)                       (ui/)
```

### 8.5 Configuration over Code
- YAML configs for strategy parameters, symbol lists, backtest settings
- No hardcoded values in modules
- Override via CLI flags

### 8.6 Reproducibility
- Every backtest run can be reproduced from: config YAML + data CSVs + code version
- Signal logs persist full context (features, SHAP, prediction, outcome)
- `uv.lock` pins all dependency versions

---

## Appendix: Key File Ownership by Version

| Version | New Files |
|---------|-----------|
| **v0.1** ✅ | `data/{loader,store,resampler}.py`, `indicators/{registry,rsi,sma,bollinger,volume_trend}.py`, `backtest/{engine,trade,portfolio,metrics}.py`, `strategies/{base,rsi_sma}.py`, `config.py`, `main.py`, `configs/default.yaml` |
| **v0.2** | `data/{fetcher,universe}.py`, `indicators/{demark,hurst,ehlers,tmi,pv_sequences}.py`, `confluence/{timeframe_aligner,scorer,probability}.py`, `tuner/{grid,runner,analyzer}.py`, `backtest/position_sizer.py`, `strategies/demark_confluence.py`, `scripts/fetch_data.py`, `configs/strategies/*.yaml` |
| **v0.3** | `labels/{triple_barrier,reinforcement}.py`, `ml/{features,trainer,cv}.py`, `explain/{shap_explainer,commentary}.py`, `strategies/ml_strategy.py`, `backtest/logger.py`, `ui/{app,pages/*,components/*}.py` |
| **v0.4** | `labels/meta_label.py`, `ml/{sampling,importance}.py`, `meta/{regime,model_monitor}.py` |
