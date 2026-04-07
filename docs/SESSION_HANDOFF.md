# AprilAlgo — Session Handoff Document

> **Date:** April 5, 2026
> **For:** Joshua (project owner)
> **Purpose:** Everything you need to pick up where we left off, explain the project to someone else, or start a new AI session.

---

## What Is AprilAlgo?

AprilAlgo is a stock trading backtesting system you're building. The idea is simple: take a bunch of technical indicators, figure out when they agree with each other (called "confluence"), and calculate the probability that a trade setup actually works. If most indicators say "buy" at the same time, across multiple timeframes, that's a high-confidence signal.

The long-term vision includes machine learning that explains *why* it thinks a trade will work (not a black box), but right now the foundation is rule-based indicators and confluence scoring.

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

### The Interior (v0.2 → Unreleased) — Done
- **Indicator descriptor system**: every indicator self-describes its parameters, display name, and category. The UI and tuner auto-generate controls from this — adding a new indicator requires zero UI code changes.
- **Configurable strategy**: instead of writing Python code for each strategy, you can pick which indicators to use from a dropdown and the system builds the strategy automatically
- **Streamlit UI**: a 4-page web dashboard you can open in your browser:
  - **Charts**: interactive candlestick charts with any indicator overlay, adjustable parameters
  - **Signal Feed**: shows the confluence score for each bar — which signals fired, how strong
  - **Dashboard**: run a backtest and see metrics, equity curve, trade log, buy/sell markers on the chart
  - **Parameter Tuner**: sweep parameter ranges, find the best combo, check if it's robust or fragile
- **Test suite**: 39 automated tests that verify everything works (run with `uv run pytest tests/ -v`)

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
│   └── ui/                  # Streamlit web dashboard (4 pages)
├── tests/                   # 39 automated tests
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

## The 3 Strategies

| Strategy | How it works |
|----------|--------------|
| **RSI + SMA** | Buy when RSI is oversold AND price is above the SMA. Sell when RSI is overbought OR price drops below SMA. |
| **DeMark Confluence** | Buy when DeMark signals exhaustion AND confluence score confirms. Sell on reversal signal, confluence flip, or stop loss. |
| **Configurable** | You pick which indicators to use from a list, set parameters, and it trades based on the total confluence score. No code changes needed. |

---

## Key Design Decisions Worth Knowing

1. **Dual-signal principle**: RSI < 30 is simultaneously "bullish" (mean reversion: bounce likely) AND "bearish" (momentum: downtrend continuing). The confluence engine resolves which interpretation wins by looking at what all the other indicators say.

2. **Parameterized columns**: `rsi_14_bull` not `rsi_bull`. This was a bug fix — calling RSI with period 14 then period 7 used to silently overwrite the first result. Now both coexist.

3. **Descriptor catalog**: instead of hardcoding indicator lists in 5 places (UI charts, UI signals, UI dashboard, UI tuner, strategies), there's one registry in `descriptor.py` that everything reads from.

4. **No look-ahead bias**: the backtester processes one bar at a time. Strategies can only see the current bar and past bars, never future bars.

---

## What's NOT Built Yet (The Roadmap)

### v0.3 — Machine Learning + Explainability
This is the next major milestone. The goal is to go beyond rule-based trading to ML-driven signals that explain themselves.

| Component | What it does | Status |
|-----------|-------------|--------|
| **Triple-barrier labeling** | Classifies each bar as win/loss/timeout based on future price hitting profit target, stop loss, or expiring. This creates the training labels for ML. | Not started |
| **Feature matrix builder** | Converts all indicator values into a feature table that XGBoost can train on. Must avoid data leakage. | Not started |
| **XGBoost classifier** | Trains a model to predict high-probability setups based on indicator features. | Not started |
| **Purged k-fold CV** | Cross-validation that respects time series ordering (no training on future data). Required for honest performance estimates. | Not started |
| **SHAP integration** | Uses TreeExplainer to show which indicators drove each trade signal and by how much. | Not started |
| **Intelligent Commentary** | Translates SHAP values into plain English: "Buy signal (85% confidence). Driven by DeMark exhaustion (+0.4) and Hurst trend alignment (+0.2)." | Not started |
| **Signal logging** | Save every signal with its features, SHAP values, and eventual outcome to a database/CSV for analysis. | Not started |
| **SHAP plots UI page** | Waterfall, force, and beeswarm plots in the Streamlit dashboard for individual trade analysis. | Not started |

**Key libraries to use:** `xgboost`, `shap`, `scikit-learn`
**Key reference:** `docs/REPO_ANALYSIS.md` has detailed notes on which GitHub repos to learn from

### v0.4 — Meta-Modeling + Advanced Features
The system learns to predict when it's about to be wrong.

| Component | What it does |
|-----------|-------------|
| **Meta-labeling** | A secondary model that predicts if the primary XGBoost model is about to make a mistake (reduces bet size in bad conditions) |
| **Regime detection** | Detects whether the market is in a trending, volatile, or calm regime. Switches strategy behavior accordingly. |
| **Information-driven bars** | Instead of time-based bars (every 5 minutes), create bars based on volume, dollar value, or tick count for better signal quality. |
| **Walk-forward optimization** | Roll the tuner forward through time: train on 2020-2022, test on 2023, slide forward. Prevents overfitting. |
| **Multi-asset portfolio** | Run strategies across all 73 symbols simultaneously with portfolio-level risk management. |
| **PDF/HTML reports** | Export backtest results as shareable reports. |

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

These are independent paths. Pick whichever sounds most useful:

### Path A: "Use What We Built"
Run the DeMark Confluence strategy and Configurable strategy across all 73 symbols. Use the Parameter Tuner to find optimal settings per symbol. See which symbols the system works best on and which it fails on. This builds intuition before adding ML.

### Path B: "Get More Data"
Set up your Massive.com API key and download longer price histories (2020-present). The current data may be limited. More data = better backtests = more meaningful tuner results.

### Path C: "Start v0.3 ML Pipeline"
Begin implementing triple-barrier labeling and the XGBoost pipeline. This is the biggest leap in capability but also the most complex work.

### Path D: "Improve the UI"
The Streamlit dashboard works but could be polished: add a SHAP plots placeholder page, improve chart interactions, add multi-symbol comparison views, add export-to-CSV buttons.

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
