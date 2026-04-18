# CLAUDE.md — Context for Claude AI

This file provides Claude with everything it needs to understand and work effectively in the AprilAlgo project.

---

## Project Overview

**AprilAlgo** is a modular, multi-timeframe backtesting system for stock trading strategies, built in Python. The system applies technical indicators, scores multi-indicator confluence, tunes parameters, and measures performance with professional-grade metrics.

- **Language:** Python 3.11+
- **Package manager:** `uv` (replaces pip + venv)
- **License:** Apache 2.0
- **Owner:** Joshua
- **Current version:** 0.4.1 (ML + meta/regime/reporting + SHAP + information bars baseline)

---

## v0.3+ development notes

- **Interface map:** `docs/DATA_SCHEMA.md` defines Raw OHLCV → Enriched Features → Confluence → Backtest metrics → ML labels (triple-barrier). Verify column names and types there before integration code.
- **ML CLI:** `uv run python -m aprilalgo.cli train|evaluate|oof|predict|importance|shap|walk-forward|bars --config configs/ml/default.yaml` (plus `--model-dir` / `--output` where applicable)
- **ML bars:** optional `information_bars` block in the ML YAML — `load_ohlcv_for_ml` in `data/loader.py` builds the series used for triple-barrier, features, and walk-forward `n_bars`; recipe is stored in `meta.json` for inference and `ml_xgboost`.
- **Agent governance:** `.cursorrules` — Model Routing Protocol first; isolation of legacy indicator/backtest internals when only contracts matter; no look-ahead; schema-first; documentation loop after substantive changes.
- **Token discipline:** Prefer `AGENTS.md`, `docs/MODEL_ROUTING.md`, and `docs/DATA_SCHEMA.md` over bulk source reads.

---

## Project Structure

```
AprilAlgo/
├── src/aprilalgo/              # Main Python package
│   ├── __init__.py             # Version and top-level exports
│   ├── config.py               # YAML config loader
│   ├── labels/                 # ML labeling (v0.3)
│   │   └── triple_barrier.py   # Triple-barrier targets from OHLC
│   ├── ml/                     # ML utilities (v0.3)
│   │   ├── features.py         # Feature matrix: registry columns only, no OHLCV
│   │   └── cv.py               # PurgedKFold + learning_matrix (t0/t1 for CV)
│   ├── data/                   # Data layer
│   │   ├── loader.py           # Load CSVs by symbol + timeframe
│   │   ├── fetcher.py          # Fetch from Massive API (ex-Polygon.io)
│   │   ├── store.py            # CSV/pickle I/O helpers
│   │   ├── resampler.py        # Resample OHLCV to different timeframes
│   │   └── universe.py         # Symbol watchlist management
│   ├── indicators/             # Technical indicators (all emit bull/bear signals)
│   │   ├── descriptor.py       # IndicatorSpec/ParamSpec — single source of truth
│   │   ├── registry.py         # Pluggable indicator pipeline + catalog lookup
│   │   ├── rsi.py              # RSI + oversold/overbought signals
│   │   ├── sma.py              # SMA + price-above/below signals
│   │   ├── bollinger.py        # Bollinger Bands + band-touch signals
│   │   ├── volume_trend.py     # Volume expansion + direction confirmation
│   │   ├── demark.py           # TD Sequential Setup + Countdown
│   │   ├── hurst.py            # Hurst Exponent (trend persistence)
│   │   ├── ehlers.py           # Super Smoother, Roofing Filter, Decycler
│   │   ├── tmi.py              # Turn Measurement Index (curvature)
│   │   └── pv_sequences.py     # Price-Volume state machine
│   ├── confluence/             # Multi-timeframe confluence scoring
│   │   ├── timeframe_aligner.py  # Align higher-TF signals to base TF
│   │   ├── scorer.py           # Tally bull/bear signals → net confluence
│   │   └── probability.py      # Historical win-rate by confluence level
│   ├── tuner/                  # Parameter optimization
│   │   ├── grid.py             # Define param ranges, generate combos
│   │   ├── runner.py           # Run backtest per combo, collect metrics
│   │   └── analyzer.py         # Rank combos, robustness checks
│   ├── backtest/               # Backtesting engine
│   │   ├── engine.py           # Bar-by-bar simulation loop
│   │   ├── trade.py            # Trade dataclass
│   │   ├── portfolio.py        # Cash, positions, equity tracking
│   │   ├── metrics.py          # Performance metrics (Sharpe, drawdown, etc.)
│   │   └── position_sizer.py   # Fractional Kelly, fixed %, ATR-based
│   ├── strategies/             # Trading strategies
│   │   ├── base.py             # Abstract base strategy
│   │   ├── rsi_sma.py          # RSI + SMA crossover
│   │   ├── demark_confluence.py # DeMark + multi-indicator confluence
│   │   └── configurable.py     # Config-driven strategy (any indicator combo)
│   └── ui/                     # Streamlit UI
│       ├── app.py              # Main entry point + page routing
│       ├── helpers.py          # Shared UI utilities
│       └── pages/              # One file per page
│           ├── charts.py       # Candlestick + indicator overlays
│           ├── signals.py      # Signal feed with confluence scores
│           ├── dashboard.py    # Backtest metrics + equity curve
│           └── tuner.py        # Parameter optimization UI
├── tests/                      # Persistent pytest test suite (39 tests)
├── data/                       # OHLCV CSV files (73 symbols, gitignored)
│   ├── daily_data/
│   └── 5min_data/
├── configs/
│   └── default.yaml            # Default backtest settings
├── scripts/
│   └── fetch_data.py           # Bulk download via Massive API
├── docs/                       # Reference documentation
│   ├── DATA_SCHEMA.md          # Column / layer contracts (interface map)
│   ├── TRIPLE_BARRIER_MATH.md  # Barrier definitions + same-bar policy
│   ├── HANDOFF.md
│   ├── LEARNING.md
│   └── REPO_ANALYSIS.md
├── main.py                     # CLI entry point
├── pyproject.toml
├── uv.lock
├── ARCHITECTURE.md             # Full system design document
└── CHANGELOG.md
```

---

## Key Commands

| Task                           | Command                                              |
|--------------------------------|------------------------------------------------------|
| Install all dependencies       | `uv sync`                                            |
| Run default backtest           | `uv run python main.py`                              |
| Run DeMark confluence          | `uv run python main.py --strategy demark_confluence`  |
| Launch Streamlit UI            | `uv run streamlit run src/aprilalgo/ui/app.py`        |
| Run tests                      | `uv run pytest tests/ -v`                            |
| Fetch data from Massive API    | `uv run python scripts/fetch_data.py --symbols AAPL,NVDA` |
| Add a new package              | `uv add <package>`                                   |

---

## Architecture Patterns

### Indicator Descriptor System
Every indicator is registered in `indicators/descriptor.py` with an `IndicatorSpec` that provides its display name, parameter ranges, category, and overlay flag. The UI and tuner auto-generate controls from this — **no hardcoded indicator lists anywhere**.

### Parameterized Column Names
All indicator columns include their parameters: `rsi_14`, `sma_20_bull`, `bb_20_upper`, `ss_10_bear`. This prevents collision when the same indicator is called with different parameters.

### Dual-Signal Indicators
Every indicator emits BOTH bullish and bearish signals. The same state carries positive reinforcement for one direction and negative for the other. The confluence engine resolves the ambiguity.

### Confluence Scoring
The confluence module tallies bull/bear signals across all indicators and timeframes, producing a net score from -1.0 (all bearish) to +1.0 (all bullish).

### Strategy as Event Handler
Strategies implement `on_bar()` — called once per bar during simulation. No look-ahead.

---

## Adding New Components

### New Indicator
1. Create `src/aprilalgo/indicators/my_indicator.py`
2. Write a function: `def my_indicator(df, period=14, ...) -> pd.DataFrame`
3. Column names MUST include parameters: `myind_{period}_bull`, not `myind_bull`
4. Export it in `indicators/__init__.py`
5. Register it in `indicators/descriptor.py` with an `IndicatorSpec` entry

### New Strategy
1. Create `src/aprilalgo/strategies/my_strategy.py`
2. Subclass `BaseStrategy`, implement `init()` and `on_bar()`
3. Register it in `strategies/__init__.py` in the `STRATEGIES` dict
4. Or use `ConfigurableStrategy` with a custom indicator list — no new class needed

---

## Coding Conventions

- Use `uv add` to add dependencies — never edit pyproject.toml by hand
- Always run scripts with `uv run`
- Follow PEP 8 style, lines under 100 characters
- Use type hints for all function signatures
- Use `pathlib.Path` for file paths
- Commit `uv.lock` to version control
- Data CSV files in `data/` are NOT committed (gitignored)
