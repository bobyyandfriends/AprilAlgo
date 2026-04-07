# Changelog

All notable changes to AprilAlgo will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Planned
- XGBoost ML model + SHAP explainability — v0.3
- Triple-barrier labeling + purged k-fold CV — v0.3
- Meta-labeling + regime detection — v0.4

### Added
- **Indicator descriptor system** (`indicators/descriptor.py`): `IndicatorSpec` and `ParamSpec` dataclasses — single source of truth for indicator metadata (name, params, ranges, category, overlay flag). The UI and tuner auto-generate controls from this instead of hardcoding.
- **Catalog function** `get_catalog()` — returns all 11 registered indicators with full metadata
- **`IndicatorRegistry.add_by_name()`** and **`IndicatorRegistry.from_config()`** — build indicator pipelines from catalog names or YAML-style config dicts
- **`ConfigurableStrategy`** (`strategies/configurable.py`): compose any indicator combination via config, no new Python class needed. Trades on confluence score.
- **Streamlit UI** (`ui/`): 4-page app (Charts, Signal Feed, Dashboard, Parameter Tuner) with interactive Plotly charts, auto-generated parameter sliders from descriptor catalog
- **Persistent test suite** (`tests/`): 39 pytest tests covering indicators, backtest, confluence, and tuner

### Changed
- **BREAKING: indicator column names now include parameters** — `rsi_bull` → `rsi_14_bull`, `sma_bull` → `sma_20_bull`, `bb_bull` → `bb_20_bull`, etc. This prevents column collision when calling the same indicator with different parameters (e.g., SMA(20) + SMA(50))
- All 4 UI pages now pull indicator options and parameter controls from the descriptor catalog instead of hardcoded lists
- Signals page `_enrich_all()` now applies every registered indicator automatically via `get_catalog()`
- Moved `HANDOFF.md`, `LEARNING.md`, `REPO_ANALYSIS.md` to `docs/` folder (root declutter)
- Removed empty `ui/components/` package
- Corrected dual-signal reinforcement model in ARCHITECTURE.md

---

## [0.2.0] - 2026-04-04

### Added
- **Data fetcher** (`data/fetcher.py`): download OHLCV from Massive.com API (ex-Polygon.io)
- **Universe manager** (`data/universe.py`): load symbol watchlists from YAML
- **Bulk download script** (`scripts/fetch_data.py`): CLI for fetching multiple symbols
- **DeMark TD Sequential** (`indicators/demark.py`): Setup (1-9) + Countdown (1-13) with buy/sell signals
- **Hurst Exponent** (`indicators/hurst.py`): R/S analysis across multiple rolling windows
- **Ehlers indicators** (`indicators/ehlers.py`): Super Smoother, Roofing Filter, Decycler
- **TMI** (`indicators/tmi.py`): curvature-based trend change detection
- **PV Sequences** (`indicators/pv_sequences.py`): Price-Volume state machine (PU_VU, PU_VD, PD_VU, PD_VD)
- **Dual-signal principle**: all indicators now emit both `_bull` and `_bear` signal columns
- **Confluence engine** (`confluence/`): align multi-timeframe signals, score agreement, compute historical probabilities
- **Parameter tuner** (`tuner/`): grid search over indicator params with robustness analysis
- **Position sizing** (`backtest/position_sizer.py`): Fractional Kelly, fixed fraction, ATR-based
- **DeMark Confluence strategy** (`strategies/demark_confluence.py`): DeMark exhaustion + multi-indicator confirmation
- **Architecture document** (`ARCHITECTURE.md`): full end-state system design through v0.4
- **Repo analysis** (`REPO_ANALYSIS.md`): research on 8 external GitHub repos for integration
- `massive` and `scipy` dependencies

### Changed
- RSI, SMA, Bollinger Bands, Volume Trend now emit `_bull`/`_bear` columns
- Bollinger Bands adds `bb_pct` (%B position within bands)
- Volume Trend now differentiates bullish vs bearish volume expansion

---

## [0.1.1] - 2026-04-04

### Added
- **Data layer** (`src/aprilalgo/data/`): loader, store, resampler — ported and cleaned from symbolikai
- **Indicator engine** (`src/aprilalgo/indicators/`): RSI, SMA, Bollinger Bands, Volume Trend with pluggable registry
- **Backtesting engine** (`src/aprilalgo/backtest/`): bar-by-bar simulation with Trade, Portfolio, and unified Metrics
- **Strategy framework** (`src/aprilalgo/strategies/`): abstract BaseStrategy + RSI/SMA crossover strategy
- **CLI entry point** (`main.py`): run backtests from the command line with `--symbol`, `--timeframe`, `--strategy`
- **YAML config** (`configs/default.yaml`): configurable capital, commission, slippage, strategy params
- 73 symbols of daily and 5-minute OHLCV data (from symbolikai/Polygon.io)
- `pyyaml` dependency

### Fixed
- Unified metrics module (symbolikai had two competing implementations)
- Backtester now does proper bar-by-bar simulation (symbolikai was post-processing only)
- Strategies now use `on_bar()` pattern instead of generating all trades at once

---

## [0.1.0] - 2026-03-29

### Added
- Initial project structure using `uv` package manager
- `pyproject.toml` with project metadata and dependencies
- Virtual environment setup with `uv sync`
- Core data science dependencies: `pandas`, `numpy`, `matplotlib`, `jupyter`
- `src/aprilalgo/__init__.py` package entry point
- `README.md` with installation and usage instructions
- `LICENSE` (Apache 2.0)
- `.gitignore` based on the official GitHub Python template
- `CLAUDE.md` with AI assistant context
- `AGENTS.md` with AI agent coding conventions
- `CHANGELOG.md` (this file)

[Unreleased]: https://github.com/bobyyandfriends/AprilAlgo/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/bobyyandfriends/AprilAlgo/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/bobyyandfriends/AprilAlgo/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/bobyyandfriends/AprilAlgo/releases/tag/v0.1.0
