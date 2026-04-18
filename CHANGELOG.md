# Changelog

All notable changes to AprilAlgo will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- **Primary OOF (v0.5 Sprint 3)** — `aprilalgo.ml.oof.compute_primary_oof` (purged k-fold stacked predictions); CLI `oof` writes `oof_primary.csv` and merges `oof.path` into existing `meta.json`; `tests/test_oof.py`, `tests/test_cli_ml.py::test_cli_oof_writes_csv`
- **Optional `hmm` extra** — `[project.optional-dependencies] hmm` with `hmmlearn` for `add_vol_regime(use_hmm=True)`; `uv sync --extra hmm` when wheels exist for your Python
- **Sampling strategies (v0.5 Sprint 2)** — CLI `_weights_for_training` supports `sampling.strategy: uniqueness` (overlap weights) and `bootstrap` (multiplicity weights from `sequential_bootstrap_sample`); `meta.json.sampling` records `random_state` / `n_draw` for bootstrap; tests in `tests/test_sampling.py`, `tests/test_cli_ml.py`
- **Sample weights (v0.5 Sprint 1)** — `train_xgb_classifier(..., sample_weight=...)`; CLI `_weights_for_training` stub (uniform when `sampling` absent); `meta.json` key `sampling` with default `strategy: none`; tests `tests/test_trainer_sample_weight.py`
- **ML pipeline information bars** — optional YAML `information_bars` (`enabled`, `bar_type`, `threshold`, optional `source_timeframe`) applies tick/volume/dollar aggregation after `load_price_data` and before triple-barrier + features (`data/loader.py` `load_ohlcv_for_ml`, `data/bars.py` `apply_information_bars_from_config`)
- **`meta.json`** — persists `information_bars` recipe for predict/SHAP and `ml_xgboost` replay; backtest engine iterates bar-aligned rows when the strategy exposes `_backtest_bars_df`
- **`tests/test_loader_ml_bars.py`**, **`tests/test_cli_ml.py`** — coverage for bar-enabled train/predict

### Planned
- Full SHAP commentary UX polish
- Additional information-driven bars research and tuning presets
- Walk-forward UI deep analytics polish

---

## [0.4.1] - 2026-04-17

### Added
- **`src/aprilalgo/ml/explain.py`** — SHAP values + SHAP importance tables for saved model bundles; **`tests/test_shap.py`**
- **CLI** — new `shap` subcommand writing `shap_values.csv` and `shap_importance.csv`
- **`src/aprilalgo/data/bars.py`** — information-driven bar builders (`tick`, `volume`, `dollar`); **`tests/test_data_bars.py`**
- **CLI** — new `bars` subcommand for building information-driven bars from OHLCV CSV
- **Walk-forward summary** in CLI JSON (`summary.n_splits`, `coverage_pct`, mean sizes) and per-fold `test_return`

### Changed
- **`src/aprilalgo/reporting/report.py`** — new HTML sections: `section-shap`, `section-walk-forward`
- **`src/aprilalgo/ui/pages/model_trainer.py`** — added SHAP button/action
- **`src/aprilalgo/ui/pages/model_metrics.py`** — richer metrics presentation (F1 display, ROC quick view, latest importance chart, proxy equity curve)
- **`src/aprilalgo/ui/pages/walk_forward_lab.py`** — summary + fold dataframe + chart + CSV export
- **`docs/DATA_SCHEMA.md`** — explainability artifacts, bars schema, walk-forward output schema
- **`tests.md`** — inventory regenerated for latest test count

### Dependencies
- Added **`shap`** runtime dependency.

---

## [0.4.0] - 2026-04-16

### Added
- **`src/aprilalgo/labels/meta_label.py`** — meta-label construction + purged meta logistic; **`tests/test_meta_label.py`**
- **`src/aprilalgo/ml/sampling.py`** — overlap / uniqueness weights; **`tests/test_sampling.py`**
- **`src/aprilalgo/meta/regime.py`** — realized vol + `vol_regime` quantile buckets; **`tests/test_regime.py`**
- **`src/aprilalgo/tuner/walk_forward.py`** — walk-forward index splits; **`tests/test_walk_forward.py`**
- **`src/aprilalgo/backtest/portfolio_runner.py`** — multi-symbol backtest helper
- **`src/aprilalgo/reporting/report.py`** — Jinja2 HTML report; **`tests/test_reporting.py`**
- **Streamlit** — `ML lab`, `Regime lab`, `Portfolio lab` pages
- **`jinja2`** dependency

---

## [0.3.0] - 2026-04-16

### Added
- **`src/aprilalgo/labels/targets.py`** — unified triple-barrier target columns; **`tests/test_targets.py`**
- **`src/aprilalgo/ml/trainer.py`** — XGBoost train/save/load via Booster bundle (`meta.json` + `xgboost.json`); **`tests/test_trainer.py`**
- **`src/aprilalgo/ml/evaluator.py`** — purged CV metrics; **`tests/test_evaluator.py`**
- **`src/aprilalgo/ml/importance.py`** — gain + permutation importance
- **`src/aprilalgo/backtest/logger.py`** — JSONL signal logger; **`tests/test_logger.py`**
- **`src/aprilalgo/strategies/ml_strategy.py`** — `ml_xgboost` strategy; registered in **`STRATEGIES`**
- **`src/aprilalgo/cli.py`** — `train` / `evaluate` / `importance` subcommands; **`configs/ml/default.yaml`**
- **`tests/fixtures/daily_data/TEST_daily.csv`** — small OHLCV fixture for ML smoke tests
- **`xgboost`**, **`scikit-learn`** dependencies
- **`docs/MODEL_ROUTING.md`** — Cursor model tier routing; **`.cursorrules`** Model Routing Protocol

### Changed
- **`README.md`**, **`AGENTS.md`**, **`docs/DATA_SCHEMA.md`**, **`docs/SESSION_HANDOFF.md`** — aligned with ML pipeline
- **`tests.md`** — regenerate after test count changes

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
