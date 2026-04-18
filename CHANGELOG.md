# Changelog

All notable changes to AprilAlgo will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Changed
- **Documentation consolidation (post-v0.5 audit, 2026-04-18)** — reduced markdown from 17 → 11 files so a normal coding session updates only two (`PROJECT_STATE.md` + `CHANGELOG.md`). New unified **`AGENTS.md`** absorbs `CLAUDE.md` structural content, `COMMAND_CHEAT_SHEET.md`, `docs/MODEL_ROUTING.md`, the end-of-session handoff protocol from `docs/SESSION_HANDOFF.md`, and a new §6 "Testing" section (catalog of 28 test files, pytest invocations, optional `hmm` extra). **`CLAUDE.md`** slimmed to a thin pointer → `AGENTS.md`. **`PROJECT_STATE.md`** rewritten lean (§1 status, §2 today's handoff, §3 next sprint, §4 warning zone seeded from `AUDIT_FINDINGS.md` §B, §5 operational contracts, §6 open questions, Appendix A collapsed sprint log, Appendix B CLI edge cases absorbed from `DEBUG_LOG.md`). **`ARCHITECTURE.md`** §2 "Versioned Roadmap" trimmed — shipped per-sprint history deferred to `CHANGELOG.md`, kept forward-looking v0.6+ themes only. **`README.md`** doc links repointed. **`.cursorrules`** now references `AGENTS.md` §7 for model routing. `docs/LEARNING.md` and `docs/REPO_ANALYSIS.md` moved to `docs/archive/`. Deleted: `BACKLOG.md`, `COMMAND_CHEAT_SHEET.md`, `DEBUG_LOG.md`, `tests.md`, `docs/SESSION_HANDOFF.md`, `docs/MODEL_ROUTING.md`.

### Fixed
- **Principal-Staff audit bug hunt (2026-04-18)** — 14 definitive correctness defects fixed across `backtest/`, `indicators/`, `ml/`, `labels/`, `tuner/`, `ui/`, and `reporting/`. Full detail in `AUDIT_FINDINGS.md`; test suite green at 183 passed / 1 skipped.
  - **A1 (CRITICAL)** `backtest/portfolio.py::record_equity` no longer double-counts short positions; equity snapshot equals `cash + MTM` for both sides.
  - **A2 (HIGH)** `backtest/portfolio.py::close_trade` now accumulates exit-side slippage + commission onto `Trade` before `Trade.close`, so `trade.realized_pnl` (and every derived metric) matches actual cash balance. Replaced `list.remove` with identity-match loop to avoid `dataclass.__eq__` collisions on open-positions tracking.
  - **A3 (CRITICAL)** `indicators/pv_sequences.py` + `indicators/tmi.py` early-return empty DataFrames with typed expected columns (was: `IndexError` on `price_up[0] = False` when upstream dropped all rows).
  - **A4 (HIGH)** `indicators/demark.py` gates countdown activation and the "weak completed-setup" signal on the `<9 → 9` transition (was: countdown could never reach 13 if the setup stayed at 9, and `td_bull` fired on every bar during a standing 9).
  - **A5 (CRITICAL, ML correctness)** `ml/oof.py::compute_primary_oof` now builds the OOF probability matrix on a global class axis computed from `np.unique(y)` and maps each fold's `est.classes_` onto it (was: silent column-vs-class misalignment when a fold's training block excluded a class under `task="multiclass"`).
  - **A6 (CRITICAL, ML correctness)** `labels/meta_label.py::build_meta_labels` returns `np.float64` with `NaN` for undefined rows; `fit_meta_logit_purged` restricts inner CV + final fit to the valid-mask and re-expands `meta_oof`/`z` with NaN (was: `NaN == anything == False` labeled every purged-empty row as "primary incorrect", biasing the meta dataset).
  - **A7 (HIGH)** `ml/explain.py::_shap_matrix` list-branch now applies `np.abs` before averaging, so multiclass SHAP importance no longer cancels `+0.3`/`-0.3` across classes to 0.
  - **A8 (HIGH)** `tuner/ml_walk_forward.py::ml_walk_forward_tune` uses `copy.deepcopy(cfg)` (was: `json.loads(json.dumps(cfg))` crashed on `datetime` / `Path` / numpy scalars in YAML configs).
  - **A9 (HIGH)** `tuner/analyzer.py::_check_robustness` degradation formula is now `(best - neighbor) / abs(best)` so robustness semantics hold for negative metrics (was: inverted for negative Sharpe).
  - **A10 (HIGH)** `ml/evaluator.py` replaces bare `except Exception` with `except (ValueError, IndexError)` and adds an explicit `len(np.unique(y_tr)) < 2` guard before `predict_proba[:, 1]` (was: crashed on single-class training folds and swallowed unrelated bugs).
  - **A11 (HIGH, UI)** `ui/helpers.py::format_metric` guards `None` / `NaN` and returns `"—"`; wraps numeric-format fallbacks in `(TypeError, ValueError)` so the UI never surfaces a raw traceback.
  - **A12 (HIGH, UI)** `ui/pages/signals.py` adds `_safe_int` and a `pd.isna` guard on `datetime`/`close`/counts so warm-up rows are skipped cleanly (was: `int(NaN)` crashed the Signal Feed page).
  - **A13 (HIGH perf + MEDIUM crash, UI)** `ui/pages/model_metrics.py` consolidates the double `clf.fit` on binary models; added an in-sample caveat caption; wrapped `importance_gain.csv` read in a column-schema check.
  - **A14 (MEDIUM)** `reporting/report.py` now uses `Template(..., autoescape=select_autoescape(["html","xml"]))` and renames the local `html` variable so it no longer shadows the stdlib `html` module.
- **Principal-Staff audit architectural follow-ups (2026-04-18, same session)** — swept the 23 unfixed architectural risks from `AUDIT_FINDINGS.md` §B. All 23 addressed; full suite green at 183 passed / 1 skipped.
  - **B1** `backtest/Portfolio` gained optional `margin_ratio` (rejects shorts violating the notional margin requirement) and `borrow_rate_bps_per_day` (accrues daily borrow cost on short notional in `record_equity`).
  - **B2** New `backtest/metrics_v2.py` with `compute_metrics_from_equity` + `infer_periods_per_year` — compound-return CAGR, annualised Sharpe/Sortino on excess returns, drawdown computed from the equity curve itself (side-by-side with legacy `metrics.py`; no call-sites migrated yet so existing reports keep their numbers).
  - **B3** Introduced the `BaseStrategy._backtest_bars_df` + `_backtest_frame_matches_input` contract. Default `init` publishes the input frame; `ConfigurableStrategy` / `RSIStrategy` / `DeMarkConfluenceStrategy` publish their enriched frames; `MLStrategy` opts into `_backtest_frame_matches_input = False` for information bars. `run_backtest` validates shape + required columns and asserts length equality when the flag is True.
  - **B4** New `ml/pipeline.py` (`prepare_xy`, `xgb_estimator_factory`, `weights_for_training`, `apply_regime_if_enabled`) gives both `cli.py` and `tuner/ml_walk_forward.py` a single source of truth. `purged_cv_evaluate` now accepts `sample_weight` and slices it per fold; `ml_walk_forward_tune` threads the configured weights through so tuner fold scores match production training.
  - **B5** `ml/cv.PurgedKFold` added opt-in `symmetric_embargo` — also drops training samples whose `t1` precedes the test block's `min_t0` by the embargo distance, matching AFML §7 Chapter 4 symmetric-embargo.
  - **B6** Vectorised `PurgedKFold._purge_train`, `_embargo_train`, and `ml/evaluator.fold_train_test_interval_disjoint` (merge-style scans + `np.searchsorted`) — removed quadratic pure-Python loops; large-dataset CV is now dominated by model fitting, not purging.
  - **B7** `ModelBundle.predict_proba` (binary) anchors column order on `self.classes_`, picks the positive index via `argmax(classes_)`, and gracefully handles degenerate one-class bundles (returns canonical `[1-p, p]`) so per-regime slices with a single observed class keep working.
  - **B8** `save_model_bundle` guards the multiclass fallback: if `_APRIL_LABEL_CLASSES` was never set and `clf.classes_` looks like an encoded `[0..k-1]` index space, it raises `ValueError` instead of silently persisting a wrong class map.
  - **B9** `walk_forward_splits` now uses ceiling division for the auto-computed `test_size` so it emits exactly `n_folds` windows (was: floor-div produced `n_folds+1` when `(n-min_train) % n_folds != 0`). Explicit `test_size` preserves the legacy cover-the-tail behaviour.
  - **B11** `labels/targets.label_binary` docstring explicitly documents the TP-vs-rest semantics — stop-loss and vertical-timeout collapse into "0" and callers needing a directional negative class should use `label_multiclass`.
  - **B12** `confluence/scorer.score_confluence` coerces bull/bear columns with `pd.to_numeric(errors="coerce")` so string / boolean / NA-like indicator outputs no longer crash with `TypeError`.
  - **B13** `confluence/timeframe_aligner.align_timeframes` validates that both `base_df` and every `higher_dfs` entry has a unique, monotonic datetime index before joining — prevents silent Cartesian blow-ups.
  - **B14** `meta/regime.realized_vol` sets non-positive closes to NaN (kills `log(0) = -inf`), raises `min_periods` to `max(2, window//2)` to reject meaningless one-sample std. `add_vol_regime` sanitises HMM input to finite returns and records `df.attrs["vol_regime_buckets_actual"]` plus a warning when `pd.qcut` collapses fewer buckets than requested.
  - **B15** `indicators/rsi.rsi` gained `mode` in `{"sma","wilder"}` — defaults stay on the existing SMA smoothing, but callers can now opt into Wilder's EMA (`alpha = 1/period, adjust=False`) for canonical RSI.
  - **B16** `indicators/hurst.hurst` docstring now accurately describes `hurst_bull` / `hurst_bear` as trend detectors (not mean-reversion) and clarifies `hurst_mean_revert` is the separate signal.
  - **B17** `indicators/ehlers.super_smoother`, `roofing_filter`, `decycler` now forward-fill the source array through a new `_forward_fill_source` helper and initialise recursion from the first valid index so a single NaN early in the series no longer poisons every downstream value.
  - **B18** `ml/features.align_features_and_labels` logs an INFO line with rows dropped by label NaNs, feature NaNs, and the overall percentage — silent truncation is now visible in operator logs.
  - **B19** `reporting/report.py` HTML template adds a truncation footer under the equity table ("Showing first 500 of {{ equity|length }} equity rows") so HTML snapshots never pretend to be complete.
  - **B20** `ui/pages/tuner.py` filters the `hover_data` list against `results_df.columns` so sparse tuner runs (missing columns) no longer crash the scatter plot.
  - **B21** `ui/pages/walk_forward_lab.py` replaced `except Exception: pass` with narrow `JSONDecodeError` (silent, expected) and `(KeyError, ValueError)` (surfaces as `st.warning`).
  - **B22** `ui/pages/model_lab.py`, `model_trainer.py`, and `walk_forward_lab.py` pass `timeout=3600` to the CLI subprocess, catch `subprocess.TimeoutExpired` + `OSError` / `ValueError` specifically, and surface partial stdout on timeout.
  - **B23** `backtest/logger.hash_features_row` guards empty frames (returns `"empty"`), coerces mixed dtypes via `pd.to_numeric(errors="coerce")`, and falls back to `repr`-based hashing if coercion still fails — no more `IndexError` on empty feature rows or `TypeError` on string-valued indicator columns.

### Planned
- Full SHAP commentary UX polish
- Additional information-driven bars research and tuning presets
- Walk-forward UI deep analytics polish

---

## [0.5.0] - 2026-04-17

ML-depth release: twelve sprints (sample weights → release hygiene). Baseline remains **0.4.1** behavior when optional YAML blocks are absent.

### Added
- **Reporting + per-regime SHAP (v0.5 Sprint 10)** — `render_sampling_section`, `render_meta_section`, `render_regime_section`, `render_wf_tuner_section` (stable ids `section-sampling`, `section-meta`, `section-regime`, `section-wf-tuner`), `render_full_ml_report_html`; backtest template uses `section-regime-timeline` for the legacy regime table; `shap_values_per_regime` + `load_regime_bundles_shap` in `ml/explain.py`; CLI `shap --per-regime`; `docs/DATA_SCHEMA.md` §16; tests `test_reporting.py`, `test_shap.py`, `test_cli_ml.py`
- **Walk-forward ML tuner CLI + Streamlit (v0.5 Sprint 9)** — CLI `wf-tune` reads YAML `wf_tuner` (`metric`, `grid` as expand-grid dict or explicit list of dicts, optional `n_folds`); writes `wf_tune_results.csv` under `model.out_dir`; prints top five grid points by aggregated mean score; Streamlit Walk-forward page **Tuner** tab auto-loads that CSV (or upload) with Plotly box/violin by `grid_id`; `tests/test_cli_ml.py`; README quick-command row
- **Purged walk-forward tuner core (v0.5 Sprint 8)** — `aprilalgo.tuner.ml_walk_forward` (`ml_walk_forward_tune`, `expand_grid`, `aggregate_grid`, metrics `accuracy` / `f1_macro` / `neg_log_loss`); lazy imports from `aprilalgo.cli` inside the tuner to avoid cycles when Sprint 9 adds `wf-tune`; re-exported from `aprilalgo.tuner`; `tests/test_ml_walk_forward.py`
- **Per-regime model routing (v0.5 Sprint 7)** — YAML ``regime.groupby: true`` with ``regime.enabled`` trains one bundle per ``vol_regime`` under ``regime_<k>/`` plus ``regime_index.json``; ``predict`` routes rows and merges ``proba_*`` over a class union (binary maps two-column proba to ``0.0``/``1.0``); ``shap`` uses the default sub-bundle; ``ml_xgboost`` loads all sub-bundles and selects by bar; ``docs/DATA_SCHEMA.md`` §15; tests in ``test_cli_ml.py``, ``test_ml_strategy.py``
- **Regime as an ML feature (v0.5 Sprint 6)** — optional YAML ``regime`` + :func:`~aprilalgo.meta.regime.add_vol_regime` in :func:`~aprilalgo.cli._prepare_xy` before ``build_feature_matrix``; ``meta.json`` persists ``regime``; ``predict``/``shap`` merge via :func:`~aprilalgo.cli._cfg_for_inference`; ``realized_vol`` excluded from ``X`` by default, ``vol_regime`` included; ``ml_xgboost`` mirrors regime in ``init``; ``docs/DATA_SCHEMA.md`` §14; tests ``test_cli_ml.py``, ``test_features.py``, ``test_ml_strategy.py``
- **`ml_xgboost` meta gate (v0.5 Sprint 5)** — when `meta.json` has `meta_logit.enabled`, `MLStrategy` loads `MetaLogitBundle`, stacks `primary_pred` on the primary feature row, gates entries with `meta_proba_threshold`, and logs `pred_proba_meta` on full-schema JSONL events (`null` when the gate is off); `load_meta_logit_bundle(..., rel_path=...)`; `tests/test_ml_strategy.py`
- **Meta-label bundle (v0.5 Sprint 4)** — `aprilalgo.ml.meta_bundle` (`MetaLogitBundle`, JSON save/load); CLI `meta-train` fits :func:`~aprilalgo.labels.meta_label.fit_meta_logit_purged` from `oof_primary.csv` + features, writes `meta_logit.json` + `meta_oof.csv`, updates `meta.json` `meta_logit`; `tests/test_meta_bundle.py`, `tests/test_cli_ml.py`
- **Primary OOF (v0.5 Sprint 3)** — `aprilalgo.ml.oof.compute_primary_oof` (purged k-fold stacked predictions); CLI `oof` writes `oof_primary.csv` and merges `oof.path` into existing `meta.json`; `tests/test_oof.py`, `tests/test_cli_ml.py::test_cli_oof_writes_csv`
- **Sampling strategies (v0.5 Sprint 2)** — CLI `_weights_for_training` supports `sampling.strategy: uniqueness` (overlap weights) and `bootstrap` (multiplicity weights from `sequential_bootstrap_sample`); `meta.json.sampling` records `random_state` / `n_draw` for bootstrap; tests in `tests/test_sampling.py`, `tests/test_cli_ml.py`
- **Sample weights (v0.5 Sprint 1)** — `train_xgb_classifier(..., sample_weight=...)`; CLI `_weights_for_training` stub (uniform when `sampling` absent); `meta.json` key `sampling` with default `strategy: none`; tests `tests/test_trainer_sample_weight.py`
- **ML pipeline information bars** — optional YAML `information_bars` (`enabled`, `bar_type`, `threshold`, optional `source_timeframe`) applies tick/volume/dollar aggregation after `load_price_data` and before triple-barrier + features (`data/loader.py` `load_ohlcv_for_ml`, `data/bars.py` `apply_information_bars_from_config`)
- **`meta.json`** — persists `information_bars` recipe for predict/SHAP and `ml_xgboost` replay; backtest engine iterates bar-aligned rows when the strategy exposes `_backtest_bars_df`
- **`tests/test_loader_ml_bars.py`**, **`tests/test_cli_ml.py`** — coverage for bar-enabled train/predict

### Changed
- **Documentation (v0.5 Sprint 11)** — `ARCHITECTURE.md` §2 full v0.5 sprint roadmap; `README.md` / `CLAUDE.md` / `AGENTS.md` / `docs/MODEL_ROUTING.md` / `docs/SESSION_HANDOFF.md` aligned with shipped ML CLI and regime/meta flows; `docs/DATA_SCHEMA.md` §11 uses explicit `### 11.1` / `### 11.2` anchors.
- **Release hygiene (v0.5 Sprint 12)** — `pyproject.toml` / `aprilalgo.__version__` set to **0.5.0**; `BACKLOG.md` sprint tasks archived under **Done (v0.5.0)**; `docs/REPO_ANALYSIS.md` **historical** note clarifying legacy “v0.2” roadmap labels vs current semver; `tests.md` regenerated.

### Dependencies
- **Optional `hmm` extra** — `[project.optional-dependencies] hmm` with `hmmlearn` for `add_vol_regime(use_hmm=True)`; `uv sync --extra hmm` when wheels exist for your Python

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

[Unreleased]: https://github.com/bobyyandfriends/AprilAlgo/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/bobyyandfriends/AprilAlgo/compare/v0.4.1...v0.5.0
[0.2.0]: https://github.com/bobyyandfriends/AprilAlgo/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/bobyyandfriends/AprilAlgo/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/bobyyandfriends/AprilAlgo/releases/tag/v0.1.0
