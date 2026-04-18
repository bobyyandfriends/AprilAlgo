# AGENTS.md — AI Coding Agent Guide for AprilAlgo

This is the single canonical file any AI agent (or human) working on AprilAlgo should read first. It covers: project identity, non-negotiable rules, project layout, coding conventions, the command reference, the testing reference, Cursor model routing, and the session handoff protocol.

If anything below conflicts with `CLAUDE.md`, this file wins.

---

## 1. Project identity

| Field | Value |
|---|---|
| Name | AprilAlgo |
| Type | Modular multi-timeframe trading backtester + ML pipeline (XGBoost + purged CV + meta-label + regime routing + walk-forward) |
| Language | Python 3.11+ |
| Package manager | `uv` (never `pip`, never `venv`) |
| License | Apache-2.0 |
| Owner | Joshua |
| Current version | source of truth is `pyproject.toml` / `aprilalgo.__version__` |

---

## 2. Non-negotiable rules

1. **No look-ahead bias.** Indicators, strategies, and features must use only past and current data. Strategies implement `on_bar()` — never peek at future bars.
2. **No hardcoded API keys.** Use environment variables or config files.
3. **No modifying `data/` contents.** CSV data is gitignored and managed separately.
4. **Every indicator emits parameterized bull/bear columns**: `{name}_{params}_bull` and `{name}_{params}_bear` (e.g. `rsi_14_bull`, `sma_20_bear`). Never `rsi_bull`.
5. **Pure functions for indicators**: `DataFrame → DataFrame`, no side effects.
6. **Type hints required** on all function signatures.
7. **PEP 8 style**, lines under 100 characters.
8. **Use `pathlib.Path`** for all file paths.
9. **Register new indicators in `indicators/descriptor.py`** — the catalog is the single source of truth. UI, tuner, and strategies all read from it.
10. **Never edit `pyproject.toml` dependencies by hand.** Use `uv add <package>`; `uv.lock` is managed automatically.
11. **When using ML `regime`**, the regime features must be computed with the **same window / n_buckets / HMM flag** as training. `meta.json.regime` (and `_cfg_for_inference` for `predict`/`shap`) must match the bundle used at train time; do not hand-edit inference windows without retraining.
12. **Training must not use regime computed with a different window than the model was trained with.** Any new bar frame used for `train` / `_prepare_xy` / feature rebuild must match the persisted `meta.json` config or the model must be retrained.
13. **If you change `meta.json` keys (new/renamed/removed/retyped), you must update `docs/DATA_SCHEMA.md` in the same PR/session.** A sprint that bumps the bundle schema without touching DATA_SCHEMA is invalid even if tests pass.
14. **Token discipline.** Prefer reading `AGENTS.md`, `docs/DATA_SCHEMA.md`, `PROJECT_STATE.md`, `ARCHITECTURE.md` over bulk source reads. Use targeted file reads, not whole subtrees.

---

## 3. Project structure

```
AprilAlgo/
├── main.py                         # CLI entry point for backtests
├── pyproject.toml / uv.lock        # dependencies
├── configs/                        # YAML configs (backtest + ml)
│   ├── default.yaml
│   ├── smoke_backtest.yaml / smoke_backtest_ml.yaml
│   └── ml/ (default.yaml, meta_train_smoke.yaml, regime_groupby_smoke.yaml, …)
├── src/aprilalgo/
│   ├── __init__.py                 # version + top-level exports
│   ├── config.py                   # YAML loader
│   ├── cli.py                      # `train | evaluate | oof | meta-train | predict | importance | shap | walk-forward | wf-tune | bars`
│   ├── data/                       # loader, fetcher, store, resampler, universe, bars
│   ├── indicators/                 # 11 indicators + descriptor.py (catalog)
│   ├── confluence/                 # timeframe_aligner, scorer, probability
│   ├── labels/                     # triple_barrier, targets, meta_label
│   ├── ml/                         # features, trainer, cv, evaluator, importance, sampling, oof, meta_bundle, explain
│   ├── meta/                       # regime
│   ├── tuner/                      # grid, runner, analyzer, walk_forward, ml_walk_forward
│   ├── backtest/                   # engine, trade, portfolio, metrics, position_sizer, logger, portfolio_runner
│   ├── strategies/                 # base, rsi_sma, demark_confluence, configurable, ml_strategy (`ml_xgboost`)
│   ├── reporting/                  # report.py (Jinja2 HTML sections)
│   └── ui/                         # Streamlit: app.py + pages/*.py
├── tests/                          # pytest suite (fixture: tests/fixtures/daily_data/TEST_daily.csv)
├── docs/
│   ├── DATA_SCHEMA.md              # column contracts (load-bearing)
│   ├── TRIPLE_BARRIER_MATH.md      # labeling math reference
│   ├── HANDOFF.md                  # project-bootstrap narrative
│   └── archive/                    # LEARNING.md, REPO_ANALYSIS.md (historical)
├── ARCHITECTURE.md                 # system design + forward roadmap
├── CHANGELOG.md                    # semver history
├── PROJECT_STATE.md                # live: today's handoff, next sprint, warning zone
└── CLAUDE.md                       # thin pointer → this file
```

### Adding a new indicator

1. Create `src/aprilalgo/indicators/my_indicator.py`.
2. Write a pure function: `def my_indicator(df, period=14, ...) -> pd.DataFrame`.
3. Emit parameterized columns — `myind_{period}_bull`, `myind_{period}_bear`, never `myind_bull`.
4. Export in `indicators/__init__.py`.
5. Register in `indicators/descriptor.py` with an `IndicatorSpec` entry. The UI and tuner auto-pick it up from here — no other wiring is needed.

### Adding a new strategy

1. Create `src/aprilalgo/strategies/my_strategy.py`.
2. Subclass `BaseStrategy`, implement `init()` and `on_bar()`.
3. Register in `strategies/__init__.py` under the `STRATEGIES` dict.
4. Alternatively, use `ConfigurableStrategy` with a custom indicator list from YAML — no new class needed.

---

## 4. Coding conventions

- `from __future__ import annotations` at the top of every module.
- Docstrings: Google style; explain what columns are added for indicator and feature functions.
- Constants: `UPPER_SNAKE_CASE` at module level.
- Classes: `PascalCase`. Functions/variables: `snake_case`.
- No comments that narrate what the code does. Comments should only explain non-obvious intent, trade-offs, or constraints.
- Commit `uv.lock` to version control.
- `data/` OHLCV CSVs are **not** committed (gitignored).
- When adding a dependency: `uv add <package>` (updates both `pyproject.toml` and `uv.lock`). Never edit by hand.
- Update `CHANGELOG.md` under `[Unreleased]` for every user-visible change:
  - `Added` — new features
  - `Changed` — modifications to existing behavior
  - `Fixed` — bug fixes
  - `Removed` — removed features
  - `Dependencies` — dependency deltas

---

## 5. Command reference

All commands are run from the repo root, under `uv run`.

### 5.1 Install / sync

```bash
uv sync                          # install everything from uv.lock
uv sync --extra hmm              # include optional hmmlearn for regime.use_hmm=True
uv add <package>                 # add a new dep (updates pyproject.toml + uv.lock)
```

### 5.2 Backtest (`main.py`)

```bash
uv run python main.py --config configs/default.yaml
uv run python main.py --config configs/smoke_backtest.yaml
uv run python main.py --config configs/smoke_backtest.yaml --strategy demark_confluence
uv run python main.py --config configs/smoke_backtest.yaml --strategy configurable
uv run python main.py --config configs/smoke_backtest_ml.yaml
uv run python main.py --symbol AAPL --timeframe daily --strategy demark_confluence
```

### 5.3 ML — core pipeline (`configs/ml/default.yaml`)

```bash
uv run python -m aprilalgo.cli train         --config configs/ml/default.yaml
uv run python -m aprilalgo.cli evaluate      --config configs/ml/default.yaml
uv run python -m aprilalgo.cli oof           --config configs/ml/default.yaml
uv run python -m aprilalgo.cli predict       --config configs/ml/default.yaml --output outputs/predictions.csv
uv run python -m aprilalgo.cli importance    --config configs/ml/default.yaml
uv run python -m aprilalgo.cli shap          --config configs/ml/default.yaml --output outputs/ml/shap_values.csv --importance-output outputs/ml/shap_importance.csv --max-samples 300
uv run python -m aprilalgo.cli walk-forward  --config configs/ml/default.yaml
uv run python -m aprilalgo.cli wf-tune       --config configs/ml/default.yaml
```

### 5.4 ML — meta-label (non-degenerate `TEST` fixture)

> The default config on the `TEST` fixture yields single-class `y`, which makes the meta logit degenerate. Use `configs/ml/meta_train_smoke.yaml` for end-to-end meta-label smoke. See Appendix A in `PROJECT_STATE.md` for the full walk-through.

```bash
uv run python -m aprilalgo.cli train      --config configs/ml/meta_train_smoke.yaml
uv run python -m aprilalgo.cli oof        --config configs/ml/meta_train_smoke.yaml
uv run python -m aprilalgo.cli meta-train --config configs/ml/meta_train_smoke.yaml
```

### 5.5 ML — regime-groupby training and per-regime SHAP

```bash
uv run python -m aprilalgo.cli train   --config configs/ml/regime_groupby_smoke.yaml
uv run python -m aprilalgo.cli predict --config configs/ml/regime_groupby_smoke.yaml --model-dir models/xgboost/regime_smoke --output outputs/predictions_regime.csv
uv run python -m aprilalgo.cli shap    --config configs/ml/regime_groupby_smoke.yaml --model-dir models/xgboost/regime_smoke --per-regime --max-samples 300
```

### 5.6 Information bars (OHLCV CSV → bars CSV)

```bash
uv run python -m aprilalgo.cli bars --input tests/fixtures/daily_data/TEST_daily.csv --bar-type volume --threshold 500000   --output outputs/bars_volume.csv
uv run python -m aprilalgo.cli bars --input tests/fixtures/daily_data/TEST_daily.csv --bar-type tick   --threshold 3        --output outputs/bars_tick.csv
uv run python -m aprilalgo.cli bars --input tests/fixtures/daily_data/TEST_daily.csv --bar-type dollar --threshold 50000000 --output outputs/bars_dollar.csv
```

### 5.7 UI

```bash
uv run streamlit run src/aprilalgo/ui/app.py
```

---

## 6. Testing

The full suite currently reports **184 collected tests** (183 pass, 1 skipped when `hmmlearn` is unavailable). Run any of the following:

### 6.1 Running the suite

```bash
uv run pytest tests/ -v                      # full suite, verbose
uv run pytest tests/ -q                      # quick green/red summary
uv run pytest tests/ --collect-only -q       # LIVE test inventory (authoritative)
uv run pytest tests/ -x                      # stop at first failure
uv run pytest tests/ -rs                     # show reasons for skips
uv run pytest tests/ --lf                    # rerun only last-failed
uv run pytest tests/ --ff                    # run last-failed first, then the rest
uv run pytest tests/ --cov=aprilalgo         # coverage report (needs pytest-cov)
uv run pytest tests/ -n auto                 # parallel (needs pytest-xdist)
```

### 6.2 Running by module / keyword

```bash
uv run pytest tests/test_indicators.py -v
uv run pytest tests/ -k "meta_label"         # keyword filter across the suite
uv run pytest tests/test_cli_ml.py::test_cli_train_predict_roundtrip -v
```

### 6.3 Test file catalog

| File | Covers |
|---|---|
| `tests/test_indicators.py` | All 11 indicators: RSI, SMA, Bollinger, Volume Trend, DeMark, Hurst, Ehlers, TMI, PV Sequences |
| `tests/test_confluence.py` | Timeframe aligner, scorer, probability |
| `tests/test_tuner.py` | Grid, runner, analyzer (including robustness) |
| `tests/test_walk_forward.py` | Walk-forward index splits |
| `tests/test_backtest.py` | Engine, trade, portfolio, metrics, position sizer |
| `tests/test_logger.py` | JSONL signal logger + full-schema validation |
| `tests/test_portfolio_runner.py` | Multi-symbol portfolio runner |
| `tests/test_labels.py` | Triple-barrier labeling |
| `tests/test_targets.py` | Binary/multiclass target construction |
| `tests/test_features.py` | Feature matrix, exclusions, regime inclusion rules |
| `tests/test_cv.py` | Purged k-fold CV + embargo |
| `tests/test_sampling.py` | Uniqueness + sequential bootstrap weights |
| `tests/test_trainer.py` | XGBoost train/save/load bundle roundtrip |
| `tests/test_trainer_sample_weight.py` | `sample_weight` effect on training |
| `tests/test_evaluator.py` | Purged-CV evaluator metrics |
| `tests/test_importance.py` | Gain + permutation importance |
| `tests/test_oof.py` | Primary OOF capture |
| `tests/test_meta_label.py` | `build_meta_labels`, purged meta logistic |
| `tests/test_meta_bundle.py` | `MetaLogitBundle` JSON save/load + inference |
| `tests/test_ml_walk_forward.py` | Walk-forward tuner core (grid expand, aggregate, metrics) |
| `tests/test_regime.py` | Realized vol + `vol_regime` quantile buckets (+ optional HMM) |
| `tests/test_ml_strategy.py` | `ml_xgboost` strategy: meta gate, regime routing, bars |
| `tests/test_data_bars.py` | Information-driven bars (tick/volume/dollar) |
| `tests/test_loader_ml_bars.py` | ML loader with YAML-driven bar aggregation |
| `tests/test_shap.py` | SHAP values + per-regime SHAP export |
| `tests/test_reporting.py` | HTML report sections + stable section ids |
| `tests/test_cli_ml.py` | End-to-end CLI smoke for every verb |
| `tests/test_streamlit_smoke.py` | Streamlit page imports |

### 6.4 Optional extras

```bash
uv sync --extra hmm                                 # install hmmlearn (when wheels exist for your Python)
uv run pytest tests/ -k hmm -v                      # run the HMM smoke path (otherwise skipped)
```

---

## 7. Model routing (Cursor agents)

Use this before launching any agent or picking a chat model. Pick the **tier** based on task type, then the first-available slug in the row.

### 7.1 Tiers

| Tier | When to use | Primary slug / fallback |
|---|---|---|
| **Fast** | Docs, YAML, version bumps, mechanical edits, Jinja templates, regenerating inventories | `composer-2-fast` / `composer-2` |
| **Standard** | Normal coding against a fixed spec; tests; CLI wiring; Streamlit pages; single-module changes | `claude-4.6-sonnet-medium-thinking` / `gpt-5.4-medium` |
| **Heavy** | Public APIs other modules import; trainer or logger schemas; no-lookahead strategy paths; non-trivial math (purged CV, sampling, meta-labels, regime) | `claude-opus-4-7-thinking-high` / `gpt-5.3-codex` |

### 7.2 Heuristics

- New function signatures or artifacts imported across packages → **Heavy**.
- Touches time order, labels, purged CV, or strategy `on_bar()` → **Heavy**.
- Single module, tests already specified, low coupling → **Standard**.
- Pure prose, config, or template edits → **Fast**.
- Estimated diff `> ~400 lines` or `≥ 4 files` → bump one tier.
- If a **Fast** run fails the same test twice → retry at **Standard**.
- If a **Standard** run fails twice on logic-heavy code → switch to **Heavy**.

### 7.3 Scope shift

If work moves from one tier to another mid-task, stop, tell the user, and recommend the new tier — do not continue with an under-matched model on Heavy-tier work.

---

## 8. Session protocol

### 8.1 Start-of-session gate

Before starting any substantive task, an agent must:

1. Read `PROJECT_STATE.md` §2 "Today's handoff" and §3 "Next sprint".
2. Confirm `uv run pytest tests/ -q` is green on the current branch.
3. Confirm the previous session's handoff marked `Next gate: approved`.
4. Only then begin the task.

### 8.2 End-of-session handoff

At the end of every session with substantive changes, update `PROJECT_STATE.md` §2 "Today's handoff" with this template:

```
### Handoff — <YYYY-MM-DD>
- Sprint / theme: <what this session was about>
- Tasks completed: <bullet list>
- Tasks deferred: <bullet list with reason>
- Commits / file diffs: <SHAs or file list>
- Test result: `uv run pytest tests/ -q` → X passed / Y skipped / 0 failed
- New public symbols: <module.symbol>, ...
- meta.json / bundle schema changes: <new keys + types>; DATA_SCHEMA §N updated
- Docs updated: CHANGELOG Unreleased row(s), DATA_SCHEMA §, README row, other
- Warning Zone deltas: items added / items retired (see §4 of PROJECT_STATE.md)
- Next gate: approved / blocked (reason)
```

### 8.3 Standing rules

1. **No session may end with failing tests or with a red checkbox for items claimed as completed.** If a task can't finish, leave its box unchecked and record why.
2. **Any change to `meta.json` keys requires a same-session update to `docs/DATA_SCHEMA.md`**. A session that bumps the bundle schema without touching DATA_SCHEMA is invalid regardless of passing tests.
3. **No proactive documentation creation.** Only create `.md` files when explicitly requested.
4. **No committing unless explicitly asked.** Stage and describe changes; let the owner pull the trigger.

---

## 9. Canonical references

| Question | Read |
|---|---|
| "How does column X flow between layers?" | `docs/DATA_SCHEMA.md` |
| "What's the triple-barrier math?" | `docs/TRIPLE_BARRIER_MATH.md` |
| "How is the system designed?" | `ARCHITECTURE.md` |
| "What ships in v0.5.0? v0.4.1?" | `CHANGELOG.md` |
| "What are we actively working on?" | `PROJECT_STATE.md` |
| "How was the project bootstrapped?" | `docs/HANDOFF.md` |
| "Archived onboarding / external research?" | `docs/archive/LEARNING.md`, `docs/archive/REPO_ANALYSIS.md` |
