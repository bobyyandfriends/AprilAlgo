# AprilAlgo — Test suite diagnostic report

**Role:** Lead QA automation — audit only (no test or production code was modified for this report).  
**Environment:** Windows, Python 3.14.2, `uv run pytest tests/ -v` from repository root (`AprilAlgo/`).  
**Date:** 2026-04-18  

---

## 1. Execution summary

| Metric | Value |
|--------|--------|
| **Tests collected** | 208 |
| **Passed** | 207 |
| **Failed** | 0 |
| **Skipped** | 1 |
| **Wall clock (last run)** | ~238–241 s (~4 min) with project default pytest plugins |
| **Coverage (package `aprilalgo`, pytest-cov)** | **~60.13%** line coverage (gate in `pyproject.toml`: fail-under 55%) |

**Command note:** The suite was executed as `uv run pytest tests/ -v --tb=short` (equivalent to `pytest tests/ -v` using the project virtualenv). The first attempt used a shell operator unsupported on this host’s PowerShell; the successful run used `cd …; uv run pytest …`.

---

## 2. Hard failures

**None.** No tests crashed with errors, and no assertion failures occurred in the audited run.

---

## 3. Architectural / silent failures (passing but weak or misleading)

These items **passed** in the last run but represent **low signal**, **implicit assertions only**, or **optional-dependency gaps** that can hide regressions.

### 3.1 Import-only smoke tests (no `assert`)

| Location | Issue |
|----------|--------|
| `tests/test_streamlit_smoke.py` — `test_ui_app_importable` (lines 6–7) | Only performs `import aprilalgo.ui.app`. Success means import graph resolves; **no behavioral or UI assertion**. |
| `tests/test_streamlit_smoke.py` — `test_streamlit_pages_importable` (lines 10–25) | Loops `importlib.import_module` over page modules. Again, **no `assert`**; failures only appear if import raises. **Does not** exercise `render()`, Streamlit session state, or subprocess/CLI wiring. |

**Why it matters:** These guard against broken imports after refactors but do **not** validate dashboards, widgets, or error paths.

### 3.2 Indicator catalog test with no outcome assertion

| Location | Issue |
|----------|--------|
| `tests/test_indicators.py` — `TestDescriptorCatalog.test_each_spec_callable` (lines 148–151) | Iterates `for spec in catalog.values(): spec(price_data)` with **no asserts on outputs**. It only checks that each catalog entry runs on fixture OHLCV without raising. Pseudo-indicators (e.g. catalog entries such as `ml_proba`, `shap_local`) may “run” without proving correct integration with model artifacts or SHAP. |

**Why it matters:** High risk of **false confidence**: renamed columns, wrong dtypes, or degenerate outputs would still pass.

### 3.3 Backtest smoke tests keyed on structure, not invariants

| Location | Issue |
|----------|--------|
| `tests/test_backtest.py` — `TestBacktestEngine` (e.g. `test_rsi_sma_produces_result`, `test_demark_confluence_produces_result`, `test_configurable_strategy_produces_result`) | Assertions are mostly **`"metrics" in result`**, **`isinstance(result["metrics"], dict)`**, etc. They do **not** assert equity monotonicity bounds, trade count sanity, short/margin math, or agreement with `metrics_v2` / legacy metrics. |

**Why it matters:** The engine could return internally inconsistent metrics while still satisfying these checks.

### 3.4 Broad `pytest.mark.skipif` on ML CLI integration tests

| Location | Issue |
|----------|--------|
| `tests/test_cli_ml.py` — many tests decorated with `@pytest.mark.skipif(not _CFG.is_file(), reason="default ML config missing")` (e.g. line 19 onward) | If `configs/ml/default.yaml` were **removed or renamed in CI**, a **large slice of ML/CLI coverage would silently skip** rather than fail. |

**Why it matters:** Acceptable for local sandboxes, but it is an **architectural coupling** between “tests exist” and “fixture config path exists”; worth treating as a release gate risk.

### 3.5 Optional dependencies: SHAP and HMM

| Location | Issue |
|----------|--------|
| `tests/test_shap.py` — both tests call `pytest.importorskip("shap")` | If `shap` were dropped from the environment, **SHAP explainability would go untested** while the rest of the suite stays green. |
| `tests/test_regime.py` — `test_add_vol_regime_hmm_smoke` (line 18–19) | **`SKIPPED`** in the audited run: `pytest.importorskip("hmmlearn")` — `No module named 'hmmlearn'`. This is the **sole skipped test** in the last run. |

**Why it matters:** Regime **HMM** paths in `meta/regime.py` are **not exercised** in the default dev install reflected in this run. SHAP is a declared dependency in `pyproject.toml`, so skips are less likely there, but `importorskip` still allows silent loss of coverage if packaging changes.

### 3.6 No `unittest.mock` / `pytest` patching detected

A repository-wide search for `Mock`, `patch`, and `unittest.mock` under `tests/` returned **no matches**. The suite is predominantly **integration-style** (real XGBoost, real subprocess CLI, real CSV fixtures). That is **good for fidelity** but means **slow runs** and **less isolation** for unit-level defects (not a “silent failure,” but a trade-off to document).

---

## 4. Coverage gaps (what is poorly tested or untested)

The following maps **high-risk or user-facing `src/aprilalgo/` modules** to **test gaps**, using the **line coverage table** from the same `pytest tests/ -v` run (pytest-cov). Percentages are **line** coverage unless noted.

### 4.1 Effectively untested or near-zero coverage

| Module / area | Approx. line coverage | Notes |
|---------------|----------------------|--------|
| `aprilalgo/config.py` | **0%** | No tests target configuration loading/helpers. |
| `aprilalgo/confluence/probability.py` | **~9%** | Probability/confluence math largely unexercised; `score_confluence` in `scorer.py` is well covered instead. |
| `aprilalgo/confluence/timeframe_aligner.py` | **~11%** | Multi-timeframe alignment and index validation rarely hit from tests. |
| `aprilalgo/cli.py` | **~11%** | Many subcommands and branches are only touched indirectly; no dense unit matrix on `main()` / individual `cmd_*` handlers. |

### 4.2 Low coverage — data layer and IO

| Module | Approx. line coverage | Risk |
|--------|----------------------|------|
| `aprilalgo/data/fetcher.py` | **~21%** | Network/file fetch paths largely unvalidated in CI. |
| `aprilalgo/data/universe.py` | **~33%** | Universe construction edge cases. |
| `aprilalgo/data/store.py` | **~44%** | Local store semantics. |
| `aprilalgo/data/resampler.py` | **~44%** | Resampling used in pipelines but thinly asserted. |

**Information bars (v0.4.x-style pipeline):** There **is** meaningful coverage via `tests/test_loader_ml_bars.py`, `tests/test_data_bars.py`, and CLI train paths in `tests/test_cli_ml.py` (e.g. information-bars meta persistence). Gaps remain in **fetch/resample/store** integration and failure modes (missing files, bad schemas).

### 4.3 Low coverage — execution / money / sizing

| Module | Approx. line coverage | Risk |
|--------|----------------------|------|
| `aprilalgo/backtest/position_sizer.py` | **~42%** | Position sizing logic under-tested relative to portfolio/engine. |

Portfolio and engine are **moderately** strong (~81% / ~84%); `metrics_v2` is **strong** (~98%).

### 4.4 Low coverage — tuning and robustness

| Module | Approx. line coverage | Risk |
|--------|----------------------|------|
| `aprilalgo/tuner/analyzer.py` | **~55%** | Robustness / neighbor analysis branches under-tested vs grid runner. |

`ml_walk_forward.py` and `walk_forward.py` are comparatively well covered from dedicated tests.

### 4.5 Low coverage — SHAP explainers (library code vs UI)

| Module | Approx. line coverage | Notes |
|--------|----------------------|--------|
| `aprilalgo/ml/explain.py` | **~75%** | `test_shap.py` hits happy paths; `ImportError` / alternate SHAP API branches in `_shap_matrix` less covered. |

### 4.6 Streamlit UI — import smoke vs behavior

| Module cluster | Approx. line coverage | Notes |
|----------------|----------------------|--------|
| `ui/pages/dashboard.py` | **~10%** | Almost no behavioral tests. |
| `ui/pages/tuner.py` | **~7%** | Plotly + dataframe paths untested. |
| `ui/pages/signals.py` | **~13%** | Row rendering and NaN guards not exercised under pytest. |
| `ui/pages/walk_forward_lab.py` | **~17%** | JSON parse / subprocess success and error branches largely unhit. |
| `ui/pages/charts/layers/ml_proba.py`, `shap_local.py`, `overlays.py`, `panels.py` | **~15–29%** | Chart composition and ML overlays need browser-level or component tests to cover properly. |

**Summary:** The **core numerical and ML training/eval** paths are comparatively well tested (~60% overall with concentration in backtest, labels, CV, indicators). **Greatest holes** are **`cli.py` surface area**, **`config`**, **data fetch/store/universe**, **`timeframe_aligner` / `probability`**, **`position_sizer`**, **tuner analyzer edge cases**, and **all Streamlit UI behavior** beyond import.

---

## 5. Stop line

This report is complete. **No fixes were applied** and **no test code was edited** per instructions. Await direction on how to plan remediation (priorities, CI gates, optional extras such as `hmmlearn`, UI test strategy, etc.).
