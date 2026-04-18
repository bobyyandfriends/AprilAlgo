# AprilAlgo — baseline audit report

Date: 2026-04-18  
Scope: full test suite, coverage gate, ruff/mypy/bandit, pip-audit, manual trace of critical paths, reconciliation with `AUDIT_FINDINGS.md`.

---

## Executive summary

| Check | Result |
|--------|--------|
| `uv run pytest tests/` | **207 passed**, **1 skipped** (HMM unless `uv sync --extra hmm`) |
| Coverage (`aprilalgo`) | **~60%** total lines; gate **`--cov-fail-under=55`** in `pyproject.toml` |
| `uv run ruff check src tests` | **Clean** (line length 120; auto-fixes applied) |
| `uv run mypy src/aprilalgo` | **Clean** with pragmatic overrides for pandas/numpy-heavy modules (see below) |
| `uv run bandit -r src -q` | **Clean** (B404/B603 `# nosec` on trusted Streamlit CLI wrappers) |
| `pip-audit` (prod export) | **CVE-2026-40192** in Pillow **12.1.1** → resolved by pinning **`pillow>=12.2.0`** in dependencies |

---

## Quick fixes (applied this session)

1. **Tooling** — Added dev deps and `[tool.pytest.ini_options]`, `[tool.ruff]`, `[tool.mypy]`, `[tool.bandit]` in `pyproject.toml`; documented commands in `AGENTS.md` §6.5.
2. **Ruff** — Import sort, unused imports removed, `zip(..., strict=True)` in `tuner/grid.py`, removed dead `n` in `ml/sampling.py`, line length 120, SIM108 fixes where safe.
3. **Mypy** — Added `pandas-stubs`; narrowed `walk_forward_splits` so `test_size` is always `int` in the loop; removed stale `type: ignore` comments; `warn_return_any = false`; explicit `ignore_errors` overrides for six modules where ExtensionArray stubs are still noisy.
4. **Security / correctness** — `pillow>=12.2.0`; `load_pickle` `# nosec B301` with comment; `MLStrategy._bundle_for_row` uses `RuntimeError` instead of `assert` (bandit B101).
5. **Docs** — `AGENTS.md` §6: default pytest+coverage behavior and `--no-cov` fast path.

---

## Quick fixes (backlog, optional)

| Item | Notes |
|------|--------|
| **`metrics_v2` tests** | **Done (2026-04-18 verification pass):** `tests/test_metrics_v2.py` — module line coverage **≥85%** on gate run. |
| **CLI / UI coverage** | **Partially done:** `evaluate` + `importance` CLI smoke in `tests/test_cli_ml.py`; extra Streamlit page import smoke in `tests/test_streamlit_smoke.py`. Deep dashboard tests still deferred. |
| **Bandit B404/B603** | **Done:** per-line `# nosec B404` / `# nosec B603` on the three Streamlit CLI wrapper pages. |

---

## Deep architectural risks (discussion / future sprints)

These align with `PROJECT_STATE.md` §4 “design decisions” and open themes — not regressions from this audit.

1. **Legacy `metrics.py` vs `metrics_v2`** — Headline numbers differ; migration needs an explicit regression harness (`PROJECT_STATE.md` §3).
2. **Binary TP-vs-rest labeling** — Product choice; multiclass path remains the escape hatch (`PROJECT_STATE.md` §4).
3. **Purged CV `symmetric_embargo`** — Still opt-in; flipping default needs metric-delta communication (`PROJECT_STATE.md` §3–§4).
4. **Pickle artifacts** — `load_pickle` / bundle loads assume trusted paths; hostile inputs are out of scope but relevant for any future networked ingestion.
5. **Mypy overrides** — `aprilalgo.meta.regime`, several `indicators/*`, `data.resampler` are excluded from error reporting until stubs/annotations improve.

---

## Manual systems trace (no-lookahead and money paths)

- **Backtest** — `backtest/engine.py` iterates bar indices; strategies consume `row` / pre-built frames; `BaseStrategy` documents `_backtest_bars_df` / `_backtest_frame_matches_input` contract (enforced before loop when set).
- **Portfolio** — `portfolio.py` equity snapshot and `Trade` costs align with audit fixes (short liability, exit fees on `Trade`); margin/borrow kwargs exist for short realism.
- **ML** — Features/labels: triple-barrier and targets use post-*t* path only where defined for labels; `PurgedKFold` / evaluator guard degenerate folds; OOF uses global class axis; `ml/pipeline.py` centralizes XY prep and weights; `ml_walk_forward` uses deepcopy for configs.
- **Walk-forward** — Auto `test_size` uses ceiling division and caps fold count; explicit `test_size` preserves tail coverage (documented in `walk_forward.py`).

---

## AUDIT_FINDINGS.md reconciliation

### Section A (bugs fixed in the original audit)

Spot-checked **2026-04-18**: corrective patterns called out in `AUDIT_FINDINGS.md` §A remain present in code (e.g. portfolio equity/short handling, DeMark transition gating, OOF class alignment, meta-label NaN handling, tuner robustness formula, evaluator guards, HTML autoescape). **Full suite green** provides ongoing regression coverage for these paths.

### Section B (architectural risks)

`AUDIT_FINDINGS.md` includes a **status banner (2026-04-18)** stating B1–B23 were **addressed** in session 2, with narrative retained for history. This baseline audit **confirms**:

- **Resolved in code** — Examples: `Portfolio` margin/borrow kwargs (B1); `metrics_v2` module exists (B2 — migration to call-sites still open); engine/strategy frame validation (B3); `sample_weight` through evaluator and WF tuner (B4); symmetric embargo flag + vectorized purge helpers (B5–B6); `ModelBundle.predict_proba` / `save_model_bundle` guards (B7–B8); `walk_forward_splits` fold semantics (B9); `ml/pipeline.py` (B10); confluence/regime/indicator/UI items B12–B23 per `CHANGELOG` / `PROJECT_STATE` handoff.

- **Still “product / roadmap”** — Binary label semantics, legacy metrics in reports, optional `metrics_v2` migration, default symmetric embargo — tracked in `PROJECT_STATE.md` §3–§4, not as unfixed §B defects.

**Recommendation:** Keep `AUDIT_FINDINGS.md` **frozen as historical narrative** with the existing status banner; use `PROJECT_STATE.md` §3 for forward work and this `BUG_REPORT.md` for tooling/security baselines.

---

## Commands reference

```bash
uv sync --group dev
uv run pytest tests/
uv run pytest tests/ --no-cov
uv run ruff check src tests && uv run ruff format src tests --check
uv run mypy src/aprilalgo
uv run bandit -r src -q
```

---

## Verification pass (2026-04-18)

Runbook: `audit_findings_verify-and-fix` (verify A1–A14 / B1–B23 first, patch only regressions; then BUG_REPORT C1–C3; final static-analysis gate).

- **Baseline pytest:** `183 passed`, `1 skipped` (`uv run pytest tests/ --no-cov -q`).
- **A1–A14 / B1–B23:** Spot-checked in source; **no regressions** — corrective patterns from `AUDIT_FINDINGS.md` remain present (no re-patches required this pass).
- **C1 — `metrics_v2`:** Added `tests/test_metrics_v2.py`; `aprilalgo.backtest.metrics_v2` line coverage **≥85%** on full-suite run (terminal gate: **98%** with only unreachable infer edge branches missing).
- **C2 — smoke:** Extended `tests/test_streamlit_smoke.py` (dashboard, signals, tuner, regime_lab, portfolio_lab, charts page); added `test_cli_evaluate_prints_json_metrics` and `test_cli_importance_writes_csvs` in `tests/test_cli_ml.py`.
- **C3 — Bandit:** `# nosec B404` / `# nosec B603` on `model_lab.py`, `model_trainer.py`, `walk_forward_lab.py` subprocess usage; `uv run bandit -r src -q` **0 issues**.
- **Final gate:** `207 passed`, `1 skipped`; coverage **~60%**; `ruff check` / `ruff format --check` / `mypy` / `bandit` all clean. `ruff format src tests` applied to clear pre-existing format drift on 25 files.

### Deferred (per runbook C4 — product / roadmap; explicitly out of scope for this pass)

1. Legacy `metrics.py` vs `metrics_v2` call-site migration.  
2. Binary TP-vs-rest labeling semantics change.  
3. Symmetric embargo default flip.  
4. Pickle trust boundary / networked ingestion.  
5. Mypy overrides for pandas-heavy modules beyond current baseline.

---

## Test suite remediation (2026-04-18)

- **Silent-failure tightening**: strengthened assertions in:
  - `tests/test_streamlit_smoke.py`
  - `tests/test_indicators.py`
  - `tests/test_backtest.py`
  - `tests/test_regime.py` / `tests/test_shap.py` / `tests/test_cli_ml.py` (skip guards removed where deps are hard)
- **Behavioral Streamlit UI tests**: added `streamlit.testing.v1.AppTest` coverage under `tests/ui/`, including full page navigation in `tests/ui/test_app_navigation.py`.
  - Note: `AppTest.from_function(render)` failed due to Streamlit execution context; UI tests use small runner scripts under `tests/ui/apps/` and `AppTest.from_file(...)` to patch `discover_symbols` / `subprocess.run` deterministically.
- **Coverage lifted + gate raised**:
  - Coverage increased from ~60% to **70.77%**.
  - `pyproject.toml` `--cov-fail-under` raised from **55** to **68** (achieved − 2).
- **Dependency/config coupling hardened**:
  - SHAP is treated as a hard dependency in tests (no `importorskip`).
  - `configs/ml/default.yaml` is now a **collection-time requirement** for ML CLI tests (no more silent `skipif`).
  - `hmmlearn` remains an optional extra (`[project.optional-dependencies].hmm`) because it does not install cleanly on this Windows + Python 3.14 environment (native build toolchain/wheels); the HMM test continues to `importorskip("hmmlearn")` accordingly.
- **New unit test modules added** (coverage gaps):
  - `tests/test_config.py`
  - `tests/test_confluence_probability.py`
  - `tests/test_confluence_timeframe_aligner.py`
  - `tests/test_position_sizer.py`
  - `tests/test_tuner_analyzer.py`
  - `tests/test_data_universe.py`
  - `tests/test_data_store.py`
  - `tests/test_data_resampler.py`
  - `tests/test_data_fetcher.py`
  - `tests/test_cli_helpers.py`

---

## Test suite remediation — pass 2 (2026-04-18)

- **Baseline before pass 2**: **290 passed**, **1 skipped**, total line coverage **~70.8%** (`--cov-fail-under=68`).
- **After pass 2**: **341 passed**, **1 skipped**, total line coverage **~76.7%**; `--cov-fail-under` raised to **74** (floor of achieved total minus 2).
- **CLI**: new `tests/test_cli_bars.py` exercises `cmd_bars` (tick/volume/dollar, NaT drop, unknown bar type), `cmd_walk_forward` JSON shape (with `load_ohlcv_for_ml` mocked), `cmd_shap` `--per-regime` error without `regime_index.json`, `cmd_predict` missing symbol, and `python -m aprilalgo.cli` entry for `bars`.
- **UI helpers & reporting**: `tests/test_ui_helpers.py` (`discover_symbols`, `format_metric`); extended `tests/test_cli_helpers.py` (`_regime_bucket_key_series`, sampling branch); extended `tests/test_reporting.py` (empty sections, truncation note, escaped notes).
- **Streamlit AppTest**: `tests/ui/test_portfolio_model_pages.py` plus runners `tests/ui/apps/portfolio_lab_fixture.py`, `model_lab_mocked.py`, `model_trainer_mocked.py`, `model_metrics_mocked.py` (portfolio backtest on `TEST` fixture; ML lab / trainer with mocked `subprocess.run`; model metrics with mocked purged CV).
- **Charts / data**: `tests/test_charts_ml_artifacts.py`, `tests/test_demark_counts_layer.py`; loader/bars/targets edges in `tests/test_data_loader_edges.py` and additions to `tests/test_data_bars.py` / `tests/test_targets.py`.

---

## Phases A–E — CLI, UI, charts, core ML, gate (2026-04-18)

- **Result**: **389 passed**, **1 skipped**; total line coverage for `aprilalgo` **~84.2%**; `--cov-fail-under` set to **82** (floor of achieved total minus 2).
- **Phase A (CLI internals)**: `tests/test_cli_internals.py` — OOF alignment, `cmd_meta_train` / `cmd_oof`, regime groupby training, `cmd_wf_tune`, `cmd_shap` per-regime error path, `_predict_regime_routed` validation, `main` bars + missing subcommand.
- **Phase B (Streamlit AppTest)**: `tests/ui/apps/dash_run_success.py`, `tuner_run_mocked.py`, `charts_smoke.py`; tests in `test_dashboard_app.py`, `test_tuner_app.py`, `test_charts_page_app.py`.
- **Phase C (charts + `ml_artifacts`)**: `tests/test_charts_page_helpers.py`, `tests/test_ml_artifacts_helpers.py`, extended `tests/ui/test_chart_layers.py` (DeMark counts, ML proba stacked / shading).
- **Phase D (regime / pipeline / evaluator)**: `tests/test_regime_pipeline_eval.py` — `realized_vol` / `add_vol_regime` edges, sampling `ValueError`, multiclass XGB factory, `_safe_roc_auc`, `purged_cv_evaluate` weight mismatch, `fold_train_test_interval_disjoint`.
- **Phase E**: Full `pytest` gate with coverage; `pyproject.toml` threshold bump; this section; `ruff` / `mypy` / `bandit` re-run as part of verification (see commands in `AGENTS.md` §6).
