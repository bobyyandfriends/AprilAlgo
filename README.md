# AprilAlgo

Modular **multi-timeframe** stock backtester in Python: technical indicators, **confluence** scoring, parameter tuning, **XGBoost** + purged CV, sampling / OOF / meta-label / **regime** routing / walk-forward **wf-tune**, reporting, SHAP, and a **Streamlit** UI (**v0.5.0**).

- **Python** 3.11+ · **uv** · **Apache-2.0**

## Install

```bash
uv sync
# Optional: HMM regimes (`add_vol_regime(use_hmm=True)`) — needs wheels for your Python (often 3.11–3.12):
uv sync --extra hmm
```

Fenced blocks labeled `bash` are ordinary shell commands: run them from a **repo root** terminal. On macOS or Linux use your default terminal; on **Windows** use **PowerShell**, **cmd**, or **Git Bash** — the same `uv run …` lines work wherever `uv` is installed (the `bash` tag is Markdown convention, not a requirement to use Git Bash).

## Quick commands

| Task | Command |
|------|---------|
| Backtest (CLI) | `uv run python main.py --symbol AAPL --strategy demark_confluence` |
| Backtest smoke (`TEST` fixture, no `data/` CSV) | `uv run python main.py --config configs/smoke_backtest.yaml` |
| Streamlit UI | `uv run streamlit run src/aprilalgo/ui/app.py` |
| Tests | `uv run pytest tests/ -v` |
| ML train | `uv run python -m aprilalgo.cli train --config configs/ml/default.yaml` |
| ML sampling (overlap / bootstrap weights) | Same config + optional YAML `sampling` block: `strategy: uniqueness` or `strategy: bootstrap` (see `docs/DATA_SCHEMA.md` §11) |
| ML evaluate (purged CV) | `uv run python -m aprilalgo.cli evaluate --config configs/ml/default.yaml` |
| ML OOF (`oof`) | After `train`: `uv run python -m aprilalgo.cli oof --config configs/ml/default.yaml` → `oof_primary.csv` under `model.out_dir` (see `docs/DATA_SCHEMA.md` §12) |
| ML meta-label (`meta-train`) | After OOF: `uv run python -m aprilalgo.cli meta-train --config configs/ml/default.yaml` → `meta_logit.json` + `meta_oof.csv` (see §13). On symbol **`TEST`**, use `configs/ml/meta_train_smoke.yaml` for `train` / `oof` / `meta-train` so meta labels are not degenerate (see `PROJECT_STATE.md` Appendix B). |
| ML backtest with meta gate | Bundle dir with `meta.json` `meta_logit.enabled: true` + `meta_logit.json`; `MLStrategy` loads the meta logit and gates long entries on `meta_proba_threshold` (see `docs/DATA_SCHEMA.md` §6 `pred_proba_meta`) |
| Per-regime train / predict | YAML `regime.groupby: true` with `regime.enabled` → one bundle per `vol_regime` under `regime_<k>/` plus `regime_index.json`; `predict` and `ml_xgboost` route rows by regime (see `docs/DATA_SCHEMA.md` §15) |
| ML predict | `uv run python -m aprilalgo.cli predict --config configs/ml/default.yaml --output predictions.csv` |
| Feature importance | `uv run python -m aprilalgo.cli importance --config configs/ml/default.yaml` |
| SHAP export | `uv run python -m aprilalgo.cli shap --config configs/ml/default.yaml` |
| SHAP per-regime (`--per-regime`) | With `regime_index.json`: `uv run python -m aprilalgo.cli shap --config configs/ml/default.yaml --model-dir models/xgboost/latest --per-regime` → `regime_<k>_shap_values.csv` / `regime_<k>_shap_importance.csv` (see `docs/DATA_SCHEMA.md` §16) |
| Walk-forward (JSON) | `uv run python -m aprilalgo.cli walk-forward --config configs/ml/default.yaml` |
| Walk-forward ML tuner | `uv run python -m aprilalgo.cli wf-tune --config configs/ml/default.yaml` → `wf_tune_results.csv` under `model.out_dir` (YAML `wf_tuner.grid` / `wf_tuner.metric`; Streamlit Walk-forward **Tuner** tab) |
| Information bars (CSV) | `uv run python -m aprilalgo.cli bars --input path.csv --bar-type volume --threshold 1e6 --output out.csv` |

Use **`configs/ml/default.yaml`** with `data_dir: tests/fixtures` and symbol **`TEST`** for a built-in OHLCV smoke path (no live data required). For **backtests**, `configs/smoke_backtest.yaml` sets `data_dir: tests/fixtures` and symbol **`TEST`**; **`main.py`** reads optional `data_dir` from YAML so OHLCV is resolved under `tests/fixtures/daily_data/`. After ML **`train`**, **`configs/smoke_backtest_ml.yaml`** runs an **`ml_xgboost`** backtest against `model.out_dir`. Copy-paste commands for every CLI path (including testing, model routing, and the end-of-session handoff protocol): see **[AGENTS.md](AGENTS.md) §5 "Command reference"** and **§6 "Testing"**. Optional **`information_bars`** in YAML aggregates the loaded series before labels and features (see `docs/DATA_SCHEMA.md`). Optional **`sampling`** controls per-row XGBoost weights (`uniqueness` vs sequential `bootstrap`; see §11). Optional **`wf_tuner`** drives **`wf-tune`** (walk-forward grid search); optional **`regime.groupby`** enables **`shap --per-regime`** when trained (smoke config: `configs/ml/regime_groupby_smoke.yaml`). CLI edge cases (`meta-train` on the `TEST` fixture, `shap --per-regime` prerequisites) are documented in `PROJECT_STATE.md` Appendix B.

## Docs

- **[AGENTS.md](AGENTS.md)** — AI agent guide: rules, commands, testing, model routing, session protocol (the one file any agent should read first)
- **[CLAUDE.md](CLAUDE.md)** — Claude-specific preferences (thin pointer to `AGENTS.md`)
- **[PROJECT_STATE.md](PROJECT_STATE.md)** — today's handoff, next sprint, warning zone (live, updated per session)
- **[ARCHITECTURE.md](ARCHITECTURE.md)** — system design + forward roadmap
- **[CHANGELOG.md](CHANGELOG.md)** — semver release history
- **[docs/DATA_SCHEMA.md](docs/DATA_SCHEMA.md)** — column contracts between layers (load-bearing)
- **[docs/TRIPLE_BARRIER_MATH.md](docs/TRIPLE_BARRIER_MATH.md)** — triple-barrier labeling math
- **[docs/HANDOFF.md](docs/HANDOFF.md)** — project-bootstrap narrative
- **[docs/archive/](docs/archive/)** — historical references (`LEARNING.md`, `REPO_ANALYSIS.md`)

## Layout

```
src/aprilalgo/
  data/ indicators/ confluence/ tuner/ backtest/ strategies/
  labels/ ml/ meta/ reporting/ ui/
configs/     # YAML (backtest + ml)
tests/       # pytest + fixtures/daily_data/TEST_daily.csv
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
