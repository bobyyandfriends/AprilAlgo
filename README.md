# AprilAlgo

Modular **multi-timeframe** stock backtester in Python: technical indicators, **confluence** scoring, parameter tuning, **XGBoost** + purged CV (v0.3), meta-label / regime / reporting helpers (v0.4), and a **Streamlit** UI.

- **Python** 3.11+ · **uv** · **Apache-2.0**

## Install

```bash
uv sync
# Optional: HMM regimes (`add_vol_regime(use_hmm=True)`) — needs wheels for your Python (often 3.11–3.12):
uv sync --extra hmm
```

## Quick commands

| Task | Command |
|------|---------|
| Backtest (CLI) | `uv run python main.py --symbol AAPL --strategy demark_confluence` |
| Streamlit UI | `uv run streamlit run src/aprilalgo/ui/app.py` |
| Tests | `uv run pytest tests/ -v` |
| ML train | `uv run python -m aprilalgo.cli train --config configs/ml/default.yaml` |
| ML sampling (overlap / bootstrap weights) | Same config + optional YAML `sampling` block: `strategy: uniqueness` or `strategy: bootstrap` (see `docs/DATA_SCHEMA.md` §11) |
| ML evaluate (purged CV) | `uv run python -m aprilalgo.cli evaluate --config configs/ml/default.yaml` |
| ML OOF (purged folds → CSV) | `uv run python -m aprilalgo.cli train --config configs/ml/default.yaml` then `uv run python -m aprilalgo.cli oof --config configs/ml/default.yaml` (writes `oof_primary.csv`; see `docs/DATA_SCHEMA.md` §12) |
| ML predict | `uv run python -m aprilalgo.cli predict --config configs/ml/default.yaml --output predictions.csv` |
| Feature importance | `uv run python -m aprilalgo.cli importance --config configs/ml/default.yaml` |
| SHAP export | `uv run python -m aprilalgo.cli shap --config configs/ml/default.yaml` |
| Walk-forward (JSON) | `uv run python -m aprilalgo.cli walk-forward --config configs/ml/default.yaml` |
| Information bars (CSV) | `uv run python -m aprilalgo.cli bars --input path.csv --bar-type volume --threshold 1e6 --output out.csv` |

Use **`configs/ml/default.yaml`** with `data_dir: tests/fixtures` and symbol **`TEST`** for a built-in OHLCV smoke path (no live data required). Optional **`information_bars`** in that YAML aggregates the loaded series before labels and features (see `docs/DATA_SCHEMA.md`). Optional **`sampling`** controls per-row XGBoost weights (`uniqueness` vs sequential `bootstrap`; see §11).

## Docs

- **[docs/DATA_SCHEMA.md](docs/DATA_SCHEMA.md)** — column contracts between layers  
- **[docs/MODEL_ROUTING.md](docs/MODEL_ROUTING.md)** — Cursor agent model tiers  
- **[ARCHITECTURE.md](ARCHITECTURE.md)** — system design  
- **[AGENTS.md](AGENTS.md)** — rules for AI coding agents  
- **[CHANGELOG.md](CHANGELOG.md)** — releases

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
