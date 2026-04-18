"""Strategy driven by a saved XGBoost :class:`aprilalgo.ml.trainer.ModelBundle`.

Uses precomputed indicator features (causal); at bar *t* only row *t* is read.
Logs optional JSONL events when ``signal_log_path`` is set.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aprilalgo.backtest.logger import (
    SignalJsonlLogger,
    hash_features_row,
    log_event,
)
from aprilalgo.backtest.portfolio import Portfolio
from aprilalgo.backtest.position_sizer import (
    FixedFraction,
    FractionalKelly,
    PositionSizer,
)
from aprilalgo.data.bars import apply_information_bars_from_config
from aprilalgo.indicators.registry import IndicatorRegistry
from aprilalgo.meta.regime import add_vol_regime
from aprilalgo.ml.features import extract_feature_matrix
from aprilalgo.ml.meta_bundle import MetaLogitBundle, load_meta_logit_bundle
from aprilalgo.ml.trainer import (
    ModelBundle,
    load_model_bundle,
    proba_positive_takeprofit,
)
from aprilalgo.strategies.base import BaseStrategy


def _meta_proba_positive_class(bundle: MetaLogitBundle, proba_row: np.ndarray) -> float:
    """P(z=1) from meta logit ``predict_proba`` output with shape ``(1, n_classes)``."""
    classes = [float(c) for c in np.asarray(bundle.classes_).ravel()]
    pr = np.asarray(proba_row, dtype=np.float64).ravel()
    if len(classes) == 2:
        for i, c in enumerate(classes):
            if abs(c - 1.0) < 1e-9:
                return float(pr[i])
        return float(pr[int(np.argmax(classes))])
    for i, c in enumerate(classes):
        if abs(float(c) - 1.0) < 1e-9:
            return float(pr[i])
    return float(pr[-1])


class MLStrategy(BaseStrategy):
    """Long-only: enter when primary and (optional) meta probabilities pass thresholds."""

    name = "ml_xgboost"
    # Information-bars / resampling may legitimately change the row count between
    # ``price_data`` and ``_backtest_bars_df``; skip the engine length assertion.
    _backtest_frame_matches_input: bool = False

    def __init__(
        self,
        model_dir: str | Path,
        *,
        entry_proba_threshold: float = 0.5,
        meta_proba_threshold: float = 0.5,
        indicator_config: list[dict[str, Any]] | None = None,
        symbol: str = "UNKNOWN",
        timeframe: str = "daily",
        position_pct: float = 0.95,
        stop_loss_pct: float = 0.05,
        model_id: str = "default",
        signal_log_path: str | Path | None = None,
        position_sizer: str | None = None,
        sizer_fraction: float = 0.02,
        kelly_fraction: float = 0.5,
        kelly_max_pct: float = 0.25,
        avg_win: float = 0.02,
        avg_loss: float = 0.02,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model_dir = Path(model_dir)
        self.entry_proba_threshold = entry_proba_threshold
        self.meta_proba_threshold = meta_proba_threshold
        self.indicator_config_override = indicator_config
        self.symbol = symbol
        self.timeframe = timeframe
        self.position_pct = position_pct
        self.stop_loss_pct = stop_loss_pct
        self.model_id = model_id
        self._log_path = Path(signal_log_path) if signal_log_path else None
        self._logger: SignalJsonlLogger | None = None
        self._position_sizer: PositionSizer | None = None
        if position_sizer == "fixed_fraction":
            self._position_sizer = FixedFraction(fraction=sizer_fraction)
        elif position_sizer == "fractional_kelly":
            self._position_sizer = FractionalKelly(
                kelly_fraction=kelly_fraction,
                max_position_pct=kelly_max_pct,
            )
        self._avg_win = avg_win
        self._avg_loss = avg_loss
        self._bundle: ModelBundle | None = None
        self._regime_bundles: dict[str, ModelBundle] | None = None
        self._regime_default_key: str | None = None
        self._meta_bundle: MetaLogitBundle | None = None
        self._meta_gate_enabled: bool = False
        self._X: pd.DataFrame = pd.DataFrame()
        self._dt: pd.Series = pd.Series(dtype="datetime64[ns]")
        self._backtest_bars_df: pd.DataFrame | None = None

    def init(self, price_data: pd.DataFrame) -> None:
        idx_path = self.model_dir / "regime_index.json"
        if idx_path.is_file():
            data = json.loads(idx_path.read_text(encoding="utf-8"))
            buckets = data["buckets"]
            default_sub = data["default"]
            self._regime_bundles = {str(k): load_model_bundle(self.model_dir / sub) for k, sub in buckets.items()}
            self._regime_default_key = next(k for k, v in buckets.items() if v == default_sub)
            self._bundle = load_model_bundle(self.model_dir / default_sub)
        else:
            self._regime_bundles = None
            self._regime_default_key = None
            self._bundle = load_model_bundle(self.model_dir)
        bundle = self._bundle
        self._meta_bundle = None
        self._meta_gate_enabled = False
        ml = bundle.meta.get("meta_logit")
        if isinstance(ml, dict) and ml.get("enabled"):
            rel = str(ml.get("path", "meta_logit.json"))
            self._meta_bundle = load_meta_logit_bundle(self.model_dir, rel_path=rel)
            self._meta_gate_enabled = True
        cfg = self.indicator_config_override or bundle.indicator_config
        if not cfg:
            raise ValueError("ml_xgboost requires indicator_config in meta.json or strategy params")
        work = price_data
        ib = bundle.meta.get("information_bars")
        if isinstance(ib, dict) and ib.get("enabled"):
            work = apply_information_bars_from_config(work, ib)
        reg = bundle.meta.get("regime")
        if isinstance(reg, dict) and reg.get("enabled"):
            work = add_vol_regime(
                work,
                window=int(reg.get("window", 20)),
                n_buckets=int(reg.get("n_buckets", 3)),
                use_hmm=bool(reg.get("use_hmm", False)),
            )
        enriched = IndicatorRegistry.from_config([dict(c) for c in cfg]).apply(work)
        X = extract_feature_matrix(enriched)
        for c in bundle.feature_names:
            if c not in X.columns:
                X[c] = np.nan
        X = X[bundle.feature_names]
        self._X = X.reset_index(drop=True)
        self._dt = work["datetime"].reset_index(drop=True)
        self._backtest_bars_df = work
        if self._regime_bundles:
            ref = list(bundle.feature_names)
            for bb in self._regime_bundles.values():
                if list(bb.feature_names) != ref:
                    raise ValueError("regime sub-bundles must share identical feature_names for ml_xgboost")
        if self._log_path:
            self._logger = SignalJsonlLogger(self._log_path)

    def _bundle_for_row(self, xrow: pd.DataFrame) -> ModelBundle:
        if self._regime_bundles is None or self._regime_default_key is None or "vol_regime" not in xrow.columns:
            if self._bundle is None:
                raise RuntimeError("MLStrategy: primary bundle missing for non-regime path")
            return self._bundle
        v = xrow["vol_regime"].iloc[0]
        keys = self._regime_bundles
        if pd.isna(v):
            return keys[self._regime_default_key]
        kk = str(int(round(float(v))))
        return keys.get(kk) or keys[self._regime_default_key]

    def _p_meta(self, xrow: pd.DataFrame, pred: float) -> float | None:
        if not self._meta_gate_enabled or self._meta_bundle is None:
            return None
        mb = self._meta_bundle
        x_meta = xrow.copy()
        x_meta["primary_pred"] = float(pred)
        x_meta = x_meta[list(mb.feature_names)]
        proba_m = mb.predict_proba(x_meta)
        return _meta_proba_positive_class(mb, proba_m)

    def on_bar(self, idx: int, row: pd.Series, portfolio: Portfolio) -> None:
        if self._bundle is None or self._X.empty:
            return
        xrow = self._X.iloc[idx : idx + 1]
        if xrow.isna().any(axis=None):
            return

        bundle = self._bundle_for_row(xrow)
        proba = bundle.predict_proba_row(xrow)
        p_tp = proba_positive_takeprofit(bundle, proba)
        pred = float(bundle.predict(xrow)[0])
        p_meta = self._p_meta(xrow, pred)
        ts = row["datetime"]

        if not portfolio.has_open_position:
            allow_primary = p_tp >= self.entry_proba_threshold
            allow_meta = p_meta is None or p_meta >= self.meta_proba_threshold
            if allow_primary and allow_meta:
                close = float(row["close"])
                if self._position_sizer is not None:
                    shares = self._position_sizer.size(
                        portfolio.cash,
                        close,
                        win_prob=float(p_tp),
                        avg_win=self._avg_win,
                        avg_loss=self._avg_loss,
                    )
                else:
                    shares = int(portfolio.cash * self.position_pct / close)
                if shares > 0:
                    portfolio.open_trade(ts, close, side="long", quantity=shares)
                    if self._logger:
                        proba_list = [float(x) for x in np.asarray(proba).ravel()]
                        evt = {
                            "ts": str(ts),
                            "symbol": self.symbol,
                            "tf": self.timeframe,
                            "model_id": self.model_id,
                            "features_hash": hash_features_row(xrow),
                            "pred": pred,
                            "pred_proba": proba_list,
                            "label_multiclass": None,
                            "label_binary": None,
                            "meta_pred": None,
                            "pred_proba_meta": p_meta,
                            "outcome": None,
                            "pnl": None,
                            "bar_index": idx,
                            "pred_proba_tp": p_tp,
                            "event": "entry",
                        }
                        log_event(self._logger, evt, schema="full")
        else:
            for trade in list(portfolio.open_positions):
                entry = trade.entry_price
                close = float(row["close"])
                drawdown = (entry - close) / entry
                exit_trade = False
                if drawdown >= self.stop_loss_pct or p_tp < self.entry_proba_threshold * 0.5:
                    exit_trade = True
                if exit_trade:
                    portfolio.close_trade(trade, ts, close)
                    if self._logger:
                        proba_list = [float(x) for x in np.asarray(proba).ravel()]
                        evt = {
                            "ts": str(ts),
                            "symbol": self.symbol,
                            "tf": self.timeframe,
                            "model_id": self.model_id,
                            "features_hash": hash_features_row(xrow),
                            "pred": pred,
                            "pred_proba": proba_list,
                            "label_multiclass": None,
                            "label_binary": None,
                            "meta_pred": None,
                            "pred_proba_meta": p_meta,
                            "outcome": "exit",
                            "pnl": None,
                            "bar_index": idx,
                            "pred_proba_tp": p_tp,
                            "event": "exit",
                        }
                        log_event(self._logger, evt, schema="full")
