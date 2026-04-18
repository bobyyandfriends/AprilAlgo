"""Strategy driven by a saved XGBoost :class:`aprilalgo.ml.trainer.ModelBundle`.

Uses precomputed indicator features (causal); at bar *t* only row *t* is read.
Logs optional JSONL events when ``signal_log_path`` is set.
"""

from __future__ import annotations

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
from aprilalgo.ml.features import extract_feature_matrix
from aprilalgo.ml.trainer import (
    ModelBundle,
    load_model_bundle,
    proba_positive_takeprofit,
)
from aprilalgo.strategies.base import BaseStrategy


class MLStrategy(BaseStrategy):
    """Long-only: enter when P(take-profit class) >= ``entry_proba_threshold``."""

    name = "ml_xgboost"

    def __init__(
        self,
        model_dir: str | Path,
        *,
        entry_proba_threshold: float = 0.5,
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
        self._X: pd.DataFrame = pd.DataFrame()
        self._dt: pd.Series = pd.Series(dtype="datetime64[ns]")
        self._backtest_bars_df: pd.DataFrame | None = None

    def init(self, price_data: pd.DataFrame) -> None:
        self._bundle = load_model_bundle(self.model_dir)
        bundle = self._bundle
        cfg = self.indicator_config_override or bundle.indicator_config
        if not cfg:
            raise ValueError(
                "ml_xgboost requires indicator_config in meta.json or strategy params"
            )
        work = price_data
        ib = bundle.meta.get("information_bars")
        if isinstance(ib, dict) and ib.get("enabled"):
            work = apply_information_bars_from_config(work, ib)
        enriched = IndicatorRegistry.from_config([dict(c) for c in cfg]).apply(work)
        X = extract_feature_matrix(enriched)
        for c in bundle.feature_names:
            if c not in X.columns:
                X[c] = np.nan
        X = X[bundle.feature_names]
        self._X = X.reset_index(drop=True)
        self._dt = work["datetime"].reset_index(drop=True)
        self._backtest_bars_df = work
        if self._log_path:
            self._logger = SignalJsonlLogger(self._log_path)

    def on_bar(self, idx: int, row: pd.Series, portfolio: Portfolio) -> None:
        if self._bundle is None or self._X.empty:
            return
        bundle = self._bundle
        xrow = self._X.iloc[idx : idx + 1]
        if xrow.isna().any(axis=None):
            return

        proba = bundle.predict_proba_row(xrow)
        p_tp = proba_positive_takeprofit(bundle, proba)
        pred = float(bundle.predict(xrow)[0])
        ts = row["datetime"]

        if not portfolio.has_open_position:
            if p_tp >= self.entry_proba_threshold:
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
                if drawdown >= self.stop_loss_pct:
                    exit_trade = True
                elif p_tp < self.entry_proba_threshold * 0.5:
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
                            "outcome": "exit",
                            "pnl": None,
                            "bar_index": idx,
                            "pred_proba_tp": p_tp,
                            "event": "exit",
                        }
                        log_event(self._logger, evt, schema="full")
