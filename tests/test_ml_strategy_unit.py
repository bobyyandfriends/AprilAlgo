"""Unit tests for ml_strategy helpers and regime routing (no full bundle / OHLCV)."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from aprilalgo.strategies.ml_strategy import MLStrategy, _meta_proba_positive_class


def test_meta_proba_positive_class_binary_standard_order() -> None:
    mb = MagicMock()
    mb.classes_ = np.array([0.0, 1.0])
    pr = np.array([[0.25, 0.75]])
    assert _meta_proba_positive_class(mb, pr) == pytest.approx(0.75)


def test_meta_proba_positive_class_binary_positive_first_column() -> None:
    mb = MagicMock()
    mb.classes_ = np.array([1.0, 0.0])
    pr = np.array([[0.6, 0.4]])
    assert _meta_proba_positive_class(mb, pr) == pytest.approx(0.6)


def test_meta_proba_positive_class_multiclass_no_one_uses_last() -> None:
    mb = MagicMock()
    mb.classes_ = np.array([-1.0, 0.0, 2.0])
    pr = np.array([[0.1, 0.2, 0.7]])
    assert _meta_proba_positive_class(mb, pr) == pytest.approx(0.7)


def test_bundle_for_row_non_regime_uses_primary() -> None:
    s = MLStrategy.__new__(MLStrategy)
    primary = MagicMock()
    s._bundle = primary
    s._regime_bundles = None
    s._regime_default_key = None
    x = pd.DataFrame({"f": [1.0]})
    assert MLStrategy._bundle_for_row(s, x) is primary


def test_bundle_for_row_regime_nan_and_missing_key_use_default() -> None:
    s = MLStrategy.__new__(MLStrategy)
    b0, b1 = MagicMock(name="b0"), MagicMock(name="b1")
    s._bundle = b0
    s._regime_bundles = {"0": b0, "1": b1}
    s._regime_default_key = "0"
    assert MLStrategy._bundle_for_row(s, pd.DataFrame({"vol_regime": [np.nan], "a": [1.0]})) is b0
    assert MLStrategy._bundle_for_row(s, pd.DataFrame({"vol_regime": [1.0], "a": [1.0]})) is b1
    assert MLStrategy._bundle_for_row(s, pd.DataFrame({"vol_regime": [99.0], "a": [1.0]})) is b0


def test_bundle_for_row_runtime_error_when_bundle_missing() -> None:
    s = MLStrategy.__new__(MLStrategy)
    s._bundle = None
    s._regime_bundles = None
    s._regime_default_key = None
    with pytest.raises(RuntimeError, match="primary bundle missing"):
        MLStrategy._bundle_for_row(s, pd.DataFrame({"f": [1.0]}))
