"""Tests for purged walk-forward ML tuner (Sprint 8)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from aprilalgo.tuner import ml_walk_forward as mwf
from aprilalgo.tuner.ml_walk_forward import (
    SUPPORTED_METRICS,
    aggregate_grid,
    expand_grid,
    ml_walk_forward_tune,
    supported_metrics,
)

_ROOT = Path(__file__).resolve().parents[1]
_CFG = _ROOT / "configs" / "ml" / "default.yaml"


def _fixture_tune_cfg() -> dict:
    cfg = yaml.safe_load(_CFG.read_text(encoding="utf-8"))
    cfg["walk_forward"] = {"n_folds": 2, "min_train": 28, "test_size": 22}
    cfg["cv"] = {"n_splits": 2, "embargo": 0}
    cfg["model"] = dict(cfg.get("model", {}))
    cfg["model"]["xgb"] = {
        "n_estimators": 12,
        "max_depth": 2,
        "learning_rate": 0.2,
    }
    return cfg


@pytest.mark.skipif(not _CFG.is_file(), reason="default ML config missing")
def test_returns_one_row_per_grid_fold() -> None:
    from aprilalgo.cli import _prepare_xy
    from aprilalgo.tuner.walk_forward import walk_forward_splits

    cfg = _fixture_tune_cfg()
    grid = [{"max_depth": 2}, {"max_depth": 3}]
    wf = cfg["walk_forward"]
    X, _y, _t0, _t1, _task = _prepare_xy(cfg, symbol="TEST")
    splits = list(
        walk_forward_splits(
            len(X),
            n_folds=wf["n_folds"],
            min_train=int(wf["min_train"]),
            test_size=wf.get("test_size"),
        )
    )
    df = ml_walk_forward_tune(cfg, grid, n_folds=wf["n_folds"], metric="accuracy")
    assert len(df) == len(grid) * len(splits)
    assert set(df["grid_id"].unique()) == {mwf._grid_id(grid[0]), mwf._grid_id(grid[1])}
    assert df["wf_fold"].nunique() == len(splits)


def test_aggregate_grid_columns() -> None:
    raw = pd.DataFrame(
        {
            "grid_id": ["a", "a", "b", "b"],
            "score": [0.5, 0.7, 0.2, 0.4],
        }
    )
    agg = aggregate_grid(raw, "score")
    assert list(agg.columns) == ["grid_id", "mean", "std", "n_folds"]
    row_a = agg.loc[agg["grid_id"] == "a"].iloc[0]
    assert row_a["mean"] == pytest.approx(0.6)
    assert row_a["n_folds"] == 2


def test_aggregate_grid_empty() -> None:
    agg = aggregate_grid(pd.DataFrame(), "score")
    assert agg.empty
    assert list(agg.columns) == ["grid_id", "mean", "std", "n_folds"]


def test_aggregate_grid_single_row_std_nan() -> None:
    raw = pd.DataFrame({"grid_id": ["x"], "score": [0.5]})
    agg = aggregate_grid(raw, "score")
    assert pd.isna(agg.iloc[0]["std"])
    assert int(agg.iloc[0]["n_folds"]) == 1


def test_expand_grid_product() -> None:
    g = expand_grid({"a": [1, 2], "b": ["x", "y"]})
    assert len(g) == 4
    pairs = {(d["a"], d["b"]) for d in g}
    assert pairs == {(1, "x"), (1, "y"), (2, "x"), (2, "y")}


def test_expand_grid_scalar_promoted() -> None:
    g = expand_grid({"k": 7})
    assert g == [{"k": 7}]


def test_expand_grid_empty_spec_returns_singleton() -> None:
    assert expand_grid({}) == [{}]


def test_expand_grid_order_follows_product() -> None:
    """Later keys vary fastest (itertools.product order)."""
    g = expand_grid({"a": [1, 2], "b": ["x", "y"]})
    assert [d["a"] for d in g] == [1, 1, 2, 2]
    assert [d["b"] for d in g] == ["x", "y", "x", "y"]


def test_metric_registry_keys() -> None:
    assert supported_metrics() == SUPPORTED_METRICS
    assert {"accuracy", "f1_macro", "neg_log_loss"} <= set(SUPPORTED_METRICS)


@pytest.mark.skipif(not _CFG.is_file(), reason="default ML config missing")
def test_deterministic_with_seed() -> None:
    cfg = _fixture_tune_cfg()
    cfg["random_state"] = 123
    grid = [{"max_depth": 2}]
    wf = cfg["walk_forward"]
    df1 = ml_walk_forward_tune(cfg, grid, n_folds=wf["n_folds"], metric="accuracy")
    df2 = ml_walk_forward_tune(cfg, grid, n_folds=wf["n_folds"], metric="accuracy")
    pd.testing.assert_frame_equal(
        df1.reset_index(drop=True),
        df2.reset_index(drop=True),
        check_exact=False,
        rtol=0.0,
        atol=0.0,
    )


def test_empty_grid_raises() -> None:
    # Validated before _prepare_xy — stub cfg is enough.
    with pytest.raises(ValueError, match="non-empty"):
        ml_walk_forward_tune({"symbol": "TEST"}, [], n_folds=2, metric="accuracy")


def test_unknown_metric_raises() -> None:
    with pytest.raises(ValueError, match="metric must be one of"):
        ml_walk_forward_tune(
            {"symbol": "TEST"}, [{}], n_folds=2, metric="not_a_metric"
        )


@pytest.mark.skipif(not _CFG.is_file(), reason="default ML config missing")
def test_single_fold_runs() -> None:
    """``n_folds=1`` is accepted by ``walk_forward_splits`` (no ValueError)."""
    cfg = _fixture_tune_cfg()
    cfg["walk_forward"] = {"n_folds": 1, "min_train": 25, "test_size": 30}
    grid = [{"max_depth": 2}]
    df = ml_walk_forward_tune(cfg, grid, n_folds=1, metric="accuracy")
    assert len(df) >= 1
    assert df["wf_fold"].min() == 0
    assert df["wf_fold"].is_monotonic_increasing


@pytest.mark.skipif(not _CFG.is_file(), reason="default ML config missing")
def test_neg_log_loss_metric_runs() -> None:
    cfg = _fixture_tune_cfg()
    grid = [{"max_depth": 2}]
    wf = cfg["walk_forward"]
    df = ml_walk_forward_tune(cfg, grid, n_folds=wf["n_folds"], metric="neg_log_loss")
    assert not df.empty
    assert "score" in df.columns


@pytest.mark.skipif(not _CFG.is_file(), reason="default ML config missing")
def test_f1_macro_metric_runs() -> None:
    cfg = _fixture_tune_cfg()
    grid = [{"max_depth": 2}]
    wf = cfg["walk_forward"]
    df = ml_walk_forward_tune(cfg, grid, n_folds=wf["n_folds"], metric="f1_macro")
    assert "score" in df.columns


def test_grid_id_stable() -> None:
    a = mwf._grid_id({"max_depth": 3, "z": 1})
    b = mwf._grid_id({"z": 1, "max_depth": 3})
    assert a == b


def test_aggregate_grid_raises_on_missing_column() -> None:
    with pytest.raises(ValueError, match="grid_id"):
        aggregate_grid(pd.DataFrame({"score": [1]}), "score")


def test_ml_walk_forward_tune_docstring() -> None:
    assert ml_walk_forward_tune.__doc__ and "Walk-forward" in ml_walk_forward_tune.__doc__


@pytest.mark.skipif(not _CFG.is_file(), reason="default ML config missing")
def test_multi_grid_point_scores_two_ids() -> None:
    cfg = _fixture_tune_cfg()
    grid = [{"max_depth": 2, "n_estimators": 8}, {"max_depth": 3, "n_estimators": 8}]
    wf = cfg["walk_forward"]
    df = ml_walk_forward_tune(cfg, grid, n_folds=wf["n_folds"], metric="accuracy")
    by_gid = df.groupby("grid_id")["score"].mean()
    assert len(by_gid) == 2


@pytest.mark.skipif(not _CFG.is_file(), reason="default ML config missing")
def test_aggregate_from_tune_output() -> None:
    cfg = _fixture_tune_cfg()
    grid = [{"max_depth": 2}]
    wf = cfg["walk_forward"]
    df = ml_walk_forward_tune(cfg, grid, n_folds=wf["n_folds"], metric="accuracy")
    agg = aggregate_grid(df, "score")
    assert len(agg) == 1


def test_import_ml_walk_forward_public_api() -> None:
    from aprilalgo.tuner.ml_walk_forward import ml_walk_forward_tune as tune

    assert callable(tune)


def test_score_from_purged_mean_unknown_raises() -> None:
    from aprilalgo.tuner.ml_walk_forward import _score_from_purged_mean

    with pytest.raises(ValueError, match="Unknown metric"):
        _score_from_purged_mean({}, "bogus", "binary")
