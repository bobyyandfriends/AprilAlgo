"""Tests for the parameter tuning engine."""

import pytest
import pandas as pd
from aprilalgo.data import load_price_data
from aprilalgo.tuner import ParameterGrid, TunerRunner, analyze_results
from aprilalgo.strategies import STRATEGIES


@pytest.fixture
def price_data():
    return load_price_data("AAPL", "daily")


class TestParameterGrid:

    def test_single_indicator(self):
        g = ParameterGrid()
        g.add("rsi", rsi_period=[12, 14])
        assert g.total_combinations == 2

    def test_multi_indicator(self):
        g = ParameterGrid()
        g.add("rsi", rsi_period=[12, 14])
        g.add("sma", sma_period=[20, 50])
        assert g.total_combinations == 4
        combos = g.generate()
        assert len(combos) == 4

    def test_empty_grid(self):
        g = ParameterGrid()
        assert g.total_combinations == 1
        combos = g.generate()
        assert len(combos) == 1


class TestTunerRunner:

    def test_single_combo(self, price_data):
        g = ParameterGrid()
        g.add("rsi", rsi_period=[14])
        g.add("sma", sma_period=[20])
        runner = TunerRunner(STRATEGIES["rsi_sma"], price_data, g)
        results = runner.run()
        assert len(results) == 1
        assert "sharpe_ratio" in results.columns

    def test_multi_combo(self, price_data):
        g = ParameterGrid()
        g.add("rsi", rsi_period=[12, 14])
        g.add("sma", sma_period=[20, 50])
        runner = TunerRunner(STRATEGIES["rsi_sma"], price_data, g)
        results = runner.run()
        assert len(results) == 4


class TestAnalyzer:

    def test_analyze_results(self, price_data):
        g = ParameterGrid()
        g.add("rsi", rsi_period=[12, 14])
        g.add("sma", sma_period=[20, 50])
        runner = TunerRunner(STRATEGIES["rsi_sma"], price_data, g)
        results = runner.run()
        analysis = analyze_results(results, metric="sharpe_ratio")
        assert "best" in analysis
        assert "top_n" in analysis
        assert "robustness" in analysis

    def test_empty_results(self):
        analysis = analyze_results(pd.DataFrame(), metric="sharpe_ratio")
        assert analysis["best"] == {}
