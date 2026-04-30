"""Тесты модуля src/backtest."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest import (
    EqualWeightStrategy,
    MaxSharpeStrategy,
    MinVarianceStrategy,
    PerformanceMetrics,
    RiskParityStrategy,
    StatisticsModule,
)


def test_metrics_basic(synthetic_multi_asset_returns):
    df = synthetic_multi_asset_returns
    eq_returns = df.mean(axis=1)
    metrics = PerformanceMetrics.compute(eq_returns)
    assert "sharpe_ratio" in metrics
    assert "max_drawdown" in metrics
    assert metrics["max_drawdown"] <= 0


def test_equal_weight_strategy_simplex(synthetic_multi_asset_returns):
    s = EqualWeightStrategy()
    w = s.compute_weights(synthetic_multi_asset_returns)
    assert abs(w.sum() - 1.0) < 1e-9
    assert (w >= 0).all()


def test_risk_parity_inversely_proportional(synthetic_multi_asset_returns):
    s = RiskParityStrategy(window=63)
    w = s.compute_weights(synthetic_multi_asset_returns)
    assert abs(w.sum() - 1.0) < 1e-9
    assert (w > 0).all()


def test_min_variance_strategy(synthetic_multi_asset_returns):
    w = MinVarianceStrategy().compute_weights(synthetic_multi_asset_returns)
    assert abs(w.sum() - 1.0) < 1e-6
    assert (w >= -1e-6).all()


def test_max_sharpe_strategy(synthetic_multi_asset_returns):
    w = MaxSharpeStrategy().compute_weights(synthetic_multi_asset_returns)
    assert abs(w.sum() - 1.0) < 1e-6


def test_alpha_ttest_zero_for_identical():
    rng = np.random.default_rng(42)
    r = pd.Series(rng.normal(0, 0.01, size=200), index=pd.bdate_range("2024-01-01", periods=200))
    out = StatisticsModule.alpha_ttest(r, r)
    assert abs(out["alpha"]) < 1e-10
