"""Интеграционный сквозной тест полного пайплайна."""
from __future__ import annotations

from datetime import date

from src.backtest import Backtester, EqualWeightStrategy, ImoexStrategy
from src.data import MoexDataLoader
from src.optimizer import RegimeOptimizer
from src.regime import GaussianHMMRegimeModel


def test_end_to_end_pipeline(tmp_path):
    loader = MoexDataLoader(cache_dir=str(tmp_path))
    bundle = loader.load(
        ["A", "B", "C", "D"],
        start=date(2022, 1, 1), end=date(2024, 6, 1),
        force_synthetic=True,
    )
    model = GaussianHMMRegimeModel(k_regimes=2)
    model.fit(bundle.returns.mean(axis=1))

    opt = RegimeOptimizer(lambda_risk=2.0, max_weight=0.5)
    opt.fit(bundle, model)

    bt = Backtester(train_window=200, test_window=20, step=20)
    result = bt.run(
        bundle, GaussianHMMRegimeModel, opt,
        strategies=[EqualWeightStrategy(), ImoexStrategy()],
        regime_model_kwargs={"k_regimes": 2},
    )
    assert "Adaptive" in result.equity_curves.columns
    assert "EW" in result.equity_curves.columns
    assert len(result.equity_curves) > 0
    assert "sharpe_ratio" in result.metrics.columns
