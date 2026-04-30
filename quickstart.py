"""Скрипт быстрого старта для проверки пайплайна без дашборда.

Запуск:
    python notebooks/quickstart.py

Использует синтетические данные (force_synthetic=True), чтобы работать без интернета.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import (
    Backtester,
    EqualWeightStrategy,
    ImoexStrategy,
    StaticMVOStrategy,
)
from src.data import MoexDataLoader
from src.optimizer import RegimeOptimizer
from src.regime import (
    GaussianHMMRegimeModel,
    MSARRegimeModel,
    MSGARCHRegimeModel,
    RegimeInterpreter,
    RegimeSelector,
)


def main() -> None:
    loader = MoexDataLoader()
    bundle = loader.load(
        ["SBER", "LKOH", "GAZP", "GMKN", "ROSN"],
        start=date(2020, 1, 1),
        end=date(2024, 12, 31),
        force_synthetic=True,
    )
    print(f"[data] loaded {bundle.n_assets} assets, {bundle.n_obs} obs, source={bundle.metadata['source']}")

    sel = RegimeSelector.select(
        bundle.returns.mean(axis=1), k_max=3, k_min=1,
        model_class=MSGARCHRegimeModel, criterion="aic",
    )
    print(f"[selector] AIC suggests K={sel.recommended_k}")

    model = MSGARCHRegimeModel(k_regimes=sel.recommended_k)
    model.fit(bundle.returns.mean(axis=1))
    summary = RegimeInterpreter.summarize(model.get_regime_params(), model.get_filtered_proba())
    print("[regimes]")
    print(summary.to_string(index=False))

    opt = RegimeOptimizer(lambda_risk=2.0, max_weight=0.40)
    opt.fit(bundle, model)
    print("[regime_weights]")
    print(opt.get_all_regime_weights().round(3))

    bt = Backtester(train_window=252, test_window=21, step=21)
    result = bt.run(
        bundle,
        MSGARCHRegimeModel,
        opt,
        strategies=[
            EqualWeightStrategy(),
            ImoexStrategy(),
            StaticMVOStrategy(),
        ],
        regime_model_kwargs={"k_regimes": max(sel.recommended_k, 1)},
    )
    print("[metrics]")
    print(result.metrics.round(3).to_string())


if __name__ == "__main__":
    main()
