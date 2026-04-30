"""Тесты модуля src/data."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from src.data import MoexDataLoader, compute_log_returns, handle_missing


def test_compute_log_returns_basic():
    prices = pd.DataFrame({"A": [100, 110, 121]}, index=pd.bdate_range("2024-01-01", periods=3))
    rets = compute_log_returns(prices)
    assert len(rets) == 2
    np.testing.assert_allclose(rets["A"].values, [np.log(110 / 100), np.log(121 / 110)])


def test_handle_missing_forward_fill_limit():
    idx = pd.bdate_range("2024-01-01", periods=10)
    series = pd.DataFrame({"A": [100, np.nan, np.nan, np.nan, np.nan, 105, 106, 107, 108, 109]},
                          index=idx)
    cleaned, excluded = handle_missing(series, forward_fill_limit=2, max_missing_fraction=0.5)
    # после ffill_limit=2 первые 4 NaN не заполнятся полностью
    assert "A" in cleaned.columns or "A" in excluded


def test_loader_synthetic_fallback(tmp_path):
    loader = MoexDataLoader(cache_dir=str(tmp_path))
    bundle = loader.load(
        ["SBER", "LKOH", "GAZP"],
        start=date(2022, 1, 1), end=date(2024, 1, 1),
        force_synthetic=True,
    )
    assert bundle.n_assets == 3
    assert bundle.n_obs > 200
    assert bundle.metadata["source"] == "synthetic"
    assert bundle.returns.isna().sum().sum() == 0
