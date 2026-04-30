"""Бенчмарк-стратегии для бэктеста."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd

from ..optimizer.mvo import solve_max_sharpe, solve_min_variance, solve_mvo


class BaseStrategy(ABC):
    """Стратегия с фиксированными весами на тестовом окне."""

    name: str = "base"

    @abstractmethod
    def compute_weights(
        self,
        returns: pd.DataFrame,
        as_of: Optional[pd.Timestamp] = None,
        index_weights: Optional[pd.Series] = None,
    ) -> pd.Series:
        ...


class EqualWeightStrategy(BaseStrategy):
    name = "EW"

    def compute_weights(self, returns, as_of=None, index_weights=None):
        n = returns.shape[1]
        return pd.Series(np.ones(n) / n, index=returns.columns)


class ImoexStrategy(BaseStrategy):
    """Маркер-стратегия.

    Индекс МосБиржи — взвешенный по капитализации бенчмарк, доходность которого нельзя
    реплицировать по весам выбранного подмножества тикеров. Бэктестер использует в качестве
    дневных доходностей IMOEX ряд ``data_bundle.index_returns`` (взятый из MOEX ISS API).
    Этот класс присутствует лишь для регистрации имени «IMOEX» в наборе бенчмарков.
    """

    name = "IMOEX"

    def compute_weights(self, returns, as_of=None, index_weights=None):
        # Веса не используются: доходность берётся напрямую из bundle.index_returns
        n = returns.shape[1]
        return pd.Series(np.ones(n) / n, index=returns.columns)


class StaticMVOStrategy(BaseStrategy):
    name = "MVO_static"

    def __init__(self, lambda_risk: float = 2.0, max_weight: float = 0.30):
        self.lambda_risk = lambda_risk
        self.max_weight = max_weight

    def compute_weights(self, returns, as_of=None, index_weights=None):
        n = returns.shape[1]
        max_w = max(self.max_weight, 1.0 / n + 1e-6)
        mu = returns.mean().values
        sigma = np.cov(returns.values.T, ddof=1)
        try:
            w = solve_mvo(mu, sigma, self.lambda_risk, 0.0, max_w)
        except Exception:
            w = np.ones(n) / n
        return pd.Series(w, index=returns.columns)


class RiskParityStrategy(BaseStrategy):
    name = "RiskParity"

    def __init__(self, window: int = 63):
        self.window = window

    def compute_weights(self, returns, as_of=None, index_weights=None):
        win = min(self.window, len(returns))
        rolling = returns.tail(win).std()
        rolling = rolling.replace(0, np.nan).fillna(rolling.mean())
        inv = 1.0 / rolling
        return inv / inv.sum()


class MinVarianceStrategy(BaseStrategy):
    name = "MinVar"

    def __init__(self, max_weight: float = 0.30):
        self.max_weight = max_weight

    def compute_weights(self, returns, as_of=None, index_weights=None):
        n = returns.shape[1]
        max_w = max(self.max_weight, 1.0 / n + 1e-6)
        sigma = np.cov(returns.values.T, ddof=1)
        try:
            w = solve_min_variance(sigma, 0.0, max_w)
        except Exception:
            w = np.ones(n) / n
        return pd.Series(w, index=returns.columns)


class MaxSharpeStrategy(BaseStrategy):
    name = "MaxSharpe"

    def __init__(self, rf: float = 0.16 / 252, max_weight: float = 0.30):
        self.rf = rf
        self.max_weight = max_weight

    def compute_weights(self, returns, as_of=None, index_weights=None):
        n = returns.shape[1]
        max_w = max(self.max_weight, 1.0 / n + 1e-6)
        mu = returns.mean().values
        sigma = np.cov(returns.values.T, ddof=1)
        try:
            w = solve_max_sharpe(mu, sigma, self.rf, 0.0, max_w)
        except Exception:
            w = np.ones(n) / n
        return pd.Series(w, index=returns.columns)


def build_strategy(name: str, **kwargs) -> BaseStrategy:
    name_norm = name.strip()
    mapping = {
        "EW": EqualWeightStrategy,
        "IMOEX": ImoexStrategy,
        "MVO_static": StaticMVOStrategy,
        "RiskParity": RiskParityStrategy,
        "MinVar": MinVarianceStrategy,
        "MaxSharpe": MaxSharpeStrategy,
    }
    cls = mapping.get(name_norm)
    if cls is None:
        raise ValueError(f"Неизвестная стратегия: {name}")
    return cls(**kwargs)


ALL_BENCHMARKS = ["IMOEX", "EW", "MVO_static", "RiskParity", "MinVar", "MaxSharpe"]
