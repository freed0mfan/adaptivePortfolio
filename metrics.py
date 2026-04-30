"""Метрики производительности для портфельных стратегий."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ..utils.config import DEFAULT_RF_RATE_DAILY, TRADING_DAYS_PER_YEAR


class PerformanceMetrics:
    """Стандартные метрики бэктеста."""

    @staticmethod
    def equity_curve(returns: pd.Series) -> pd.Series:
        return np.exp(returns.fillna(0.0).cumsum())

    @staticmethod
    def max_drawdown(returns: pd.Series) -> float:
        eq = PerformanceMetrics.equity_curve(returns)
        peak = eq.cummax()
        dd = (eq - peak) / peak
        return float(dd.min())

    @staticmethod
    def compute(
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        rf: float = DEFAULT_RF_RATE_DAILY,
        trading_days: int = TRADING_DAYS_PER_YEAR,
        weights_history: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        r = returns.dropna()
        if len(r) == 0:
            return pd.Series(dtype=float)
        T = len(r)
        # Аннуализация:
        ann_return = float(np.exp(r.mean() * trading_days) - 1.0)
        ann_vol = float(r.std(ddof=1) * np.sqrt(trading_days))
        sharpe = ((r.mean() - rf) / r.std(ddof=1)) * np.sqrt(trading_days) if r.std(ddof=1) > 1e-10 else 0.0
        downside = r[r < 0]
        downside_vol = downside.std(ddof=1) * np.sqrt(trading_days) if len(downside) > 1 else np.nan
        sortino = float((ann_return - rf * trading_days) / downside_vol) if downside_vol and downside_vol > 1e-10 else 0.0
        mdd = PerformanceMetrics.max_drawdown(r)
        calmar = float(ann_return / abs(mdd)) if mdd < 0 else np.nan
        var95 = float(np.percentile(r, 5))
        cvar95 = float(r[r <= var95].mean()) if (r <= var95).any() else float(var95)

        result = {
            "ann_return": ann_return,
            "ann_volatility": ann_vol,
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "max_drawdown": float(mdd),
            "calmar_ratio": calmar,
            "var_95": float(var95 * np.sqrt(trading_days)),
            "cvar_95": float(cvar95 * np.sqrt(trading_days)),
            "n_obs": T,
        }

        if weights_history is not None and len(weights_history) > 1:
            turnover = float(weights_history.diff().abs().sum(axis=1).mean())
            result["turnover"] = turnover

        if benchmark_returns is not None:
            common = r.index.intersection(benchmark_returns.index)
            if len(common) > 5:
                rs = r.loc[common]
                rb = benchmark_returns.loc[common]
                # OLS α / β
                X = np.column_stack([np.ones(len(rb)), rb.values])
                y = rs.values
                try:
                    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
                    alpha = float(coef[0])
                    beta = float(coef[1])
                except Exception:
                    alpha, beta = 0.0, 1.0
                tracking_error = float((rs - rb).std(ddof=1) * np.sqrt(trading_days))
                info_ratio = (
                    float(alpha * trading_days / tracking_error)
                    if tracking_error > 1e-10
                    else 0.0
                )
                hit = float((rs > rb).mean())
                result.update(
                    {
                        "alpha_daily": alpha,
                        "alpha_annual": alpha * trading_days,
                        "beta": beta,
                        "tracking_error": tracking_error,
                        "info_ratio": info_ratio,
                        "hit_ratio": hit,
                    }
                )
        return pd.Series(result)
