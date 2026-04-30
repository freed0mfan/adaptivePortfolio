"""Статистические тесты для бэктеста."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats


class StatisticsModule:
    """t-тест α, блочный бутстрап Шарпа и тест Дибольда–Мариано."""

    @staticmethod
    def alpha_ttest(
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> Dict[str, float]:
        common = strategy_returns.index.intersection(benchmark_returns.index)
        if len(common) < 10:
            return {"alpha": float("nan"), "p_value": float("nan"), "t_stat": float("nan")}
        rs = strategy_returns.loc[common].values
        rb = benchmark_returns.loc[common].values
        X = np.column_stack([np.ones(len(rb)), rb])
        try:
            beta_hat, _, _, _ = np.linalg.lstsq(X, rs, rcond=None)
        except Exception:
            return {"alpha": 0.0, "p_value": 1.0, "t_stat": 0.0}
        residuals = rs - X @ beta_hat
        n = len(rs)
        sigma2 = (residuals @ residuals) / max(n - 2, 1)
        XtX_inv = np.linalg.inv(X.T @ X)
        se_alpha = float(np.sqrt(sigma2 * XtX_inv[0, 0]))
        alpha = float(beta_hat[0])
        t_stat = alpha / se_alpha if se_alpha > 1e-12 else 0.0
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=max(n - 2, 1)))
        return {
            "alpha": alpha,
            "alpha_annual": alpha * 252,
            "t_stat": float(t_stat),
            "p_value": float(p_value),
            "se_alpha": se_alpha,
            "ci_low_95": float(alpha - 1.96 * se_alpha),
            "ci_high_95": float(alpha + 1.96 * se_alpha),
        }

    @staticmethod
    def sharpe_bootstrap_ci(
        returns: pd.Series,
        n_bootstrap: int = 1000,
        block_size: int = 22,
        confidence: float = 0.95,
        rf: float = 0.0,
        random_state: int = 42,
    ) -> Tuple[float, float, float]:
        r = returns.dropna().values
        T = len(r)
        if T < block_size * 2:
            return float("nan"), float("nan"), float("nan")
        rng = np.random.default_rng(random_state)
        n_blocks = int(np.ceil(T / block_size))
        sharpes = np.empty(n_bootstrap)
        for i in range(n_bootstrap):
            starts = rng.integers(0, T - block_size + 1, size=n_blocks)
            sample = np.concatenate([r[s : s + block_size] for s in starts])[:T]
            std = sample.std(ddof=1)
            if std < 1e-12:
                sharpes[i] = 0.0
            else:
                sharpes[i] = (sample.mean() - rf) / std * np.sqrt(252)
        alpha = (1.0 - confidence) / 2.0
        lower = float(np.quantile(sharpes, alpha))
        upper = float(np.quantile(sharpes, 1 - alpha))
        point = float(sharpes.mean())
        return point, lower, upper

    @staticmethod
    def diebold_mariano_test(
        e1: pd.Series,
        e2: pd.Series,
        h: int = 1,
    ) -> Dict[str, float]:
        common = e1.index.intersection(e2.index)
        if len(common) < 10:
            return {"dm_stat": float("nan"), "p_value": float("nan")}
        d = (e1.loc[common].values ** 2) - (e2.loc[common].values ** 2)
        n = len(d)
        d_mean = d.mean()
        # HAC variance with Newey-West kernel up to h-1 lags
        gamma0 = ((d - d_mean) ** 2).sum() / n
        var = gamma0
        for k in range(1, h):
            gk = ((d[k:] - d_mean) * (d[:-k] - d_mean)).sum() / n
            var += 2 * (1 - k / h) * gk
        var = max(var, 1e-12)
        dm_stat = d_mean / np.sqrt(var / n)
        p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
        return {"dm_stat": float(dm_stat), "p_value": float(p_value), "mean_loss_diff": float(d_mean)}
