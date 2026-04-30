"""Главный класс RegimeOptimizer."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..data.loader import DataBundle
from ..regime.base import RegimeModel
from ..utils.exceptions import OptimizationFailedError
from .base import PortfolioWeights
from .cvar import solve_cvar
from .mvo import solve_max_sharpe, solve_min_variance, solve_mvo
from .soft_weighting import soft_blend

logger = logging.getLogger(__name__)


class RegimeOptimizer:
    """Адаптивный режимный оптимизатор портфеля."""

    def __init__(
        self,
        lambda_risk: float = 2.0,
        use_cvar: bool = False,
        cvar_confidence: float = 0.95,
        n_cvar_scenarios: int = 1000,
        max_weight: float = 0.30,
        min_weight: float = 0.0,
        sector_caps: Optional[Dict[str, float]] = None,
        estimation_window: int = 252,
        rebalance_threshold: float = 0.02,
        min_rebalance_days: int = 5,
        tc_bps: float = 10.0,
        random_state: int = 42,
    ):
        if lambda_risk <= 0:
            raise ValueError("lambda_risk должен быть > 0")
        self.lambda_risk = float(lambda_risk)
        self.use_cvar = bool(use_cvar)
        self.cvar_confidence = float(cvar_confidence)
        self.n_cvar_scenarios = int(n_cvar_scenarios)
        self.max_weight = float(max_weight)
        self.min_weight = float(min_weight)
        self.sector_caps = sector_caps
        self.estimation_window = int(estimation_window)
        self.rebalance_threshold = float(rebalance_threshold)
        self.min_rebalance_days = int(min_rebalance_days)
        self.tc_bps = float(tc_bps)
        self.random_state = int(random_state)

        self._tickers: Optional[list] = None
        self._regime_weights: Dict[int, pd.Series] = {}
        self._regime_mu: Dict[int, np.ndarray] = {}
        self._regime_sigma: Dict[int, np.ndarray] = {}
        self._k_regimes: int = 0
        self._fitted = False

    # --------------------------------------------------------------- helpers
    def _shrink_cov(self, returns: pd.DataFrame, fallback_cov: np.ndarray) -> np.ndarray:
        from sklearn.covariance import LedoitWolf
        if len(returns) < returns.shape[1] + 2:
            return fallback_cov
        lw = LedoitWolf().fit(returns.values)
        return np.asarray(lw.covariance_)

    def _solve_for_regime(
        self,
        k: int,
        mu_k: np.ndarray,
        sigma_k: np.ndarray,
        scenarios: np.ndarray,
    ) -> np.ndarray:
        n = len(mu_k)
        max_w = max(self.max_weight, 1.0 / n + 1e-6)
        if self.use_cvar:
            try:
                return solve_cvar(
                    scenarios,
                    confidence=self.cvar_confidence,
                    min_weight=self.min_weight,
                    max_weight=max_w,
                )
            except OptimizationFailedError as exc:
                logger.warning("CVaR не сошёлся для k=%d: %s. Падаем на MVO.", k, exc)
        return solve_mvo(
            mu_k,
            sigma_k,
            lambda_risk=self.lambda_risk,
            min_weight=self.min_weight,
            max_weight=max_w,
        )

    # --------------------------------------------------------------- API
    def fit(self, data_bundle: DataBundle, regime_model: RegimeModel) -> "RegimeOptimizer":
        returns_df = data_bundle.returns
        if returns_df.shape[1] < 2:
            raise ValueError("Нужно минимум 2 актива.")
        self._tickers = list(returns_df.columns)
        K = regime_model.k_regimes
        self._k_regimes = K

        equal_weighted = returns_df.mean(axis=1)
        # Используем тот ряд, на котором модель уже была обучена
        try:
            filtered = regime_model.get_filtered_proba(equal_weighted)
        except Exception:
            filtered = regime_model.get_filtered_proba()

        # Глобальная ковариация — для shrinkage при малой выборке
        global_cov = np.cov(returns_df.tail(self.estimation_window).values.T)

        rng = np.random.default_rng(self.random_state)
        for k in range(K):
            mask = filtered.idxmax(axis=1) == f"regime_{k}"
            r_k = returns_df.loc[mask].tail(self.estimation_window)
            n_assets = returns_df.shape[1]
            n_min = max(n_assets + 2, 30)
            if len(r_k) < n_min:
                # fallback: используем глобальные данные с blending
                frac = max(len(r_k) / n_min, 0.2) if len(r_k) > 0 else 0.0
                global_recent = returns_df.tail(self.estimation_window)
                if len(r_k) > 0:
                    sigma_k = (
                        frac * np.cov(r_k.values.T, ddof=1)
                        + (1 - frac) * np.cov(global_recent.values.T, ddof=1)
                    )
                    mu_k = frac * r_k.mean().values + (1 - frac) * global_recent.mean().values
                    base_for_scenarios = pd.concat(
                        [r_k, global_recent.tail(max(n_min - len(r_k), 1))]
                    )
                else:
                    sigma_k = global_cov.copy()
                    mu_k = global_recent.mean().values
                    base_for_scenarios = global_recent
            else:
                sigma_k = self._shrink_cov(r_k, fallback_cov=global_cov)
                mu_k = r_k.mean().values
                base_for_scenarios = r_k

            scenarios = base_for_scenarios.values[
                rng.integers(0, len(base_for_scenarios), size=self.n_cvar_scenarios)
            ]

            try:
                w = self._solve_for_regime(k, mu_k, sigma_k, scenarios)
            except OptimizationFailedError as exc:
                logger.warning("Оптимизация для k=%d упала, использую равные веса: %s", k, exc)
                w = np.ones(n_assets) / n_assets

            self._regime_weights[k] = pd.Series(w, index=self._tickers)
            self._regime_mu[k] = mu_k
            self._regime_sigma[k] = sigma_k

        self._fitted = True
        return self

    def get_regime_weights(self, k: int) -> pd.Series:
        if not self._fitted:
            raise RuntimeError("Не обучено")
        return self._regime_weights[k].copy()

    def get_all_regime_weights(self) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Не обучено")
        df = pd.DataFrame({k: ws for k, ws in self._regime_weights.items()})
        df.columns = [f"regime_{k}" for k in df.columns]
        return df

    def compute_portfolio_weights(
        self,
        pi_t: np.ndarray,
        prev_weights: Optional[pd.Series] = None,
        days_since_rebalance: int = 0,
        timestamp: Optional[datetime] = None,
    ) -> PortfolioWeights:
        if not self._fitted:
            raise RuntimeError("Не обучено")
        pi_t = np.asarray(pi_t, dtype=float)
        if not np.isclose(pi_t.sum(), 1.0, atol=1e-6):
            pi_t = pi_t / pi_t.sum() if pi_t.sum() > 0 else np.ones_like(pi_t) / len(pi_t)
        w_t = soft_blend(self._regime_weights, pi_t)

        rebalanced = True
        applied = w_t.copy()
        if prev_weights is not None:
            delta = float((w_t - prev_weights).abs().sum())
            if delta < self.rebalance_threshold or days_since_rebalance < self.min_rebalance_days:
                rebalanced = False
                applied = prev_weights.copy()

        return PortfolioWeights(
            weights=applied,
            regime_proba=pi_t,
            timestamp=timestamp or datetime.now(),
            lambda_param=self.lambda_risk,
            rebalanced=rebalanced,
            regime_weights={k: v.copy() for k, v in self._regime_weights.items()},
            raw_weights=w_t,
        )

    # ------------------------------------------------------------- analytics
    def get_efficient_frontier(self, k: int, n_points: int = 50) -> pd.DataFrame:
        """Эффективная граница для режима k через λ-sweep.

        В выходной DataFrame дополнительно размещаются веса по каждой точке в столбцах
        w_<ticker> и в колонке 'weights' (dict ticker->вес) — для hover в интерфейсе.
        """
        if not self._fitted:
            raise RuntimeError("Не обучено")
        mu_k = self._regime_mu[k]
        sigma_k = self._regime_sigma[k]
        n = len(mu_k)
        max_w = max(self.max_weight, 1.0 / n + 1e-6)
        lambdas = np.geomspace(0.5, 50.0, n_points)
        rows = []
        last_w = None
        for lam in lambdas:
            try:
                w = solve_mvo(
                    mu_k, sigma_k, lambda_risk=float(lam),
                    min_weight=self.min_weight, max_weight=max_w,
                )
            except OptimizationFailedError:
                if last_w is None:
                    continue
                w = last_w
            ret = float(mu_k @ w)
            vol = float(np.sqrt(w @ sigma_k @ w))
            sharpe = ret / vol if vol > 1e-10 else 0.0
            row = {
                "return": ret,
                "volatility": vol,
                "sharpe": sharpe,
                "lambda": float(lam),
            }
            for tic, wi in zip(self._tickers, w):
                row[f"w_{tic}"] = float(wi)
            row["weights"] = {t: float(wi) for t, wi in zip(self._tickers, w)}
            rows.append(row)
            last_w = w
        return pd.DataFrame(rows)

    def get_regime_portfolio_stats(self, k: int) -> dict:
        """Годовые ожидаемая доходность, волатильность и коэффициент Шарпа режимного портфеля.

        Ожидаемая дневная доходность и дисперсия берутся по оценкам режима (окно estimation_window),
        затем масштабируются на 252 торговых дня — также, как в сводных характеристиках режима.
        """
        from ..utils.config import TRADING_DAYS_PER_YEAR
        if not self._fitted:
            raise RuntimeError("Не обучено")
        w = self._regime_weights[k].values
        mu_k = self._regime_mu[k]
        sigma_k = self._regime_sigma[k]
        ret_d = float(mu_k @ w)
        vol_d = float(np.sqrt(max(w @ sigma_k @ w, 0.0)))
        ret_a = ret_d * TRADING_DAYS_PER_YEAR
        vol_a = vol_d * np.sqrt(TRADING_DAYS_PER_YEAR)
        sharpe = ret_a / vol_a if vol_a > 1e-10 else 0.0
        return {
            "return_annual": ret_a,
            "vol_annual": vol_a,
            "sharpe": sharpe,
            "return_daily": ret_d,
            "vol_daily": vol_d,
        }

    def get_min_variance_weights(self, k: int) -> pd.Series:
        sigma_k = self._regime_sigma[k]
        n = sigma_k.shape[0]
        max_w = max(self.max_weight, 1.0 / n + 1e-6)
        w = solve_min_variance(sigma_k, self.min_weight, max_w)
        return pd.Series(w, index=self._tickers)

    def get_max_sharpe_weights(self, k: int, rf: float = 0.0) -> pd.Series:
        mu_k = self._regime_mu[k]
        sigma_k = self._regime_sigma[k]
        n = len(mu_k)
        max_w = max(self.max_weight, 1.0 / n + 1e-6)
        w = solve_max_sharpe(mu_k, sigma_k, rf=rf, min_weight=self.min_weight, max_weight=max_w)
        return pd.Series(w, index=self._tickers)
