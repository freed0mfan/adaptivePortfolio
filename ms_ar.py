"""Markov-Switching AR модель из statsmodels."""
from __future__ import annotations

import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..utils.exceptions import (
    ConvergenceWarning,
    DegenerateRegimeError,
    InsufficientDataError,
)
from .base import RegimeModel, RegimeParams, stationary_distribution

logger = logging.getLogger(__name__)


class MSARRegimeModel(RegimeModel):
    """MS-AR(p) с переключающейся дисперсией (statsmodels)."""

    def __init__(
        self,
        k_regimes: int = 2,
        order: int = 0,
        switching_variance: bool = True,
        max_iter: int = 500,
        tol: float = 1e-6,
        random_state: int = 42,
    ):
        super().__init__(k_regimes=k_regimes, random_state=random_state)
        self.order = int(order)
        self.switching_variance = bool(switching_variance)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self._result = None
        self._mu: Optional[np.ndarray] = None
        self._sigma2: Optional[np.ndarray] = None
        self._P: Optional[np.ndarray] = None
        self._last_filtered: Optional[np.ndarray] = None

    # ------------------------------------------------------------- internals
    def _to_numpy(self, returns: pd.Series) -> np.ndarray:
        return np.asarray(returns.values, dtype=float)

    def _extract_params(self, result, T: int) -> RegimeParams:
        K = self.k_regimes
        # Параметры: ищем по именам в param_names
        param_values = np.asarray(result.params)
        param_names = list(getattr(result.model, "param_names", []))
        if not param_names:
            param_names = list(getattr(result, "param_names", []))

        def _get(name: str, default: float = 0.0) -> float:
            if name in param_names:
                return float(param_values[param_names.index(name)])
            return default

        means = np.zeros(K)
        sigmas2 = np.zeros(K)
        for k in range(K):
            means[k] = _get(f"const[{k}]", _get(f"intercept[{k}]", 0.0))
            if self.switching_variance:
                sigmas2[k] = _get(f"sigma2[{k}]", float(np.var(result.model.endog)))
            else:
                sigmas2[k] = _get("sigma2", float(np.var(result.model.endog)))

        # Регуляризация: избегаем нулевой дисперсии
        sigmas2 = np.where(sigmas2 < 1e-10, 1e-10, sigmas2)

        # Проверка на вырожденность
        for k in range(K):
            if np.sqrt(sigmas2[k]) < 1e-6:
                raise DegenerateRegimeError(k)

        # Матрица переходов
        P = np.array(result.regime_transition).reshape(K, K)
        # statsmodels возвращает P[i,j] = P(s_t=j | s_{t-1}=i)
        # Нормализуем строки
        P = P / P.sum(axis=1, keepdims=True)
        pi_inf = stationary_distribution(P)

        return RegimeParams(
            k=K,
            mu=means,
            sigma2=sigmas2,
            transition_matrix=P,
            stationary_dist=pi_inf,
            log_likelihood=float(result.llf),
            aic=float(result.aic),
            bic=float(result.bic),
            n_obs=int(T),
        )

    # ------------------------------------------------------------- public API
    def fit(self, returns: pd.Series) -> "MSARRegimeModel":
        if len(returns) < 60:
            raise InsufficientDataError(n_available=len(returns), required=60)

        try:
            from statsmodels.tsa.regime_switching.markov_autoregression import (
                MarkovAutoregression,
            )
            from statsmodels.tsa.regime_switching.markov_regression import (
                MarkovRegression,
            )
        except ImportError as exc:  # pragma: no cover
            raise ImportError("statsmodels не установлен.") from exc

        x = self._to_numpy(returns)
        # Центрируем масштаб для стабильности EM
        scale = max(np.std(x), 1e-6)
        x_scaled = x / scale

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.order == 0:
                model = MarkovRegression(
                    endog=x_scaled,
                    k_regimes=self.k_regimes,
                    switching_variance=self.switching_variance,
                )
            else:
                model = MarkovAutoregression(
                    endog=x_scaled,
                    k_regimes=self.k_regimes,
                    order=self.order,
                    switching_variance=self.switching_variance,
                )
            try:
                result = model.fit(
                    em_iter=self.max_iter,
                    search_reps=1,
                    disp=False,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("MS-AR fit упал, фолбэк EM от случайного старта: %s", exc)
                rng = np.random.default_rng(self.random_state)
                start = rng.normal(0, 0.1, size=len(model.start_params))
                result = model.fit(
                    start_params=model.start_params + start,
                    em_iter=self.max_iter,
                    search_reps=2,
                    disp=False,
                )

        if getattr(result, "mle_retvals", {}).get("converged", True) is False:
            warnings.warn(
                "EM-алгоритм MS-AR не сошёлся за max_iter итераций.",
                ConvergenceWarning,
                stacklevel=2,
            )

        params = self._extract_params(result, T=len(x))
        # Обратное масштабирование mu и sigma2
        params.mu = params.mu * scale
        params.sigma2 = params.sigma2 * (scale ** 2)

        self._result = result
        self._params = params
        self._mu = params.mu
        self._sigma2 = params.sigma2
        self._P = params.transition_matrix
        self._returns_index = returns.index
        self._scale = scale
        self._fitted = True

        try:
            filtered = np.asarray(result.filtered_marginal_probabilities)
            if filtered.shape[0] != self.k_regimes:
                filtered = filtered.T
            self._last_filtered = filtered[:, -1]
        except Exception:  # noqa: BLE001
            self._last_filtered = params.stationary_dist.copy()
        return self

    def _filtered_array(self) -> np.ndarray:
        """Возвращает фильтрованные вероятности в форме (T, K)."""
        if self._result is None:
            raise RuntimeError("Не обучено")
        f = np.asarray(self._result.filtered_marginal_probabilities)
        if f.shape[0] == self.k_regimes:
            f = f.T
        return f

    def _smoothed_array(self) -> np.ndarray:
        if self._result is None:
            raise RuntimeError("Не обучено")
        f = np.asarray(self._result.smoothed_marginal_probabilities)
        if f.shape[0] == self.k_regimes:
            f = f.T
        return f

    def get_filtered_proba(self, returns: Optional[pd.Series] = None) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Модель не обучена.")
        f = self._filtered_array()
        idx = returns.index if returns is not None else self._returns_index
        if len(idx) != f.shape[0]:
            idx = idx[-f.shape[0]:]
        cols = [f"regime_{k}" for k in range(self.k_regimes)]
        return pd.DataFrame(f, index=idx, columns=cols)

    def get_smoothed_proba(self, returns: Optional[pd.Series] = None) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Модель не обучена.")
        f = self._smoothed_array()
        idx = returns.index if returns is not None else self._returns_index
        if len(idx) != f.shape[0]:
            idx = idx[-f.shape[0]:]
        cols = [f"regime_{k}" for k in range(self.k_regimes)]
        return pd.DataFrame(f, index=idx, columns=cols)

    def predict_next(self, last_return: float) -> np.ndarray:
        """Гамильтоновский шаг прогноза вероятностей для следующего наблюдения."""
        if not self._fitted:
            raise RuntimeError("Не обучено")
        pi_pred = self._P.T @ self._last_filtered
        likelihoods = norm.pdf(
            last_return, loc=self._mu, scale=np.sqrt(self._sigma2)
        )
        pi_updated = likelihoods * pi_pred
        s = pi_updated.sum()
        if s <= 0 or not np.isfinite(s):
            pi_updated = pi_pred
        else:
            pi_updated = pi_updated / s
        self._last_filtered = pi_updated
        return pi_updated
