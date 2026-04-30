"""GaussianHMM-обёртка из hmmlearn."""
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


class GaussianHMMRegimeModel(RegimeModel):
    """HMM с гауссовыми эмиссиями (унивариатный случай)."""

    def __init__(
        self,
        k_regimes: int = 2,
        covariance_type: str = "full",
        n_iter: int = 1000,
        random_state: int = 42,
    ):
        super().__init__(k_regimes=k_regimes, random_state=random_state)
        self.covariance_type = covariance_type
        self.n_iter = int(n_iter)
        self._model = None
        self._mu: Optional[np.ndarray] = None
        self._sigma2: Optional[np.ndarray] = None
        self._P: Optional[np.ndarray] = None
        self._last_filtered: Optional[np.ndarray] = None

    def fit(self, returns: pd.Series) -> "GaussianHMMRegimeModel":
        if len(returns) < 60:
            raise InsufficientDataError(n_available=len(returns), required=60)
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError as exc:  # pragma: no cover
            raise ImportError("hmmlearn не установлен.") from exc

        X = np.asarray(returns.values, dtype=float).reshape(-1, 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = GaussianHMM(
                n_components=self.k_regimes,
                covariance_type=self.covariance_type,
                n_iter=self.n_iter,
                random_state=self.random_state,
                tol=1e-5,
            )
            model.fit(X)

        if not getattr(model.monitor_, "converged", True):
            warnings.warn(
                "EM-алгоритм HMM не сошёлся.",
                ConvergenceWarning,
                stacklevel=2,
            )

        means = model.means_.flatten()
        if self.covariance_type == "full":
            covars = np.asarray([np.atleast_2d(c)[0, 0] for c in model.covars_])
        else:
            covars = np.asarray(model.covars_).flatten()
        sigma2 = np.where(covars < 1e-12, 1e-12, covars)

        for k in range(self.k_regimes):
            if np.sqrt(sigma2[k]) < 1e-6:
                raise DegenerateRegimeError(k)

        P = np.asarray(model.transmat_, dtype=float)
        P = P / P.sum(axis=1, keepdims=True)
        pi_inf = stationary_distribution(P)

        # AIC / BIC
        try:
            ll = float(model.score(X))
        except Exception:  # noqa: BLE001
            ll = float("nan")
        n_params = self.k_regimes - 1  # initial
        n_params += self.k_regimes * (self.k_regimes - 1)  # transitions
        n_params += 2 * self.k_regimes  # means + variances
        T = len(X)
        aic = -2 * ll + 2 * n_params
        bic = -2 * ll + n_params * np.log(T)

        self._model = model
        self._mu = means
        self._sigma2 = sigma2
        self._P = P
        self._returns_index = returns.index
        self._fitted = True
        self._params = RegimeParams(
            k=self.k_regimes,
            mu=means,
            sigma2=sigma2,
            transition_matrix=P,
            stationary_dist=pi_inf,
            log_likelihood=ll,
            aic=aic,
            bic=bic,
            n_obs=T,
        )
        try:
            self._last_filtered = model.predict_proba(X)[-1]
        except Exception:
            self._last_filtered = pi_inf.copy()
        return self

    def get_filtered_proba(self, returns: Optional[pd.Series] = None) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Не обучено")
        idx = returns.index if returns is not None else self._returns_index
        X = np.asarray(returns.values if returns is not None
                       else None, dtype=float)
        if returns is None:
            # пересчитать на исходном ряду
            raise ValueError("Передайте returns для предсказания.")
        proba = self._model.predict_proba(X.reshape(-1, 1))
        cols = [f"regime_{k}" for k in range(self.k_regimes)]
        return pd.DataFrame(proba, index=idx, columns=cols)

    def get_smoothed_proba(self, returns: Optional[pd.Series] = None) -> pd.DataFrame:
        # hmmlearn не различает filtered vs smoothed: predict_proba — апостериор по всей выборке.
        return self.get_filtered_proba(returns)

    def predict_next(self, last_return: float) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Не обучено")
        pi_pred = self._P.T @ self._last_filtered
        likelihoods = norm.pdf(last_return, loc=self._mu, scale=np.sqrt(self._sigma2))
        pi_new = likelihoods * pi_pred
        s = pi_new.sum()
        if s <= 0 or not np.isfinite(s):
            pi_new = pi_pred
        else:
            pi_new = pi_new / s
        self._last_filtered = pi_new
        return pi_new
