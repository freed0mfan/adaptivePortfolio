"""Markov-Switching GARCH(1,1) в спецификации Haas, Mittnik, Paolella (2004).

Каждому режиму k = 1..K соответствует независимая GARCH(1,1) рекурсия

    sigma2_{k,t} = omega_k + alpha_k * eps_{t-1}^2 + beta_k * sigma2_{k,t-1},

а наблюдение распределено как смесь нормальных N(mu_k, sigma2_{k,t}) с
весами s_t ~ марковская цепь с матрицей переходов P. Эта спецификация, в
отличие от Gray (1996), не требует усреднения дисперсии по режимам и
обеспечивает аналитически замкнутые рекурсии (path-independent).

Литература:
- Haas M., Mittnik S., Paolella M. S. (2004). A New Approach to Markov-Switching
  GARCH Models. Journal of Financial Econometrics, 2(4), 493-530.
- Hamilton J. D. (1989). A New Approach to the Economic Analysis of
  Nonstationary Time Series and the Business Cycle. Econometrica.
- Kim C.-J. (1994). Dynamic linear models with Markov-switching.

Оценка: EM-алгоритм. На E-шаге считаются Hamilton filter и Kim smoother
по апостериорным вероятностям режимов; на M-шаге обновляются параметры
(mu_k, omega_k, alpha_k, beta_k) численной оптимизацией концентрированной
логарифмической функции правдоподобия каждого режима, а матрица переходов
P пересчитывается из сглаженных совместных вероятностей.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

from ..utils.exceptions import (
    ConvergenceWarning,
    DegenerateRegimeError,
    InsufficientDataError,
)
from .base import RegimeModel, RegimeParams, stationary_distribution

logger = logging.getLogger(__name__)


@dataclass
class _GarchParams:
    """Параметры одной GARCH(1,1) ветви."""

    mu: float
    omega: float
    alpha: float
    beta: float

    def unconditional_var(self) -> float:
        denom = max(1.0 - self.alpha - self.beta, 1e-6)
        return float(self.omega / denom)


def _garch_recursion(
    params: _GarchParams, x: np.ndarray, sigma0: float
) -> np.ndarray:
    """Возвращает вектор условных дисперсий sigma2_{k,t} для всех t."""
    T = len(x)
    s2 = np.empty(T)
    s2[0] = sigma0
    eps0 = x[0] - params.mu
    # На первом шаге используем начальную безусловную дисперсию
    for t in range(1, T):
        eps_prev = x[t - 1] - params.mu
        s2[t] = params.omega + params.alpha * eps_prev * eps_prev + params.beta * s2[t - 1]
        if not np.isfinite(s2[t]) or s2[t] < 1e-12:
            s2[t] = 1e-12
    return s2


def _emission_density(x: np.ndarray, mu: float, sigma2: np.ndarray) -> np.ndarray:
    """Плотность нормального распределения N(mu, sigma2_t) по времени."""
    sd = np.sqrt(np.maximum(sigma2, 1e-12))
    return norm.pdf(x, loc=mu, scale=sd)


# -------------------------------------------------------------- E-step routines


def _hamilton_filter(
    densities: np.ndarray, P: np.ndarray, pi0: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Hamilton filter: возвращает p(s_t=k|F_t), p(s_t=k|F_{t-1}), log-likelihood.

    Parameters
    ----------
    densities : np.ndarray, shape (T, K)
        Условные плотности f(r_t | s_t=k, F_{t-1}).
    P : np.ndarray, shape (K, K)
        Матрица переходов: P[i, j] = P(s_t=j | s_{t-1}=i).
    pi0 : np.ndarray, shape (K,)
        Начальное распределение режимов.
    """
    T, K = densities.shape
    filt = np.zeros((T, K))
    pred = np.zeros((T, K))
    log_lik = 0.0
    pred[0] = pi0
    for t in range(T):
        if t > 0:
            pred[t] = filt[t - 1] @ P
        joint = pred[t] * densities[t]
        s = joint.sum()
        if s <= 0 or not np.isfinite(s):
            # Численное «прижатие» к стационарному распределению
            joint = pred[t] + 1e-300
            s = joint.sum()
        filt[t] = joint / s
        log_lik += np.log(s + 1e-300)
    return filt, pred, float(log_lik)


def _kim_smoother(
    filt: np.ndarray, pred: np.ndarray, P: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Kim smoother. Возвращает сглаженные p(s_t|F_T) и совместные p(s_t,s_{t+1}|F_T)."""
    T, K = filt.shape
    smooth = np.zeros_like(filt)
    smooth[-1] = filt[-1]
    joint = np.zeros((T - 1, K, K))
    for t in range(T - 2, -1, -1):
        # P(s_{t+1}=j | F_t) = pred[t+1, j]
        denom = pred[t + 1].copy()
        denom[denom < 1e-300] = 1e-300
        # совместная P(s_t=i, s_{t+1}=j | F_T)
        ratio = smooth[t + 1] / denom  # shape (K,)
        joint[t] = (filt[t][:, None] * P) * ratio[None, :]
        smooth[t] = joint[t].sum(axis=1)
        # численная нормировка
        smooth[t] = smooth[t] / max(smooth[t].sum(), 1e-300)
    return smooth, joint


# ----------------------------------------------------------- parameter packing


def _pack(p: _GarchParams) -> np.ndarray:
    return np.array([p.mu, np.log(p.omega), _logit(p.alpha), _logit(p.beta)])


def _unpack(theta: np.ndarray) -> _GarchParams:
    return _GarchParams(
        mu=float(theta[0]),
        omega=float(np.exp(np.clip(theta[1], -20.0, 5.0))),
        alpha=float(_sigmoid(theta[2]) * 0.99),
        beta=float(_sigmoid(theta[3]) * 0.99),
    )


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _logit(p: float) -> float:
    p = float(np.clip(p / 0.99, 1e-4, 1.0 - 1e-4))
    return float(np.log(p / (1.0 - p)))


# ------------------------------------------------------------- main estimator


class MSGARCHRegimeModel(RegimeModel):
    """MS-GARCH(1,1) в спецификации Haas–Mittnik–Paolella (2004).

    Parameters
    ----------
    k_regimes : int
        Число режимов K (>= 1; при K=1 модель вырождается в обычный GARCH(1,1)).
    max_iter : int
        Максимальное число итераций EM-алгоритма.
    tol : float
        Толерантность относительного прироста log-likelihood.
    random_state : int
        Сид для случайной инициализации.
    """

    def __init__(
        self,
        k_regimes: int = 2,
        max_iter: int = 80,
        tol: float = 1e-4,
        n_starts: int = 3,
        random_state: int = 42,
    ):
        if k_regimes < 1:
            raise ValueError("k_regimes должно быть >= 1")
        # Базовый класс требует >=2; обрабатываем K=1 здесь
        if k_regimes >= 2:
            super().__init__(k_regimes=k_regimes, random_state=random_state)
        else:
            self.k_regimes = 1
            self.random_state = int(random_state)
            self._fitted = False
            self._params = None
            self._returns_index = None
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.n_starts = max(int(n_starts), 1)
        self._garch: list[_GarchParams] = []
        self._P: Optional[np.ndarray] = None
        self._pi0: Optional[np.ndarray] = None
        self._sigma0: Optional[np.ndarray] = None
        self._scale: float = 1.0
        self._last_filtered: Optional[np.ndarray] = None
        self._last_sigma2: Optional[np.ndarray] = None  # последняя условная sigma2 по режимам
        self._last_eps: Optional[float] = None
        self._smoothed_cache: Optional[np.ndarray] = None
        self._filtered_cache: Optional[np.ndarray] = None

    # ------------------------------------------------------------- helpers
    def _to_numpy(self, returns: pd.Series) -> np.ndarray:
        return np.asarray(returns.values, dtype=float)

    def _initialise(self, x: np.ndarray, seed_offset: int = 0, spread: float = 4.0) -> None:
        """Квантильная инициализация режимов от низкой к высокой вол-сти.

        spread — отношение дисперсий высокого и низкого режимов; больше
        разнят режимы и помогают EM выйти из вырожденного решения.
        """
        K = self.k_regimes
        rng = np.random.default_rng(self.random_state + seed_offset)
        var0 = float(np.var(x))
        self._garch = []
        if K == 1:
            scales = np.array([var0])
        else:
            # геометрическая сетка от var0/sqrt(spread) до var0*sqrt(spread)
            ratio = np.geomspace(1.0 / np.sqrt(spread), np.sqrt(spread), K)
            scales = var0 * ratio
        for k in range(K):
            alpha0 = float(np.clip(0.08 + 0.03 * rng.standard_normal(), 0.02, 0.25))
            beta0 = float(np.clip(0.85 + 0.04 * rng.standard_normal(), 0.50, 0.95))
            target = scales[k]
            omega0 = max(target * (1.0 - alpha0 - beta0), 1e-10)
            mu0 = float(np.mean(x) + 0.0001 * rng.standard_normal())
            self._garch.append(_GarchParams(mu=mu0, omega=omega0, alpha=alpha0, beta=beta0))
        self._sigma0 = np.array([g.unconditional_var() for g in self._garch])
        if K == 1:
            self._P = np.array([[1.0]])
        else:
            # большая персистентность помогает разделить режимы
            P = np.full((K, K), 0.02 / max(K - 1, 1))
            np.fill_diagonal(P, 0.98)
            self._P = P
        self._pi0 = np.ones(K) / K

    def _all_sigma2(self, x: np.ndarray) -> np.ndarray:
        """Возвращает массив (T, K) условных дисперсий по режимам."""
        T = len(x)
        K = self.k_regimes
        out = np.empty((T, K))
        for k in range(K):
            out[:, k] = _garch_recursion(self._garch[k], x, self._sigma0[k])
        return out

    def _emission(self, x: np.ndarray, sigma2: np.ndarray) -> np.ndarray:
        T, K = sigma2.shape
        d = np.empty((T, K))
        for k in range(K):
            d[:, k] = _emission_density(x, self._garch[k].mu, sigma2[:, k])
        # numerical floor
        d = np.maximum(d, 1e-300)
        return d

    # ------------------------------------------------------------- M-step
    def _update_garch(self, x: np.ndarray, weights_k: np.ndarray, k: int) -> None:
        """Обновить параметры k-го GARCH максимизацией взвешенного log-likelihood."""
        sigma0_k = self._sigma0[k]

        def neg_ll(theta: np.ndarray) -> float:
            params = _unpack(theta)
            sigma2 = _garch_recursion(params, x, sigma0_k)
            sd = np.sqrt(np.maximum(sigma2, 1e-12))
            log_pdf = norm.logpdf(x, loc=params.mu, scale=sd)
            val = -float(np.sum(weights_k * log_pdf))
            if not np.isfinite(val):
                return 1e10
            return val

        theta0 = _pack(self._garch[k])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                res = minimize(
                    neg_ll, theta0, method="Nelder-Mead",
                    options={"maxiter": 80, "xatol": 1e-3, "fatol": 1e-3},
                )
                if res.success or res.fun < neg_ll(theta0):
                    self._garch[k] = _unpack(res.x)
            except Exception as exc:  # noqa: BLE001
                logger.debug("GARCH M-step k=%d failed: %s", k, exc)

    def _update_transitions(self, joint: np.ndarray, smooth: np.ndarray) -> None:
        K = self.k_regimes
        if K == 1:
            return
        # P[i, j] = sum_t joint[t, i, j] / sum_t smooth[t, i] (только t=0..T-2)
        num = joint.sum(axis=0)  # (K, K)
        den = smooth[:-1].sum(axis=0)  # (K,)
        den = np.where(den < 1e-12, 1e-12, den)
        P_new = num / den[:, None]
        # численно нормируем строки
        row_sum = P_new.sum(axis=1, keepdims=True)
        row_sum = np.where(row_sum < 1e-12, 1.0, row_sum)
        self._P = P_new / row_sum
        self._pi0 = smooth[0]

    def _run_em(self, x: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Один прогон EM из текущей инициализации. Возвращает (ll, sigma2, filt, pred, smooth)."""
        K = self.k_regimes
        prev_ll = -np.inf
        for it in range(self.max_iter):
            sigma2 = self._all_sigma2(x)
            d = self._emission(x, sigma2)
            filt, pred, ll = _hamilton_filter(d, self._P, self._pi0)
            if K == 1:
                smooth = filt.copy()
                joint = np.zeros((len(x) - 1, 1, 1))
            else:
                smooth, joint = _kim_smoother(filt, pred, self._P)
            for k in range(K):
                self._update_garch(x, smooth[:, k], k)
            self._update_transitions(joint, smooth)
            if it > 0 and abs(ll - prev_ll) < self.tol * max(abs(prev_ll), 1.0):
                break
            prev_ll = ll
        sigma2 = self._all_sigma2(x)
        d = self._emission(x, sigma2)
        filt, pred, ll = _hamilton_filter(d, self._P, self._pi0)
        if K == 1:
            smooth = filt.copy()
        else:
            smooth, _ = _kim_smoother(filt, pred, self._P)
        return float(ll), sigma2, filt, pred, smooth

    # ------------------------------------------------------------- public
    def fit(self, returns: pd.Series) -> "MSGARCHRegimeModel":
        if len(returns) < 60:
            raise InsufficientDataError(n_available=len(returns), required=60)
        x_raw = self._to_numpy(returns)
        # Масштабируем, чтобы EM был численно стабилен
        scale = max(np.std(x_raw), 1e-6)
        x = x_raw / scale
        self._scale = scale

        K = self.k_regimes
        # Несколько стартов с разными spread/seed; выбираем лучший по log-likelihood
        best = None  # (ll, garch_list, P, pi0, sigma0, sigma2, filt, pred, smooth)
        spreads = [4.0, 9.0, 16.0] if K >= 2 else [1.0]
        starts = max(self.n_starts, len(spreads))
        for s_idx in range(starts):
            spread = spreads[s_idx % len(spreads)]
            try:
                self._initialise(x, seed_offset=s_idx, spread=spread)
                ll, sigma2, filt, pred, smooth = self._run_em(x)
            except Exception as exc:  # noqa: BLE001
                logger.debug("start %d failed: %s", s_idx, exc)
                continue
            if not np.isfinite(ll):
                continue
            if best is None or ll > best[0]:
                best = (
                    ll,
                    [
                        _GarchParams(g.mu, g.omega, g.alpha, g.beta) for g in self._garch
                    ],
                    self._P.copy(),
                    self._pi0.copy(),
                    self._sigma0.copy(),
                    sigma2,
                    filt,
                    pred,
                    smooth,
                )
        if best is None:
            raise RuntimeError("MS-GARCH EM не сошёлся ни на одном старте.")
        ll, garch_list, P, pi0, sigma0, sigma2, filt, pred, smooth = best
        self._garch = garch_list
        self._P = P
        self._pi0 = pi0
        self._sigma0 = sigma0

        # Параметры модели — обратное масштабирование
        # x_scaled = x_raw / scale  ⇒  mu_raw = mu * scale, sigma2_raw = sigma2 * scale^2
        means = np.array([g.mu for g in self._garch]) * scale
        # «безусловная» дисперсия на режим — для интерпретации
        uncond_var = np.array([g.unconditional_var() for g in self._garch]) * (scale ** 2)
        # GARCH-параметры в исходном масштабе: omega_raw = omega * scale^2, alpha и beta безразмерны
        for g in self._garch:
            g.mu = g.mu * scale
            g.omega = g.omega * (scale ** 2)
        # σ²_t тоже скейлится
        sigma2_raw = sigma2 * (scale ** 2)
        self._last_sigma2 = sigma2_raw[-1].copy()
        self._last_eps = float(x_raw[-1] - means[np.argmax(filt[-1])])
        # сохраняем кэши
        self._filtered_cache = filt.copy()
        self._smoothed_cache = smooth.copy()
        self._last_filtered = filt[-1].copy()
        self._returns_index = returns.index

        # Информационные критерии
        # Параметров: K*(mu, omega, alpha, beta) + K*(K-1) для P
        n_params = 4 * K + (K * (K - 1) if K > 1 else 0)
        T = len(x)
        aic = -2.0 * ll + 2.0 * n_params
        bic = -2.0 * ll + n_params * np.log(T)
        # переход в исходный масштаб поправляет ll константой -T*log(scale)
        ll_raw = ll - T * np.log(scale)

        # стационарное распределение
        if K == 1:
            stat = np.array([1.0])
        else:
            stat = stationary_distribution(self._P)

        # вырожденные режимы (нулевая безусловная дисперсия)
        for k in range(K):
            if uncond_var[k] < 1e-14:
                raise DegenerateRegimeError(k)

        params = RegimeParams(
            k=K,
            mu=means,
            sigma2=uncond_var,
            transition_matrix=self._P.copy(),
            stationary_dist=stat,
            log_likelihood=float(ll_raw),
            aic=float(aic),
            bic=float(bic),
            n_obs=int(T),
            extra={
                "spec": "Haas-Mittnik-Paolella (2004), GARCH(1,1) per regime",
                "omega": [g.omega for g in self._garch],
                "alpha": [g.alpha for g in self._garch],
                "beta": [g.beta for g in self._garch],
                "persistence": [g.alpha + g.beta for g in self._garch],
                "scale": float(scale),
            },
        )
        self._params = params
        self._fitted = True
        return self

    # --------------------------------------------------------- inference API
    def get_filtered_proba(self, returns: Optional[pd.Series] = None) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Модель не обучена.")
        f = self._filtered_cache
        idx = returns.index if returns is not None else self._returns_index
        if len(idx) != f.shape[0]:
            idx = idx[-f.shape[0]:]
        cols = [f"regime_{k}" for k in range(self.k_regimes)]
        return pd.DataFrame(f, index=idx, columns=cols)

    def get_smoothed_proba(self, returns: Optional[pd.Series] = None) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Модель не обучена.")
        s = self._smoothed_cache
        idx = returns.index if returns is not None else self._returns_index
        if len(idx) != s.shape[0]:
            idx = idx[-s.shape[0]:]
        cols = [f"regime_{k}" for k in range(self.k_regimes)]
        return pd.DataFrame(s, index=idx, columns=cols)

    def predict_next(self, last_return: float) -> np.ndarray:
        """Один шаг рекурсивного прогноза. Обновляет σ² и фильтрованную вероятность."""
        if not self._fitted:
            raise RuntimeError("Не обучено")
        K = self.k_regimes
        # Обновляем условные дисперсии
        new_sigma2 = np.empty(K)
        for k in range(K):
            g = self._garch[k]
            eps2 = (last_return - g.mu) ** 2
            new_sigma2[k] = g.omega + g.alpha * eps2 + g.beta * self._last_sigma2[k]
            if not np.isfinite(new_sigma2[k]) or new_sigma2[k] < 1e-12:
                new_sigma2[k] = max(self._last_sigma2[k], 1e-12)
        self._last_sigma2 = new_sigma2

        # Hamilton-шаг прогноза + обновление по последнему наблюдению
        if K == 1:
            self._last_filtered = np.array([1.0])
            return self._last_filtered.copy()
        pi_pred = self._last_filtered @ self._P
        d = np.array(
            [
                norm.pdf(last_return, loc=self._garch[k].mu, scale=np.sqrt(new_sigma2[k]))
                for k in range(K)
            ]
        )
        joint = pi_pred * d
        s = joint.sum()
        if s <= 0 or not np.isfinite(s):
            self._last_filtered = pi_pred / pi_pred.sum()
        else:
            self._last_filtered = joint / s
        return self._last_filtered.copy()
