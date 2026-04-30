"""Mean-Variance оптимизация и связанные служебные функции."""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ..utils.exceptions import OptimizationFailedError

logger = logging.getLogger(__name__)


def solve_mvo(
    mu: np.ndarray,
    sigma: np.ndarray,
    lambda_risk: float = 2.0,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> np.ndarray:
    """Решение задачи MVO: max μ'w - λ/2 * w'Σw, sum=1, lb<=w<=ub."""
    try:
        import cvxpy as cp
    except ImportError as exc:  # pragma: no cover
        raise ImportError("cvxpy не установлен.") from exc

    n = len(mu)
    sigma_psd = _project_psd(sigma)
    w = cp.Variable(n)
    objective = cp.Maximize(mu @ w - (lambda_risk / 2.0) * cp.quad_form(w, cp.psd_wrap(sigma_psd)))
    constraints = [cp.sum(w) == 1, w >= min_weight, w <= max_weight]
    prob = cp.Problem(objective, constraints)
    _solve_with_fallback(prob)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise OptimizationFailedError(prob.status)
    sol = np.asarray(w.value, dtype=float).flatten()
    sol = np.clip(sol, min_weight, max_weight)
    s = sol.sum()
    return sol / s if s > 0 else np.ones(n) / n


def solve_min_variance(
    sigma: np.ndarray,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> np.ndarray:
    try:
        import cvxpy as cp
    except ImportError as exc:  # pragma: no cover
        raise ImportError("cvxpy не установлен.") from exc

    n = sigma.shape[0]
    sigma_psd = _project_psd(sigma)
    w = cp.Variable(n)
    objective = cp.Minimize(cp.quad_form(w, cp.psd_wrap(sigma_psd)))
    constraints = [cp.sum(w) == 1, w >= min_weight, w <= max_weight]
    prob = cp.Problem(objective, constraints)
    _solve_with_fallback(prob)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise OptimizationFailedError(prob.status)
    sol = np.asarray(w.value, dtype=float).flatten()
    sol = np.clip(sol, min_weight, max_weight)
    return sol / sol.sum()


def solve_max_sharpe(
    mu: np.ndarray,
    sigma: np.ndarray,
    rf: float = 0.0,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> np.ndarray:
    """Касательный портфель через трансформацию Чарнса-Купера."""
    try:
        import cvxpy as cp
    except ImportError as exc:  # pragma: no cover
        raise ImportError("cvxpy не установлен.") from exc

    n = len(mu)
    sigma_psd = _project_psd(sigma)
    excess = mu - rf
    if np.all(excess <= 0):
        return solve_min_variance(sigma, min_weight, max_weight)

    z = cp.Variable(n)
    kappa = cp.Variable(nonneg=True)
    objective = cp.Minimize(cp.quad_form(z, cp.psd_wrap(sigma_psd)))
    constraints = [
        excess @ z == 1,
        cp.sum(z) == kappa,
        z >= min_weight * kappa,
        z <= max_weight * kappa,
        kappa >= 1e-8,
    ]
    prob = cp.Problem(objective, constraints)
    _solve_with_fallback(prob)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        # фолбэк — равные веса
        return np.ones(n) / n
    z_val = np.asarray(z.value, dtype=float).flatten()
    s = z_val.sum()
    if s <= 0:
        return np.ones(n) / n
    sol = z_val / s
    sol = np.clip(sol, min_weight, max_weight)
    return sol / sol.sum()


def _project_psd(sigma: np.ndarray) -> np.ndarray:
    """Проектирует ковариационную матрицу на пространство PSD."""
    sigma = (sigma + sigma.T) / 2
    eigvals, eigvecs = np.linalg.eigh(sigma)
    eigvals = np.where(eigvals < 1e-10, 1e-10, eigvals)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def _solve_with_fallback(prob) -> None:
    """Пытается решить задачу разными solver-ами."""
    import cvxpy as cp

    solvers = []
    for cand in ("CLARABEL", "ECOS", "SCS", "OSQP"):
        if cand in cp.installed_solvers():
            solvers.append(cand)
    last_exc: Optional[Exception] = None
    for s in solvers:
        try:
            prob.solve(solver=s, verbose=False)
            if prob.status in ("optimal", "optimal_inaccurate"):
                return
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning("Solver %s failed: %s", s, exc)
    if prob.status not in ("optimal", "optimal_inaccurate") and last_exc is not None:
        # последняя попытка с дефолтным
        try:
            prob.solve(verbose=False)
        except Exception:  # noqa: BLE001
            raise last_exc
