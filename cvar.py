"""CVaR-оптимизация через CVXPY (Rockafellar-Uryasev)."""
from __future__ import annotations

import numpy as np

from ..utils.exceptions import OptimizationFailedError
from .mvo import _solve_with_fallback


def solve_cvar(
    scenarios: np.ndarray,
    confidence: float = 0.95,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    expected_return: float | None = None,
    mu: np.ndarray | None = None,
) -> np.ndarray:
    """Минимизация CVaR_α; опционально с минимальной ожидаемой доходностью."""
    try:
        import cvxpy as cp
    except ImportError as exc:  # pragma: no cover
        raise ImportError("cvxpy не установлен.") from exc

    n_scen, n_assets = scenarios.shape
    w = cp.Variable(n_assets)
    zeta = cp.Variable()
    losses = -scenarios @ w
    cvar_obj = zeta + (1.0 / ((1.0 - confidence) * n_scen)) * cp.sum(cp.pos(losses - zeta))
    objective = cp.Minimize(cvar_obj)
    constraints = [cp.sum(w) == 1, w >= min_weight, w <= max_weight]
    if expected_return is not None and mu is not None:
        constraints.append(mu @ w >= expected_return)
    prob = cp.Problem(objective, constraints)
    _solve_with_fallback(prob)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise OptimizationFailedError(prob.status)
    sol = np.asarray(w.value, dtype=float).flatten()
    sol = np.clip(sol, min_weight, max_weight)
    return sol / sol.sum()
