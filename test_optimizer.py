"""Тесты модуля src/optimizer."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.optimizer import solve_min_variance, solve_mvo, soft_blend


def test_mvo_weights_sum_to_one(mvo_analytic_case):
    mu, sigma = mvo_analytic_case
    w = solve_mvo(mu, sigma, lambda_risk=2.0, min_weight=0.0, max_weight=1.0)
    assert abs(w.sum() - 1.0) < 1e-6
    assert (w >= -1e-8).all()


def test_min_variance_lower_vol_than_equal():
    rng = np.random.default_rng(0)
    n = 4
    A = rng.normal(size=(n, n))
    sigma = A @ A.T + 0.01 * np.eye(n)
    w_min = solve_min_variance(sigma, 0, 1)
    eq = np.ones(n) / n
    var_min = w_min @ sigma @ w_min
    var_eq = eq @ sigma @ eq
    assert var_min <= var_eq + 1e-8


def test_soft_blend_simplex():
    tickers = ["A", "B", "C"]
    rw = {0: pd.Series([0.5, 0.3, 0.2], index=tickers),
          1: pd.Series([0.1, 0.4, 0.5], index=tickers)}
    pi = np.array([0.7, 0.3])
    blended = soft_blend(rw, pi)
    expected = 0.7 * rw[0] + 0.3 * rw[1]
    np.testing.assert_allclose(blended.values, expected.values, atol=1e-9)


def test_mvo_max_weight_constraint():
    rng = np.random.default_rng(1)
    n = 5
    mu = rng.normal(0.001, 0.0005, size=n)
    A = rng.normal(size=(n, n))
    sigma = A @ A.T + 0.01 * np.eye(n)
    w = solve_mvo(mu, sigma, lambda_risk=1.0, min_weight=0, max_weight=0.3)
    assert (w <= 0.30 + 1e-6).all()
    assert (w >= -1e-8).all()
    assert abs(w.sum() - 1.0) < 1e-6
