"""Pytest-фикстуры с синтетическими данными."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_two_regime_returns():
    """Синтетический ряд с двумя режимами волатильности."""
    rng = np.random.default_rng(42)
    T = 600
    states = np.zeros(T, dtype=int)
    P = np.array([[0.97, 0.03], [0.10, 0.90]])
    for t in range(1, T):
        states[t] = rng.choice(2, p=P[states[t - 1]])
    sigmas = np.array([0.01, 0.04])
    mus = np.array([0.0005, -0.001])
    rets = rng.normal(mus[states], sigmas[states])
    idx = pd.bdate_range("2020-01-01", periods=T)
    return pd.Series(rets, index=idx, name="ret"), states


@pytest.fixture
def synthetic_multi_asset_returns():
    """Многомерный ряд с двумя режимами и общим рыночным фактором."""
    rng = np.random.default_rng(7)
    T = 500
    n = 5
    P = np.array([[0.97, 0.03], [0.10, 0.90]])
    states = np.zeros(T, dtype=int)
    for t in range(1, T):
        states[t] = rng.choice(2, p=P[states[t - 1]])
    sigmas = np.array([0.012, 0.035])
    mus = np.array([0.0006, -0.0008])
    market = rng.normal(mus[states], sigmas[states])
    betas = rng.uniform(0.6, 1.3, size=n)
    idio = rng.normal(0.0, 0.01, size=(T, n))
    rets = market[:, None] * betas[None, :] + idio
    idx = pd.bdate_range("2021-01-01", periods=T)
    cols = [f"A{i}" for i in range(n)]
    return pd.DataFrame(rets, index=idx, columns=cols)


@pytest.fixture
def mvo_analytic_case():
    """Двухактивный случай с известным аналитическим решением."""
    mu = np.array([0.001, 0.0005])
    sigma = np.array([[0.0004, 0.0001], [0.0001, 0.0002]])
    return mu, sigma
