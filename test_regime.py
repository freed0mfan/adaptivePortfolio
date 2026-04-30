"""Тесты модуля src/regime."""
from __future__ import annotations

import numpy as np
import pytest

from src.regime import (
    GaussianHMMRegimeModel,
    MSARRegimeModel,
    MSGARCHRegimeModel,
)


def _accuracy(true_states, model_states):
    """Аккуратность с учётом возможной перестановки меток."""
    perm = np.argsort([np.mean(true_states == k) for k in range(2)])  # порядок не важен
    acc1 = np.mean(true_states == model_states)
    acc2 = np.mean(true_states == (1 - model_states))
    return max(acc1, acc2)


def test_hmm_two_regime_accuracy(synthetic_two_regime_returns):
    series, states = synthetic_two_regime_returns
    model = GaussianHMMRegimeModel(k_regimes=2)
    model.fit(series)
    proba = model.get_filtered_proba(series)
    pred = proba.idxmax(axis=1).str.extract(r"(\d+)").astype(int).iloc[:, 0].values
    acc = _accuracy(np.asarray(states), pred)
    assert acc >= 0.75


def test_ms_ar_two_regime_accuracy(synthetic_two_regime_returns):
    series, states = synthetic_two_regime_returns
    model = MSARRegimeModel(k_regimes=2, order=0)
    model.fit(series)
    proba = model.get_filtered_proba()
    pred = proba.idxmax(axis=1).str.extract(r"(\d+)").astype(int).iloc[:, 0].values
    acc = _accuracy(np.asarray(states[-len(pred):]), pred)
    assert acc >= 0.70


def test_predict_next_returns_simplex(synthetic_two_regime_returns):
    series, _ = synthetic_two_regime_returns
    model = GaussianHMMRegimeModel(k_regimes=2)
    model.fit(series)
    pi = model.predict_next(0.0)
    assert pi.shape == (2,)
    assert pytest.approx(pi.sum(), abs=1e-6) == 1.0
    assert (pi >= 0).all()


def test_ms_garch_two_regime_separates(synthetic_two_regime_returns):
    """MS-GARCH(1,1) Haas et al. разделяет режимы выше случайного."""
    series, states = synthetic_two_regime_returns
    model = MSGARCHRegimeModel(k_regimes=2, max_iter=40, n_starts=2, random_state=0)
    model.fit(series)
    proba = model.get_filtered_proba()
    pred = proba.idxmax(axis=1).str.extract(r"(\d+)").astype(int).iloc[:, 0].values
    acc = _accuracy(np.asarray(states[-len(pred):]), pred)
    assert acc >= 0.65, f"Режимы плохо разделены: acc={acc:.2f}"


def test_ms_garch_k1_degenerates_to_garch(synthetic_two_regime_returns):
    """При K=1 модель вырождается в обычный GARCH(1,1)."""
    series, _ = synthetic_two_regime_returns
    model = MSGARCHRegimeModel(k_regimes=1, max_iter=20, n_starts=1, random_state=0)
    model.fit(series)
    p = model.get_regime_params()
    assert p.k == 1
    assert p.transition_matrix.shape == (1, 1)
    assert pytest.approx(p.transition_matrix[0, 0], abs=1e-6) == 1.0
    proba = model.get_filtered_proba()
    assert (proba.values >= 0.999).all()  # все веса на единственный режим
