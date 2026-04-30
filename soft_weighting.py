"""Мягкое смешивание режимных портфелей по апостериорным вероятностям."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def soft_blend(
    regime_weights: Dict[int, pd.Series],
    pi_t: np.ndarray,
) -> pd.Series:
    """w_t = Σ_k π_{t|t}^(k) * w_k_star.

    Parameters
    ----------
    regime_weights : dict[int, pd.Series]
        Режимные веса (Series по тикерам).
    pi_t : np.ndarray
        Апостериорные вероятности режимов длиной K.
    """
    K = len(pi_t)
    if not regime_weights:
        raise ValueError("regime_weights пуст.")
    keys = sorted(regime_weights.keys())
    if len(keys) != K:
        raise ValueError(f"Число режимов ({K}) не соответствует regime_weights ({len(keys)}).")
    total = pi_t.sum()
    if total <= 0 or not np.isfinite(total):
        pi_t = np.ones(K) / K
    else:
        pi_t = np.asarray(pi_t) / total

    tickers = regime_weights[keys[0]].index
    blended = pd.Series(0.0, index=tickers)
    for k, weight in zip(keys, pi_t):
        ws = regime_weights[k].reindex(tickers).fillna(0.0)
        blended = blended + float(weight) * ws
    s = blended.sum()
    if s <= 0:
        return pd.Series(np.ones(len(tickers)) / len(tickers), index=tickers)
    return blended / s
