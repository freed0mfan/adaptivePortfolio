"""Автоматический выбор числа режимов K по AIC/BIC."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Type

import numpy as np
import pandas as pd

from .base import RegimeModel
from .ms_ar import MSARRegimeModel

logger = logging.getLogger(__name__)


@dataclass
class KSelectionResult:
    aic_table: pd.Series
    bic_table: pd.Series
    recommended_k: int
    params_per_k: pd.DataFrame


class RegimeSelector:
    """Перебор K и выбор по информационному критерию."""

    @staticmethod
    def select(
        returns: pd.Series,
        k_max: int = 4,
        k_min: int = 1,
        model_class: Type[RegimeModel] = MSARRegimeModel,
        criterion: str = "aic",
        random_state: int = 42,
    ) -> KSelectionResult:
        if k_max < k_min:
            raise ValueError("k_max должно быть >= k_min")
        if k_min < 1:
            raise ValueError("k_min должно быть >= 1")
        if criterion not in ("aic", "bic"):
            raise ValueError("criterion должен быть 'aic' или 'bic'")

        records = {}
        aic_vals: dict[int, float] = {}
        bic_vals: dict[int, float] = {}
        for K in range(k_min, k_max + 1):
            try:
                model = model_class(k_regimes=K, random_state=random_state)
                model.fit(returns)
                p = model.get_regime_params()
                aic_vals[K] = p.aic
                bic_vals[K] = p.bic
                records[K] = {
                    "log_likelihood": p.log_likelihood,
                    "aic": p.aic,
                    "bic": p.bic,
                }
            except Exception as exc:  # noqa: BLE001
                logger.warning("Ошибка при K=%d: %s", K, exc)
                aic_vals[K] = np.inf
                bic_vals[K] = np.inf
                records[K] = {"log_likelihood": np.nan, "aic": np.inf, "bic": np.inf}

        aic_series = pd.Series(aic_vals, name="aic").sort_index()
        bic_series = pd.Series(bic_vals, name="bic").sort_index()
        target = bic_series if criterion == "bic" else aic_series
        recommended = int(target.idxmin())
        params_df = pd.DataFrame.from_dict(records, orient="index").sort_index()
        return KSelectionResult(
            aic_table=aic_series,
            bic_table=bic_series,
            recommended_k=recommended,
            params_per_k=params_df,
        )
