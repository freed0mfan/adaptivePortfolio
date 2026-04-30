"""Интерпретация режимов: метки, статистика, эпизоды."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..utils.config import TRADING_DAYS_PER_YEAR
from .base import RegimeParams


class RegimeInterpreter:
    """Утилиты для экономической интерпретации режимов."""

    @staticmethod
    def assign_labels(
        params: RegimeParams, custom: Dict[int, str] | None = None
    ) -> Dict[int, str]:
        """Назначает экономические метки режимам. Если custom задан, используется он."""
        K = params.k
        order = np.argsort(params.sigma2)
        labels: Dict[int, str] = {}
        if K == 1:
            labels[int(order[0])] = "Единый режим (MVO)"
        elif K == 2:
            labels[int(order[0])] = "Низкая волатильность (бычий рынок)"
            labels[int(order[1])] = "Высокая волатильность (кризис)"
        elif K == 3:
            labels[int(order[0])] = "Низкая волатильность"
            labels[int(order[1])] = "Переходный режим"
            labels[int(order[2])] = "Высокая волатильность (кризис)"
        else:
            for rank, idx in enumerate(order):
                labels[int(idx)] = f"Режим {rank+1} (σ²={params.sigma2[idx]:.4g})"
        if custom:
            for k, v in custom.items():
                if v is None:
                    continue
                v = str(v).strip()
                if v:
                    labels[int(k)] = v
        return labels

    @staticmethod
    def summarize(
        params: RegimeParams,
        filtered_proba: pd.DataFrame,
        trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
    ) -> pd.DataFrame:
        labels = RegimeInterpreter.assign_labels(params)
        K = params.k
        rows = []
        diag = np.diag(params.transition_matrix)
        time_fraction = filtered_proba.mean(axis=0).values
        for k in range(K):
            sigma_ann = float(np.sqrt(params.sigma2[k]) * np.sqrt(trading_days_per_year))
            mu_ann = float(params.mu[k] * trading_days_per_year)
            avg_dur = float(1.0 / max(1.0 - diag[k], 1e-6))
            rows.append(
                {
                    "regime_id": k,
                    "regime_label": labels[k],
                    "mu_annual": mu_ann,
                    "sigma_annual": sigma_ann,
                    "avg_duration_days": avg_dur,
                    "time_fraction": float(time_fraction[k]),
                    "stationary_proba": float(params.stationary_dist[k]),
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def annotate_episodes(
        filtered_proba: pd.DataFrame,
        threshold: float = 0.0,
        labels: Dict[int, str] | None = None,
    ) -> pd.DataFrame:
        """Возвращает таблицу периодов доминирования режима.

        По умолчанию (threshold=0) каждый день относится к режиму с максимальной
        апостериорной вероятностью — таймлайн покрывается целиком, без «белых полос».
        """
        if filtered_proba.empty:
            return pd.DataFrame(columns=["start_date", "end_date", "regime_id", "regime_label"])
        K = filtered_proba.shape[1]
        max_proba = filtered_proba.max(axis=1)
        dominant = filtered_proba.idxmax(axis=1).str.extract(r"regime_(\d+)").astype(int).iloc[:, 0]
        # Явный «неопределённый» режим (-1) используется только при threshold > 0
        if threshold > 0:
            dominant = dominant.where(max_proba >= threshold, -1)
        dates = filtered_proba.index

        episodes: List[Tuple] = []
        if len(dominant) == 0:
            return pd.DataFrame(columns=["start_date", "end_date", "regime_id", "regime_label"])
        # Накапливаем эпизоды: конец эпизода совпадает с началом следующего,
        # чтобы vrect-ы покрывали весь таймлайн без «белых полос» на выходных.
        cur_state = int(dominant.iloc[0])
        cur_start = dates[0]
        for t in range(1, len(dominant)):
            s = int(dominant.iloc[t])
            if s != cur_state:
                episodes.append((cur_start, dates[t], cur_state))
                cur_state = s
                cur_start = dates[t]
        # Последний эпизод растягиваем вправо на один бизнес-день, чтобы
        # правый край не обрывался.
        last_end = dates[-1] + pd.tseries.offsets.BDay(1)
        episodes.append((cur_start, last_end, cur_state))

        rows = []
        for start, end, st in episodes:
            if st == -1:
                continue
            label = labels[st] if labels is not None else f"regime_{st}"
            rows.append(
                {
                    "start_date": start,
                    "end_date": end,
                    "regime_id": st,
                    "regime_label": label,
                    "duration_days": int((end - start).days),
                }
            )
        return pd.DataFrame(rows)
