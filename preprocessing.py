"""Предобработка ценовых данных: расчёт логдоходностей и обработка пропусков."""
from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Логарифмические доходности r_t = ln(P_t / P_{t-1})."""
    if prices.empty:
        return prices
    returns = np.log(prices / prices.shift(1))
    returns = returns.iloc[1:]
    return returns


def handle_missing(
    prices: pd.DataFrame,
    forward_fill_limit: int = 3,
    max_missing_fraction: float = 0.05,
) -> Tuple[pd.DataFrame, List[str]]:
    """Forward-fill пропусков; исключение тикеров с большим количеством NaN.

    Returns
    -------
    cleaned : pd.DataFrame — очищенные цены.
    excluded : list of str — список исключённых тикеров.
    """
    if prices.empty:
        return prices, []

    # Trading-дни: ставим NaN на нулевых ценах (вероятно бид-аск пустой день)
    cleaned = prices.replace(0.0, np.nan).copy()
    cleaned = cleaned.ffill(limit=forward_fill_limit)

    excluded: List[str] = []
    keep_cols = []
    for col in cleaned.columns:
        miss_frac = cleaned[col].isna().mean()
        if miss_frac > max_missing_fraction:
            excluded.append(col)
            logger.warning("Тикер %s исключён: пропусков %.1f%%", col, 100 * miss_frac)
        else:
            keep_cols.append(col)

    cleaned = cleaned[keep_cols].dropna(how="any")
    return cleaned, excluded


def align_to_index(
    asset_prices: pd.DataFrame,
    index_prices: pd.Series,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Выравнивает индексы наблюдений активов и индекса по общему DatetimeIndex."""
    common_idx = asset_prices.index.intersection(index_prices.index)
    return asset_prices.loc[common_idx], index_prices.loc[common_idx]
