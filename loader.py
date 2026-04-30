"""Загрузчик данных MOEX ISS с локальным parquet-кэшем.

Если установлены `apimoex`/`requests` и доступна сеть — данные тянутся из MOEX ISS.
Иначе срабатывает синтетический fallback: генерируется правдоподобный ряд с
корреляцией и переключающимися режимами волатильности. Это нужно для оффлайн-
демонстрации и тестов.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from ..utils.config import DEFAULT_TICKERS, TRADING_DAYS_PER_YEAR
from ..utils.exceptions import (
    InsufficientDataError,
    InvalidTickerError,
    MoexConnectionError,
)
from .preprocessing import align_to_index, compute_log_returns, handle_missing

logger = logging.getLogger(__name__)

MOEX_BASE_URL = "https://iss.moex.com/iss"


@dataclass
class DataBundle:
    """Контейнер с очищенными ценами/доходностями и метаданными."""

    returns: pd.DataFrame
    prices: pd.DataFrame
    index_returns: pd.Series
    index_prices: pd.Series
    metadata: Dict = field(default_factory=dict)

    @property
    def tickers(self) -> List[str]:
        return list(self.returns.columns)

    @property
    def n_assets(self) -> int:
        return self.returns.shape[1]

    @property
    def n_obs(self) -> int:
        return self.returns.shape[0]


# ---------------------------------------------------------------------------
# Synthetic fallback (used when MOEX ISS unreachable / for tests)
# ---------------------------------------------------------------------------

def _generate_synthetic_prices(
    tickers: List[str],
    start: date,
    end: date,
    seed: int = 42,
) -> pd.DataFrame:
    """Создаёт правдоподобный ряд цен с двумя режимами волатильности.

    Используется только в качестве запасного источника, если API недоступен.
    """
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start=start, end=end)
    if len(days) < 2:
        days = pd.bdate_range(start=start, periods=260)
    T = len(days)
    n = len(tickers)

    # двухрежимная марковская цепь
    P = np.array([[0.97, 0.03], [0.10, 0.90]])
    states = np.zeros(T, dtype=int)
    states[0] = 0
    for t in range(1, T):
        states[t] = rng.choice(2, p=P[states[t - 1]])

    sigmas = np.array([0.012, 0.035])  # дневная вол-сть в режиме
    mus = np.array([0.0006, -0.0008])

    # факторная структура: рыночный фактор + индивидуальные шоки
    market = rng.normal(0.0, 1.0, size=T) * sigmas[states] + mus[states]
    betas = rng.uniform(0.6, 1.3, size=n)
    idio = rng.normal(0.0, 0.01, size=(T, n))
    log_rets = market[:, None] * betas[None, :] + idio
    log_rets[0] = 0.0

    prices = 100.0 * np.exp(np.cumsum(log_rets, axis=0))
    df = pd.DataFrame(prices, index=days, columns=tickers)
    return df


def _generate_synthetic_index(
    asset_prices: pd.DataFrame,
    seed: int = 7,
) -> pd.Series:
    """IMOEX-аналог: равновзвешенный портфель + небольшой шум."""
    rng = np.random.default_rng(seed)
    rets = np.log(asset_prices / asset_prices.shift(1)).iloc[1:]
    idx_rets = rets.mean(axis=1) + rng.normal(0, 0.0008, size=len(rets))
    idx_prices = 3000.0 * np.exp(np.cumsum(idx_rets))
    idx_prices = pd.concat(
        [pd.Series([3000.0], index=[asset_prices.index[0]]), idx_prices]
    )
    idx_prices.name = "IMOEX"
    return idx_prices


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

class MoexDataLoader:
    """Загрузчик данных MOEX ISS.

    Parameters
    ----------
    cache_dir : str
        Каталог для parquet-кэша.
    auto_update_days : int
        Сколько дней «протух» кэш считается актуальным.
    max_missing_fraction : float
        Максимальная доля пропусков на тикер.
    forward_fill_limit : int
        Максимум подряд идущих пропусков для forward-fill.
    allow_synthetic_fallback : bool
        Разрешить ли генерировать синтетику при отсутствии сети.
    """

    def __init__(
        self,
        cache_dir: str = "data/cache",
        auto_update_days: int = 2,
        max_missing_fraction: float = 0.05,
        forward_fill_limit: int = 3,
        allow_synthetic_fallback: bool = True,
        request_timeout: float = 15.0,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.auto_update_days = auto_update_days
        self.max_missing_fraction = max_missing_fraction
        self.forward_fill_limit = forward_fill_limit
        self.allow_synthetic_fallback = allow_synthetic_fallback
        self.request_timeout = request_timeout

    # ------------------------------------------------------------------ utils
    def _cache_path(self, key: str) -> Path:
        safe = key.replace(":", "_").replace("/", "_")
        return self.cache_dir / f"{safe}.parquet"

    def _is_cache_fresh(self, path: Path) -> bool:
        if not path.exists():
            return False
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        return (datetime.now() - mtime) <= timedelta(days=self.auto_update_days)

    # --------------------------------------------------------------- requests
    def _moex_get_candles(
        self, ticker: str, start: date, end: date, interval: str
    ) -> pd.DataFrame:
        """Загружает свечи через ISS REST с ручной пагинацией.

        MOEX ISS отдаёт максимум 500 свечей за запрос. Чтобы корректно
        получить произвольный диапазон, нужно последовательно сдвигать
        параметр ``start`` (offset по индексу свечей), пока не закончатся
        новые наблюдения.
        """
        try:
            import requests
        except ImportError as exc:  # pragma: no cover
            raise MoexConnectionError(
                "Библиотека requests не установлена."
            ) from exc

        moex_interval = {"day": 24, "week": 7}[interval]
        url = (
            f"{MOEX_BASE_URL}/engines/stock/markets/shares/securities/{ticker}/"
            f"candles.json"
        )
        last_err: Optional[Exception] = None
        for attempt in range(3):
            try:
                rows_all: List[list] = []
                cols: Optional[List[str]] = None
                offset = 0
                with requests.Session() as session:
                    session.headers.update({"User-Agent": "AdaptivePortfolio/1.0"})
                    while True:
                        params = {
                            "from": start.isoformat(),
                            "till": end.isoformat(),
                            "interval": moex_interval,
                            "start": offset,
                        }
                        resp = session.get(url, params=params, timeout=self.request_timeout)
                        resp.raise_for_status()
                        payload = resp.json()
                        cols = payload["candles"]["columns"]
                        rows = payload["candles"]["data"]
                        if not rows:
                            break
                        rows_all.extend(rows)
                        if len(rows) < 500:
                            break
                        offset += len(rows)
                if not rows_all:
                    raise InvalidTickerError(ticker)
                df = pd.DataFrame(rows_all, columns=cols)
                df["begin"] = pd.to_datetime(df["begin"])
                df = df.drop_duplicates(subset="begin").set_index("begin").sort_index()
                return df[["close"]].rename(columns={"close": ticker})
            except InvalidTickerError:
                raise
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                logger.warning(
                    "MOEX ISS error attempt %d for %s: %s", attempt + 1, ticker, exc
                )
                time.sleep(2.0 * (attempt + 1))
        raise MoexConnectionError(str(last_err))

    def _moex_get_index(
        self, start: date, end: date, interval: str
    ) -> pd.Series:
        try:
            import requests
        except ImportError as exc:  # pragma: no cover
            raise MoexConnectionError("requests не установлен") from exc

        moex_interval = {"day": 24, "week": 7}[interval]
        url = (
            f"{MOEX_BASE_URL}/engines/stock/markets/index/securities/IMOEX/"
            f"candles.json"
        )
        for attempt in range(3):
            try:
                rows_all: List[list] = []
                cols: Optional[List[str]] = None
                offset = 0
                with requests.Session() as session:
                    session.headers.update({"User-Agent": "AdaptivePortfolio/1.0"})
                    while True:
                        params = {
                            "from": start.isoformat(),
                            "till": end.isoformat(),
                            "interval": moex_interval,
                            "start": offset,
                        }
                        resp = session.get(url, params=params, timeout=self.request_timeout)
                        resp.raise_for_status()
                        payload = resp.json()
                        cols = payload["candles"]["columns"]
                        rows = payload["candles"]["data"]
                        if not rows:
                            break
                        rows_all.extend(rows)
                        if len(rows) < 500:
                            break
                        offset += len(rows)
                if not rows_all:
                    raise MoexConnectionError("IMOEX candles empty")
                df = pd.DataFrame(rows_all, columns=cols)
                df["begin"] = pd.to_datetime(df["begin"])
                df = df.drop_duplicates(subset="begin").set_index("begin").sort_index()
                return df["close"].rename("IMOEX")
            except Exception as exc:  # noqa: BLE001
                logger.warning("IMOEX attempt %d failed: %s", attempt + 1, exc)
                time.sleep(2.0 * (attempt + 1))
        raise MoexConnectionError("Не удалось получить IMOEX")

    # ------------------------------------------------------------------ public
    def load(
        self,
        tickers: Union[str, List[str]],
        start: date,
        end: date,
        interval: str = "day",
        use_cache: bool = True,
        force_synthetic: bool = False,
    ) -> DataBundle:
        if isinstance(tickers, str):
            tickers = [tickers]
        tickers = [t.upper() for t in tickers]
        if interval not in ("day", "week"):
            raise ValueError("interval должен быть 'day' или 'week'")

        cache_key = f"prices_{','.join(tickers)}_{start}_{end}_{interval}"
        cache_path = self._cache_path(cache_key)
        prices_df: Optional[pd.DataFrame] = None
        index_prices: Optional[pd.Series] = None
        source = "moex"

        if use_cache and self._is_cache_fresh(cache_path):
            try:
                prices_df = pd.read_parquet(cache_path)
                idx_path = self._cache_path(f"imoex_{start}_{end}_{interval}")
                if idx_path.exists():
                    index_prices = pd.read_parquet(idx_path)["IMOEX"]
                source = "cache"
                logger.info("Загружено из кэша: %s", cache_path.name)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Кэш повреждён, перезагружаем. %s", exc)
                prices_df = None
                index_prices = None

        if prices_df is None and not force_synthetic:
            try:
                frames = []
                for t in tickers:
                    frames.append(self._moex_get_candles(t, start, end, interval))
                prices_df = pd.concat(frames, axis=1)
                index_prices = self._moex_get_index(start, end, interval)
                # write cache
                prices_df.to_parquet(cache_path)
                self._cache_path(f"imoex_{start}_{end}_{interval}").write_bytes(
                    pd.DataFrame({"IMOEX": index_prices}).to_parquet()
                )
            except (MoexConnectionError, InvalidTickerError) as exc:
                if isinstance(exc, InvalidTickerError):
                    raise
                if not self.allow_synthetic_fallback:
                    raise
                logger.warning(
                    "MOEX ISS недоступен (%s). Использую синтетический набор данных.", exc
                )
                prices_df = _generate_synthetic_prices(tickers, start, end)
                index_prices = _generate_synthetic_index(prices_df)
                source = "synthetic"

        if prices_df is None or force_synthetic:
            prices_df = _generate_synthetic_prices(tickers, start, end)
            index_prices = _generate_synthetic_index(prices_df)
            source = "synthetic"

        cleaned, excluded = handle_missing(
            prices_df,
            forward_fill_limit=self.forward_fill_limit,
            max_missing_fraction=self.max_missing_fraction,
        )
        if cleaned.empty or cleaned.shape[1] == 0:
            raise InsufficientDataError(message="Нет валидных тикеров после очистки.")

        if index_prices is None:
            index_prices = _generate_synthetic_index(cleaned)

        cleaned, index_prices = align_to_index(cleaned, index_prices)
        if len(cleaned) < 60:
            raise InsufficientDataError(n_available=len(cleaned), required=60)

        returns = compute_log_returns(cleaned)
        index_returns = compute_log_returns(index_prices.to_frame()).iloc[:, 0]
        index_returns.name = "IMOEX"

        if len(returns) < 60:
            raise InsufficientDataError(n_available=len(returns), required=60)

        metadata = {
            "source": source,
            "loaded_at": datetime.now().isoformat(),
            "n_missing": int(prices_df.isna().sum().sum()),
            "tickers": list(cleaned.columns),
            "excluded_tickers": excluded,
            "start": str(start),
            "end": str(end),
            "interval": interval,
        }
        return DataBundle(
            returns=returns,
            prices=cleaned,
            index_returns=index_returns,
            index_prices=index_prices,
            metadata=metadata,
        )

    # ------------------------------------------------------------ index utils
    def get_imoex_components(self, as_of_date: Optional[date] = None) -> List[str]:
        """Возвращает список тикеров из состава IMOEX (или дефолт)."""
        try:
            import requests
            url = f"{MOEX_BASE_URL}/statistics/engines/stock/markets/index/analytics/IMOEX.json"
            resp = requests.get(url, params={"limit": 100}, timeout=self.request_timeout)
            resp.raise_for_status()
            data = resp.json()
            cols = data["analytics"]["columns"]
            rows = data["analytics"]["data"]
            df = pd.DataFrame(rows, columns=cols)
            tickers = df["ticker"].dropna().unique().tolist()
            return [t for t in tickers if isinstance(t, str)]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Не удалось получить состав IMOEX: %s", exc)
            return list(DEFAULT_TICKERS)

    def get_index_weights(self, as_of_date: Optional[date] = None) -> pd.Series:
        """Возвращает веса IMOEX по тикерам (на доступную дату)."""
        try:
            import requests
            url = f"{MOEX_BASE_URL}/statistics/engines/stock/markets/index/analytics/IMOEX.json"
            resp = requests.get(url, params={"limit": 100}, timeout=self.request_timeout)
            resp.raise_for_status()
            data = resp.json()
            cols = data["analytics"]["columns"]
            rows = data["analytics"]["data"]
            df = pd.DataFrame(rows, columns=cols)
            df = df.dropna(subset=["ticker", "weight"])
            s = pd.Series(df["weight"].astype(float).values, index=df["ticker"]).astype(float)
            s = s / s.sum()
            return s
        except Exception as exc:  # noqa: BLE001
            logger.warning("Не удалось получить веса IMOEX: %s", exc)
            tickers = list(DEFAULT_TICKERS)
            return pd.Series(np.ones(len(tickers)) / len(tickers), index=tickers)
