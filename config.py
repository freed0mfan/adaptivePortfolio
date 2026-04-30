"""Default configuration constants for AdaptivePortfolio."""
from dataclasses import dataclass, field
from typing import List

# IMOEX-наиболее ликвидные акции по умолчанию
DEFAULT_TICKERS: List[str] = [
    "SBER", "LKOH", "GAZP", "GMKN", "ROSN", "NVTK", "PLZL", "MGNT", "MTSS", "TATN",
]

DEFAULT_START_DATE = "2019-01-01"
TRADING_DAYS_PER_YEAR = 252
DEFAULT_RF_RATE_ANNUAL = 0.16  # ключевая ставка ЦБ РФ ~16% годовых
DEFAULT_RF_RATE_DAILY = DEFAULT_RF_RATE_ANNUAL / TRADING_DAYS_PER_YEAR

# Цветовая палитра режимов и стратегий
REGIME_COLORS = ["#2E7D32", "#C62828", "#F9A825", "#1565C0"]
STRATEGY_COLORS = {
    "Adaptive": "#1f77b4",
    "IMOEX": "#ff7f0e",
    "EW": "#2ca02c",
    "MVO_static": "#d62728",
    "RiskParity": "#9467bd",
    "MinVar": "#8c564b",
    "MaxSharpe": "#e377c2",
}

HISTORICAL_EVENTS = {
    "2014-12": "Девальвация рубля",
    "2020-03": "COVID-19 шок",
    "2022-02": "Геополитический кризис",
    "2022-09": "Частичная мобилизация",
}


@dataclass
class Config:
    """Глобальная конфигурация системы."""
    cache_dir: str = "data/cache"
    auto_update_days: int = 2
    max_missing_fraction: float = 0.05
    forward_fill_limit: int = 3
    random_state: int = 42
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR
    rf_rate_annual: float = DEFAULT_RF_RATE_ANNUAL
    default_tickers: List[str] = field(default_factory=lambda: list(DEFAULT_TICKERS))


CONFIG = Config()
