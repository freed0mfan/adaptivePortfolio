"""Базовые классы для модуля оптимизации."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class PortfolioWeights:
    """Контейнер с весами портфеля и метаинформацией."""

    weights: pd.Series
    regime_proba: np.ndarray
    timestamp: datetime
    lambda_param: float
    rebalanced: bool
    regime_weights: Dict[int, pd.Series] = field(default_factory=dict)
    raw_weights: Optional[pd.Series] = None
