"""Помощники для сохранения/загрузки артефактов в parquet, Excel."""
from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Dict, Union

import pandas as pd

logger = logging.getLogger(__name__)


def save_parquet(df: pd.DataFrame, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def load_parquet(path: Union[str, Path]) -> pd.DataFrame:
    return pd.read_parquet(Path(path))


def dataframes_to_excel(sheets: Dict[str, pd.DataFrame]) -> bytes:
    """Сериализует словарь {имя_листа: DataFrame} в xlsx и возвращает bytes."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, df in sheets.items():
            safe_name = str(name)[:31]
            try:
                df.to_excel(writer, sheet_name=safe_name)
            except Exception as exc:  # pragma: no cover
                logger.warning("Не удалось записать лист %s: %s", safe_name, exc)
    return buf.getvalue()
