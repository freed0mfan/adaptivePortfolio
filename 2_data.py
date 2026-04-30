"""Страница загрузки и просмотра данных."""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data import MoexDataLoader
from src.utils import DEFAULT_TICKERS
from src.utils.exceptions import (
    InsufficientDataError,
    InvalidTickerError,
    MoexConnectionError,
)

MOEX_URL = "https://iss.moex.com"
PLOTLY_TEMPLATE = "plotly_dark"

st.title("📥 Загрузка данных MOEX ISS")
st.caption(
    f"Котировки и логдоходности акций IMOEX. Источник: "
    f"[МосБиржа ISS API]({MOEX_URL}). Кэш — `data/cache/`."
)

with st.sidebar:
    st.header("Параметры загрузки")
    tickers_text = st.text_area(
        "Тикеры (через запятую)",
        value=", ".join(DEFAULT_TICKERS),
        help="Любые тикеры режима TQBR. Пример: SBER, LKOH, GAZP",
    )
    tickers = [t.strip().upper() for t in tickers_text.split(",") if t.strip()]
    col1, col2 = st.columns(2)
    start_date = col1.date_input("Начало", value=date(2019, 1, 1))
    end_date = col2.date_input("Конец", value=date.today())
    interval = st.selectbox("Интервал", ["day", "week"], index=0)
    use_cache = st.checkbox("Использовать кэш", value=True)
    force_synth = st.checkbox(
        "Демо-режим (синтетические данные)",
        value=False,
        help="Полезно при отсутствии интернета. Использует генерируемые правдоподобные данные с двумя режимами.",
    )

if st.button("Загрузить данные", type="primary"):
    if not tickers:
        st.error("Укажите хотя бы один тикер.")
    elif start_date >= end_date:
        st.error("Дата начала должна быть раньше даты окончания.")
    else:
        with st.spinner("Загрузка..."):
            loader = MoexDataLoader()
            try:
                bundle = loader.load(
                    tickers, start_date, end_date, interval=interval,
                    use_cache=use_cache, force_synthetic=force_synth,
                )
                st.session_state.data_bundle = bundle
                st.success(
                    f"Загружено: {bundle.n_assets} тикеров, {bundle.n_obs} наблюдений. "
                    f"Источник: {bundle.metadata['source']}."
                )
            except InvalidTickerError as exc:
                st.error(f"Неверный тикер: {exc.ticker}")
            except (MoexConnectionError, InsufficientDataError) as exc:
                st.error(str(exc))

bundle = st.session_state.get("data_bundle")
if bundle is None:
    st.info("Загрузите данные, чтобы увидеть графики.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Тикеров", bundle.n_assets)
c2.metric("Наблюдений", bundle.n_obs)
c3.metric("Пропусков (исходно)", bundle.metadata.get("n_missing", 0))
source = bundle.metadata.get("source", "—")
if isinstance(source, str) and "moex" in source.lower():
    c4.markdown(f"**Источник**\n\n[{source}]({MOEX_URL})")
else:
    c4.metric("Источник", source)

if bundle.metadata.get("excluded_tickers"):
    st.warning(f"Исключены: {bundle.metadata['excluded_tickers']}")

st.subheader("Накопленная доходность активов")
import numpy as np
cum_eq = np.exp(bundle.returns.fillna(0).cumsum())
fig_eq = px.line(cum_eq, labels={"value": "Накопленная доходность", "index": "Дата"})
fig_eq.update_layout(legend_title="Тикер", height=420, template=PLOTLY_TEMPLATE)
st.plotly_chart(fig_eq, use_container_width=True)

st.subheader("Корреляционная матрица")
corr = bundle.returns.corr()
fig_corr = px.imshow(
    corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, text_auto=".2f", aspect="auto",
)
fig_corr.update_layout(height=480, template=PLOTLY_TEMPLATE)
st.plotly_chart(fig_corr, use_container_width=True)

st.subheader("Доходности (первые 20 строк)")
st.dataframe(bundle.returns.head(20), use_container_width=True)

with st.expander("Метаданные"):
    st.json(bundle.metadata)
