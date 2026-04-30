"""Главная страница дашборда."""
from __future__ import annotations

import streamlit as st

st.title("📈 AdaptivePortfolio — адаптивная портфельная оптимизация")
st.caption("Курсовая работа НГУ, 3 курс, Бизнес-информатика, 2026. Автор: **Маркелов Егор**.")

st.markdown(
    """
    **Назначение системы.** Программный артефакт курсовой работы реализует полный пайплайн
    адаптивной портфельной оптимизации на основе моделей переключения режимов волатильности
    и применяет его к российскому рынку акций (индекс МосБиржи).

    ### Этапы пайплайна
    1. **📥 Данные.** Загрузка котировок из MOEX ISS REST API, расчёт логдоходностей.
    2. **🔍 Режимы.** Идентификация скрытых режимов через MS-GARCH(1,1) (Haas et al., 2004), MS-AR или GaussianHMM.
    3. **⚖️ Оптимизация.** Решение задачи MVO/CVaR через CVXPY, мягкое смешивание режимных портфелей.
    4. **📊 Бэктест.** Walk-forward оценка адаптивной стратегии vs шесть бенчмарков.
    5. **📄 Отчёт.** Экспорт результатов в Excel/CSV/PDF.

    ### Используемые методы
    - Markov-Switching GARCH(1,1) в спецификации Haas–Mittnik–Paolella (2004)
    - Markov-Switching Autoregression (Hamilton, 1989) / Gaussian HMM с EM-алгоритмом
    - Mean-Variance Optimization (Markowitz, 1952) и CVaR (Rockafellar–Uryasev, 2000)
    - Ledoit–Wolf shrinkage ковариационной матрицы
    - Walk-forward валидация без look-ahead bias
    - Блочный бутстрап Шарпа, t-тест α, тест Дибольда–Мариано

    ### Бенчмарки
    `IMOEX`, `EW` (равные веса), `Static MVO`, `Risk Parity`, `Min Variance`, `Max Sharpe`.
    """
)

st.info(
    "Перейдите на страницу **📥 Данные** в боковом меню, чтобы загрузить котировки и начать работу."
)

with st.expander("⚙️ Состояние сессии"):
    keys = ["data_bundle", "regime_model", "optimizer", "backtest_result"]
    state = {k: ("✅ загружено" if k in st.session_state and st.session_state[k] is not None else "—") for k in keys}
    st.json(state)
