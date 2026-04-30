"""Страница walk-forward бэктеста."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.backtest import (
    ALL_BENCHMARKS,
    Backtester,
    build_strategy,
)
from src.regime import (
    GaussianHMMRegimeModel,
    MSARRegimeModel,
    MSGARCHRegimeModel,
)
from src.utils import STRATEGY_COLORS

PLOTLY_TEMPLATE = "plotly_dark"

st.title("📊 Walk-forward бэктест")
st.caption(
    "Сравнение адаптивной стратегии с шестью бенчмарками. Индекс IMOEX загружается из MOEX ISS API "
    "(взвешен по капитализации), EW — буквально 1/N по выбранным тикерам."
)

bundle = st.session_state.get("data_bundle")
model = st.session_state.get("regime_model")
opt = st.session_state.get("optimizer")
if bundle is None or model is None or opt is None:
    st.warning("Перед бэктестом загрузите данные, обучите модель и запустите оптимизацию.")
    st.stop()

def _popover(col, key: str, body: str) -> None:
    """Короткая всплывающая подсказка в ряду разметки."""
    with col.popover("ℹ️", help=key):
        st.markdown(body)


with st.sidebar:
    st.header("Параметры бэктеста")
    col_b1, col_b2 = st.columns([3, 1])
    benchmark_names = col_b1.multiselect(
        "Бенчмарки", ALL_BENCHMARKS,
        default=["IMOEX", "EW", "MVO_static"],
    )
    _popover(
        col_b2, "Бенчмарки",
        "**Бенчмарки для сравнения.**\n\n"
        "• **IMOEX** — индекс МосБиржи, взвешенный по капитализации (референс рынка).\n"
        "• **EW** — равновзвешенный портфель (1/N).\n"
        "• **MVO_static** — одноразовая Markowitz-оптимизация на всём train-периоде.\n"
        "• **MinVar** — портфель минимальной дисперсии без ожидаемых доходностей.\n"
        "• **RiskParity** — равный вклад в риск (Equal Risk Contribution).\n"
        "• **MaxSharpe** — точка касания эффективной границы.\n\n"
        "Каждый бенчмарк ребалансируется в walk-forward, кроме IMOEX (внешняя серия)."
    )

    col_t1, col_t2 = st.columns([3, 1])
    train_window = col_t1.number_input("Окно обучения (дни)", 126, 504, 252, 21)
    _popover(
        col_t2, "Окно обучения",
        "**Окно обучения (train).** На этом окне заново оцениваются параметры "
        "режимной модели и моменты доходностей перед каждым тестовым шагом.\n\n"
        "Рекомендованный диапазон — 252–504 (1–2 года): короткие окна плохо "
        "идентифицируют режимы, длинные — сглаживают переключения."
    )

    col_te1, col_te2 = st.columns([3, 1])
    test_window = col_te1.number_input("Окно теста (дни)", 5, 63, 21, 1)
    _popover(
        col_te2, "Окно теста",
        "**Окно теста (test).** Сколько дней подряд фиксируется выбранный портфель "
        "до следующей переоценки. 21 день ≈1 торговый месяц — разумный компромисс "
        "между реактивностью и издержками."
    )

    col_s1, col_s2 = st.columns([3, 1])
    step = col_s1.number_input("Шаг (дни)", 5, 63, 21, 1)
    _popover(
        col_s2, "Шаг",
        "**Шаг продвижения окна.** На сколько дней сдвигается train+test пара. "
        "Обычно равен окну теста, чтобы сегменты не пересекались (out-of-sample)."
    )

    col_tc1, col_tc2 = st.columns([3, 1])
    tc_bps = col_tc1.number_input("Транзакционные издержки (б.п.)", 0.0, 50.0, 10.0, 1.0)
    _popover(
        col_tc2, "Издержки",
        "**Транзакционные издержки** в базисных пунктах (1 б.п. = 0.01%). "
        "На каждой ребалансировке из доходности вычитается "
        "\\(\\text{tc\\_bps}/10000 \\cdot \\|w_t - w_{t-1}\\|_1\\). "
        "Рынок МОБ — порядка 5–15 б.п. для ликвидных акций."
    )
    if isinstance(model, MSGARCHRegimeModel):
        default_model = "MS-GARCH"
    elif isinstance(model, GaussianHMMRegimeModel):
        default_model = "GaussianHMM"
    else:
        default_model = "MS-AR"
    regime_model_name = st.selectbox(
        "Модель режимов в walk-forward",
        ["MS-GARCH", "MS-AR", "GaussianHMM"],
        index=["MS-GARCH", "MS-AR", "GaussianHMM"].index(default_model),
    )
    logscale = st.checkbox("Лог. шкала equity", value=False)

if st.button("Запустить бэктест", type="primary"):
    progress = st.progress(0.0)
    strategies = [build_strategy(n) for n in benchmark_names]
    backtester = Backtester(
        train_window=int(train_window),
        test_window=int(test_window),
        step=int(step),
        tc_bps=float(tc_bps),
    )
    if regime_model_name == "MS-GARCH":
        model_class = MSGARCHRegimeModel
    elif regime_model_name == "GaussianHMM":
        model_class = GaussianHMMRegimeModel
    else:
        model_class = MSARRegimeModel
    k_use = model.k_regimes
    if model_class is MSARRegimeModel and k_use < 2:
        k_use = 2
    model_kwargs = {"k_regimes": k_use}
    if model_class is MSARRegimeModel:
        model_kwargs["order"] = getattr(model, "order", 0)
    try:
        result = backtester.run(
            bundle, model_class, opt, strategies,
            regime_model_kwargs=model_kwargs,
            progress_callback=lambda p: progress.progress(min(p, 1.0)),
        )
        st.session_state.backtest_result = result
        st.success("Бэктест выполнен.")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Ошибка бэктеста: {exc}")

result = st.session_state.get("backtest_result")
if result is None:
    st.info("Запустите бэктест, чтобы увидеть метрики.")
    st.stop()

eq = result.equity_curves
dr = result.daily_returns

st.subheader("Кривые накопленной доходности")
fig_eq = go.Figure()
for col in eq.columns:
    color = STRATEGY_COLORS.get(col, None)
    fig_eq.add_trace(go.Scatter(x=eq.index, y=eq[col], mode="lines", name=col, line=dict(color=color)))
fig_eq.update_yaxes(type="log" if logscale else "linear")
fig_eq.update_layout(
    height=460, template=PLOTLY_TEMPLATE,
    xaxis_title="Дата", yaxis_title="Рост 1 ₽",
)
st.plotly_chart(fig_eq, use_container_width=True)

st.subheader("Просадки")
draws = eq.apply(lambda x: (x - x.cummax()) / x.cummax())
fig_dd = go.Figure()
for col in draws.columns:
    fig_dd.add_trace(go.Scatter(x=draws.index, y=draws[col], mode="lines", name=col,
                                line=dict(color=STRATEGY_COLORS.get(col, None))))
fig_dd.update_layout(yaxis_tickformat=".0%", height=360, template=PLOTLY_TEMPLATE)
st.plotly_chart(fig_dd, use_container_width=True)

st.subheader("Скользящий коэффициент Шарпа (252 дня)")
# Окно ≤1 года, но min_periods=63 (≈3 месяца) — чтобы левая половина графика не была пустой
window = min(252, max(63, len(dr) // 4))
min_periods = min(63, max(20, window // 4))
if len(dr) > min_periods:
    rs = dr.rolling(window, min_periods=min_periods).apply(
        lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
    )
    fig_rs = go.Figure()
    for col in rs.columns:
        fig_rs.add_trace(
            go.Scatter(
                x=rs.index, y=rs[col], mode="lines", name=col,
                line=dict(color=STRATEGY_COLORS.get(col, None)),
            )
        )
    fig_rs.update_layout(
        height=340, yaxis_title="Sharpe", xaxis_title="Дата",
        template=PLOTLY_TEMPLATE,
    )
    st.plotly_chart(fig_rs, use_container_width=True)
    st.caption(
        f"Окно — {window} торговых дней, минимум наблюдений для расчёта — {min_periods}."
    )

st.subheader("Сводные метрики")
metrics_df = result.metrics.copy()
pct_cols = [c for c in metrics_df.columns if c in
            ("ann_return", "ann_volatility", "max_drawdown", "var_95", "cvar_95",
             "alpha_annual", "tracking_error", "hit_ratio", "turnover")]
fmt = {c: "{:.2%}" for c in pct_cols}
fmt.update({c: "{:.3f}" for c in metrics_df.columns if c not in pct_cols})
try:
    styled = metrics_df.style.format(fmt)
    if "ann_return" in metrics_df.columns:
        styled = styled.highlight_max(axis=0, subset=["ann_return", "sharpe_ratio", "sortino_ratio"], color="lightgreen")
    if "max_drawdown" in metrics_df.columns:
        styled = styled.highlight_max(axis=0, subset=["max_drawdown"], color="lightgreen")  # max DD = ближе к 0
    st.dataframe(styled, use_container_width=True)
except Exception:
    st.dataframe(metrics_df, use_container_width=True)

st.subheader("Риск-доходность")
if {"ann_return", "ann_volatility"}.issubset(metrics_df.columns):
    sc = metrics_df[["ann_return", "ann_volatility"]].copy()
    fig_sc = px.scatter(
        sc, x="ann_volatility", y="ann_return", text=sc.index,
        labels={"ann_volatility": "Волатильность (годовая)", "ann_return": "Доходность (годовая)"},
    )
    fig_sc.update_traces(textposition="top center")
    fig_sc.update_layout(xaxis_tickformat=".0%", yaxis_tickformat=".0%", height=420, template=PLOTLY_TEMPLATE)
    st.plotly_chart(fig_sc, use_container_width=True)

if "IMOEX" in dr.columns and "Adaptive" in dr.columns:
    st.subheader("α/β регрессия Adaptive vs IMOEX")
    fig_reg = px.scatter(
        x=dr["IMOEX"], y=dr["Adaptive"], trendline="ols",
        labels={"x": "IMOEX дневная доходность", "y": "Adaptive дневная доходность"},
    )
    fig_reg.update_layout(height=420, template=PLOTLY_TEMPLATE)
    st.plotly_chart(fig_reg, use_container_width=True)

if not result.stats_significance.empty:
    with st.expander("Статистическая значимость (α, бутстрап Шарпа, Дибольд–Мариано)"):
        st.dataframe(result.stats_significance, use_container_width=True)

if not result.rebalance_log.empty:
    with st.expander("Лог ребалансировок"):
        st.dataframe(result.rebalance_log.tail(50), use_container_width=True)
