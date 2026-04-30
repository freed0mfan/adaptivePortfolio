"""Страница идентификации режимов волатильности."""
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

from src.regime import (
    GaussianHMMRegimeModel,
    MSARRegimeModel,
    MSGARCHRegimeModel,
    RegimeInterpreter,
    RegimeSelector,
)
from src.utils import HISTORICAL_EVENTS, REGIME_COLORS

PLOTLY_TEMPLATE = "plotly_dark"

st.title("🔍 Идентификация режимов волатильности")
st.caption(
    "MS-GARCH(1,1) в спецификации Haas–Mittnik–Paolella (2004), MS-AR (Hamilton, 1989) "
    "и Gaussian HMM. EM-алгоритм; критерии AIC/BIC для выбора числа режимов K."
)

bundle = st.session_state.get("data_bundle")
if bundle is None:
    st.warning("Сначала загрузите данные на странице **📥 Данные**.")
    st.stop()

returns_series = bundle.returns.mean(axis=1)
returns_series.name = "Equal-weighted portfolio return"

with st.sidebar:
    st.header("Параметры модели")
    spec = st.selectbox(
        "Спецификация",
        [
            "MS-GARCH(1,1) (Haas et al., 2004)",
            "MS-AR (statsmodels)",
            "GaussianHMM (hmmlearn)",
        ],
    )
    auto_k = st.checkbox("Автоподбор K по AIC", value=False)
    crit = st.selectbox(
        "Критерий выбора K", ["AIC", "BIC"], index=0, disabled=not auto_k,
    )
    k = st.slider("Число режимов K", 1, 4, 2, disabled=auto_k,
                  help="K=1 — модель вырождается в обычную одномодельную задачу (MVO).")
    order = st.slider(
        "Порядок AR",
        min_value=0, max_value=2, value=0,
        disabled=("MS-AR" not in spec),
    )

run = st.button("Оценить модель", type="primary")
if run:
    selection_result = None
    with st.spinner("EM-алгоритм работает..."):
        try:
            if "MS-GARCH" in spec:
                model_class = MSGARCHRegimeModel
                build_kwargs = lambda K: {"k_regimes": K}
            elif "MS-AR" in spec:
                model_class = MSARRegimeModel
                build_kwargs = lambda K: {"k_regimes": K, "order": order}
            else:
                model_class = GaussianHMMRegimeModel
                build_kwargs = lambda K: {"k_regimes": K}

            if auto_k:
                # MS-AR через statsmodels не поддерживает K=1 — стартуем с 2
                k_min = 1 if model_class is not MSARRegimeModel else 2
                selection_result = RegimeSelector.select(
                    returns_series, k_max=4, k_min=k_min,
                    model_class=model_class,
                    criterion=crit.lower(),
                )
                k_use = selection_result.recommended_k
            else:
                k_use = k
            # MS-AR не поддерживает K=1
            if model_class is MSARRegimeModel and k_use < 2:
                st.warning("MS-AR не поддерживает K=1. Устанавливаю K=2.")
                k_use = 2
            model = model_class(**build_kwargs(k_use))
            model.fit(returns_series)
            st.session_state.regime_model = model
            st.session_state.regime_returns_series = returns_series
            st.session_state.regime_selection = selection_result
            # Сбросить пользовательские названия режимов при пересчёте
            st.session_state.regime_labels = {}
            st.success(f"Модель обучена. K={k_use}.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Ошибка обучения: {exc}")

model = st.session_state.get("regime_model")
if model is None:
    st.info("Нажмите **Оценить модель**, чтобы идентифицировать режимы.")
    st.stop()

returns_series = st.session_state.get("regime_returns_series", returns_series)
params = model.get_regime_params()
filtered = model.get_filtered_proba(returns_series)
custom_labels: dict[int, str] = st.session_state.get("regime_labels", {})
labels = RegimeInterpreter.assign_labels(params, custom=custom_labels)

# ----------------------------- кастомные имена режимов
with st.expander("✏️ Названия режимов (можно переопределить)"):
    st.caption(
        "Введите свои названия для каждого режима. Изменения сразу применяются ко всем графикам."
    )
    new_custom = dict(custom_labels)
    cols = st.columns(min(params.k, 4))
    for k_i in range(params.k):
        col = cols[k_i % len(cols)]
        default = labels[k_i]
        new_val = col.text_input(
            f"Режим {k_i}",
            value=new_custom.get(k_i, default),
            key=f"regime_label_input_{k_i}",
        )
        new_custom[k_i] = new_val
    if new_custom != custom_labels:
        st.session_state.regime_labels = new_custom
        labels = RegimeInterpreter.assign_labels(params, custom=new_custom)

summary = RegimeInterpreter.summarize(params, filtered)
# Вшиваем актуальные пользовательские метки в summary
summary["regime_label"] = summary["regime_id"].map(lambda i: labels[int(i)])

episodes = RegimeInterpreter.annotate_episodes(filtered, threshold=0.0, labels=labels)

st.subheader("Параметры режимов")
fmt_summary = summary.copy()
for col in ("mu_annual", "sigma_annual", "time_fraction", "stationary_proba"):
    fmt_summary[col] = fmt_summary[col].astype(float)
st.dataframe(
    fmt_summary.style.format(
        {
            "mu_annual": "{:.2%}",
            "sigma_annual": "{:.2%}",
            "time_fraction": "{:.2%}",
            "stationary_proba": "{:.2%}",
            "avg_duration_days": "{:.1f}",
        }
    ),
    use_container_width=True,
)

c1, c2, c3 = st.columns(3)
c1.metric("Log-likelihood", f"{params.log_likelihood:.2f}")
c2.metric("AIC", f"{params.aic:.1f}")
c3.metric("BIC", f"{params.bic:.1f}")

st.subheader("Доходности с подсветкой режимов")
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=returns_series.index, y=returns_series.values, mode="lines",
        name="Доходность", line=dict(color="#E0E0E0", width=1),
    )
)
# Цвет линии адаптирован к тёмной теме (был чёрный)
for k_i in range(params.k):
    color = REGIME_COLORS[k_i % len(REGIME_COLORS)]
    if not episodes.empty:
        for _, row in episodes.iterrows():
            if row["regime_id"] == k_i:
                fig.add_vrect(
                    x0=row["start_date"], x1=row["end_date"],
                    fillcolor=color, opacity=0.28, line_width=0, layer="below",
                )
for date_str, label in HISTORICAL_EVENTS.items():
    try:
        d = pd.to_datetime(date_str)
        if returns_series.index.min() <= d <= returns_series.index.max():
            fig.add_vline(
                x=d, line_dash="dot", line_color="#9E9E9E",
                annotation_text=label, annotation_position="top right",
            )
    except Exception:
        continue
fig.update_layout(
    height=480, xaxis_title="Дата", yaxis_title="Доходность",
    template=PLOTLY_TEMPLATE,
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Апостериорные вероятности режимов")
fig2 = go.Figure()
for k_i in range(params.k):
    color = REGIME_COLORS[k_i % len(REGIME_COLORS)]
    fig2.add_trace(
        go.Scatter(
            x=filtered.index, y=filtered[f"regime_{k_i}"],
            stackgroup="one", name=labels[k_i], line=dict(color=color),
        )
    )
fig2.update_layout(
    height=350, yaxis=dict(range=[0, 1]),
    template=PLOTLY_TEMPLATE,
)
st.plotly_chart(fig2, use_container_width=True)

st.subheader("Матрица переходных вероятностей")
P = params.transition_matrix
fig3 = px.imshow(
    P, text_auto=".3f", color_continuous_scale="Blues",
    labels={"x": "В режим", "y": "Из режима"}, aspect="auto",
)
fig3.update_layout(
    height=360,
    xaxis=dict(tickmode="array", tickvals=list(range(params.k)), ticktext=[labels[k] for k in range(params.k)]),
    yaxis=dict(tickmode="array", tickvals=list(range(params.k)), ticktext=[labels[k] for k in range(params.k)]),
    template=PLOTLY_TEMPLATE,
)
st.plotly_chart(fig3, use_container_width=True)

if st.session_state.get("regime_selection") is not None:
    st.subheader("Информационные критерии (AIC/BIC)")
    sel = st.session_state.regime_selection
    df_sel = pd.DataFrame({"AIC": sel.aic_table, "BIC": sel.bic_table})
    fig4 = px.line(df_sel, markers=True, labels={"value": "Критерий", "index": "K", "variable": "Метрика"})
    fig4.update_layout(template=PLOTLY_TEMPLATE)
    st.plotly_chart(fig4, use_container_width=True)
    st.caption(f"Рекомендуемое K по выбранному критерию: **{sel.recommended_k}**.")

with st.expander("Эпизоды доминирующих режимов"):
    st.dataframe(episodes, use_container_width=True)
