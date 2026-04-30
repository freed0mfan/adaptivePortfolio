"""Страница режимной оптимизации портфеля."""
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

from src.optimizer import RegimeOptimizer
from src.regime import RegimeInterpreter
from src.utils import REGIME_COLORS

PLOTLY_TEMPLATE = "plotly_dark"

st.title("⚖️ Режимная оптимизация портфеля")
st.caption("MVO/CVaR через CVXPY. Мягкое смешивание режимных портфелей.")

bundle = st.session_state.get("data_bundle")
model = st.session_state.get("regime_model")
if bundle is None or model is None:
    st.warning("Сначала загрузите данные и обучите модель режимов.")
    st.stop()

with st.sidebar:
    st.header("Параметры оптимизации")
    col_l1, col_l2 = st.columns([3, 1])
    lambda_risk = col_l1.slider("λ — неприятие риска", 0.5, 10.0, 2.0, 0.5)
    with col_l2.popover("ℹ️", help="Что такое λ"):
        st.markdown(
            "**λ — коэффициент неприятия риска инвестора.**\n\n"
            "В задаче MVO максимизируется функционал \\( w^\\top \\mu - \\tfrac{\\lambda}{2}\\, w^\\top \\Sigma\\, w \\): "
            "большее λ означает большее наказание за дисперсию портфеля, поэтому инвестор предпочитает "
            "менее волатильные распределения весов.\n\n"
            "На практике λ ≈ 2 соответствует умеренной агрессии, λ ≥ 5 — консервативному инвестору, "
            "λ ≤ 1 — агрессивному."
        )
    max_weight = st.slider("Макс. вес одного актива", 0.05, 1.0, 0.30, 0.05)
    col_c1, col_c2 = st.columns([3, 1])
    use_cvar = col_c1.checkbox("Оптимизация около CVaR", value=False)
    with col_c2.popover("ℹ️", help="Про CVaR"):
        st.markdown(
            "**CVaR (Conditional Value at Risk).** Здесь *не* подменяется дисперсия в формуле MVO; "
            "вместо этого функционал портфеля строится **около CVaR**: минимизируется "
            "ожидаемый убыток в худшем хвосте (1−α) распределения сценариев.\n\n"
            "Формально решается задача Rockafellar–Uryasev (2000):\n\n"
            "\\[\\min_{w,\\eta}\\; \\eta + \\tfrac{1}{(1-\\alpha) S}\\sum_{s=1}^{S} \\max(-w^\\top r_s - \\eta,\\, 0)\\]\n\n"
            "при ограничениях \\(\\sum w = 1,\\ w \\ge 0\\). Параметр α — уровень доверия (0.95/0.99)."
        )
    cvar_conf = st.slider("Уровень CVaR (α)", 0.90, 0.99, 0.95, 0.01, disabled=not use_cvar)

    col_r1, col_r2 = st.columns([3, 1])
    rebalance_thr = col_r1.number_input("Порог ребалансировки (L1)", 0.01, 0.20, 0.02, 0.01)
    with col_r2.popover("ℹ️", help="Порог ребалансировки"):
        st.markdown(
            "**Порог ребалансировки (L1).**\n\n"
            "Наблюдаемая величина \\(\\|w_t - w_{t-1}\\|_1 = \\sum_i |w_{i,t} - w_{i,t-1}|\\) "
            "— это полный оборот (turnover) портфеля от баланса к балансу.\n\n"
            "Ребалансировка выполняется, только если эта величина превышает заданный порог — "
            "это сдерживает излишние торговые операции и снижает транзакционные издержки.\n\n"
            "На практике 0.02–0.05 — умеренные значения, 0.10+ — «ленивый» подход."
        )

    col_w1, col_w2 = st.columns([3, 1])
    estimation_window = col_w1.number_input("Окно оценки (дни)", 60, 1260, 252, 21)
    with col_w2.popover("ℹ️", help="Окно оценки"):
        st.markdown(
            "**Окно оценки (rolling window).**\n\n"
            "Количество последних торговых дней, по которым оцениваются режимные моменты: "
            "вектор ожидаемых доходностей \\(\\hat\\mu_k\\) и ковариационная матрица \\(\\hat\\Sigma_k\\) "
            "для каждого режима k.\n\n"
            "Короткое окно (60–120) — быстрая реакция на изменения рынка, но больше шума. "
            "Длинное окно (252–1260, т.е. ≈1–5 лет) — устойчивые оценки, но режимы размываются.\n\n"
            "Порекомендованный диапазон — 252 (≈1 год)."
        )

run = st.button("Оптимизировать", type="primary")
if run:
    with st.spinner("CVXPY решает задачу..."):
        try:
            opt = RegimeOptimizer(
                lambda_risk=lambda_risk,
                use_cvar=use_cvar,
                cvar_confidence=cvar_conf,
                max_weight=max_weight,
                rebalance_threshold=rebalance_thr,
                estimation_window=int(estimation_window),
            )
            opt.fit(bundle, model)
            st.session_state.optimizer = opt
            st.success("Оптимизация выполнена.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Ошибка оптимизации: {exc}")

opt = st.session_state.get("optimizer")
if opt is None:
    st.info("Запустите оптимизацию, чтобы увидеть результаты.")
    st.stop()

params = model.get_regime_params()
custom_labels = st.session_state.get("regime_labels", {})
labels = RegimeInterpreter.assign_labels(params, custom=custom_labels)
all_w = opt.get_all_regime_weights()
all_w.columns = [labels[int(c.split("_")[1])] for c in all_w.columns]

# Текущий доминирующий режим
filtered = model.get_filtered_proba(bundle.returns.mean(axis=1))
pi_now = filtered.iloc[-1].values
current_regime = int(np.argmax(pi_now))

st.subheader("Режимные оптимальные веса")
st.caption(
    f"Доминирующий режим на момент последнего наблюдения: **{labels[current_regime]}** "
    f"(π = {pi_now[current_regime]:.2%}). График этого режима выделен жёлтой рамкой."
)

# Pie chart на каждый режим + параметры портфеля
n_per_row = min(params.k, 3)
rows = [list(range(i, min(i + n_per_row, params.k))) for i in range(0, params.k, n_per_row)]
for row_indices in rows:
    cols = st.columns(len(row_indices))
    for col, k_i in zip(cols, row_indices):
        color = REGIME_COLORS[k_i % len(REGIME_COLORS)]
        weights = opt.get_regime_weights(k_i)
        weights = weights[weights > 1e-4]  # обрезаем нулевые куски — pie chart чище
        try:
            stats = opt.get_regime_portfolio_stats(k_i)
        except Exception:
            stats = {"return_annual": 0.0, "vol_annual": 0.0, "sharpe": 0.0}
        is_current = (k_i == current_regime)
        title_prefix = "★ " if is_current else ""
        with col:
            border_color = "#FFD600" if is_current else "rgba(0,0,0,0)"
            st.markdown(
                f"<div style='border:2px solid {border_color}; border-radius:12px; "
                f"padding:8px;'>"
                f"<h4 style='margin:0; color:{color};'>{title_prefix}{labels[k_i]}</h4>"
                f"</div>",
                unsafe_allow_html=True,
            )
            fig_p = px.pie(
                values=weights.values, names=weights.index,
                hole=0.40,
            )
            fig_p.update_traces(textinfo="label+percent")
            fig_p.update_layout(
                height=320, template=PLOTLY_TEMPLATE,
                showlegend=False, margin=dict(t=10, b=10, l=10, r=10),
            )
            st.plotly_chart(fig_p, use_container_width=True)
            m1, m2, m3 = st.columns(3)
            m1.metric("Доходн. (год)", f"{stats['return_annual']:.2%}")
            m2.metric("Волат. (год)", f"{stats['vol_annual']:.2%}")
            m3.metric("Sharpe", f"{stats['sharpe']:.2f}")

st.subheader("Текущий смешанный портфель (soft-blend)")
pw = opt.compute_portfolio_weights(pi_now)
mix = pw.weights
mix_nonzero = mix[mix > 1e-4]

c1, c2 = st.columns([1.1, 1])
fig_pie = px.pie(values=mix_nonzero.values, names=mix_nonzero.index, hole=0.30)
fig_pie.update_traces(textinfo="label+percent")
fig_pie.update_layout(template=PLOTLY_TEMPLATE, height=380)
c1.plotly_chart(fig_pie, use_container_width=True)
c2.metric(
    "Доминирующий режим (последний день)", labels[current_regime],
    delta=f"π={pi_now.max():.2%}",
)
c2.dataframe(
    pd.DataFrame({"вес π": pi_now}, index=[labels[k] for k in range(params.k)])
    .style.format("{:.2%}"),
    use_container_width=True,
)

st.subheader("Эффективные границы по режимам")
st.caption(
    "Для каждого режима строится отдельный λ-sweep MVO. На границе помечены "
    "**MinVar** (●) и **MaxSharpe** (★). Наведите курсор на любую точку, чтобы увидеть веса портфеля."
)
tickers = list(bundle.returns.columns)
n_assets = len(tickers)
# Отдельный график на каждый режим, по два в ряду
_ef_per_row = 2
ef_rows = [list(range(i, min(i + _ef_per_row, params.k))) for i in range(0, params.k, _ef_per_row)]
for row_indices in ef_rows:
    cols_ef = st.columns(len(row_indices))
    for col_ef, k_i in zip(cols_ef, row_indices):
        try:
            front = opt.get_efficient_frontier(k_i, n_points=40)
        except Exception as exc:  # noqa: BLE001
            col_ef.warning(f"Эффективная граница для k={k_i} не построена: {exc}")
            continue
        color = REGIME_COLORS[k_i % len(REGIME_COLORS)]
        vol_annual = front["volatility"].values * np.sqrt(252)
        ret_annual = front["return"].values * 252

        weights_array = front[[f"w_{t}" for t in tickers]].values
        hover_lines = [
            "<br>".join([f"{t}: {wi:.1%}" for t, wi in zip(tickers, row) if abs(wi) > 1e-3])
            for row in weights_array
        ]
        hover_text = [
            f"<b>{labels[k_i]}</b><br>Доходн. (год): {ra:.2%}<br>Волат. (год): {va:.2%}<br>"
            f"Sharpe: {(ra/va) if va > 1e-10 else 0:.2f}<br>λ={lam:.2f}<br><br>{wt}"
            for ra, va, lam, wt in zip(ret_annual, vol_annual, front["lambda"], hover_lines)
        ]

        fig_ef = go.Figure()
        fig_ef.add_trace(
            go.Scatter(
                x=vol_annual, y=ret_annual, mode="lines+markers",
                name="Эфф. граница",
                line=dict(color=color, width=2),
                marker=dict(size=5),
                text=hover_text, hoverinfo="text", showlegend=False,
            )
        )

        # MinVar и MaxSharpe маркеры
        try:
            mv = opt.get_min_variance_weights(k_i).values
            sigma_k = opt._regime_sigma[k_i]
            mu_k = opt._regime_mu[k_i]
            mv_vol = float(np.sqrt(max(mv @ sigma_k @ mv, 0))) * np.sqrt(252)
            mv_ret = float(mu_k @ mv) * 252
            mv_text = "<br>".join([f"{t}: {wi:.1%}" for t, wi in zip(tickers, mv) if abs(wi) > 1e-3])
            fig_ef.add_trace(
                go.Scatter(
                    x=[mv_vol], y=[mv_ret], mode="markers",
                    marker=dict(symbol="circle", size=14, color=color, line=dict(width=2, color="#FFFFFF")),
                    name="MinVar",
                    text=[f"<b>MinVar</b><br>Доходн. (год): {mv_ret:.2%}<br>Волат. (год): {mv_vol:.2%}<br><br>{mv_text}"],
                    hoverinfo="text",
                )
            )
        except Exception:
            pass
        try:
            ms = opt.get_max_sharpe_weights(k_i).values
            ms_vol = float(np.sqrt(max(ms @ sigma_k @ ms, 0))) * np.sqrt(252)
            ms_ret = float(mu_k @ ms) * 252
            ms_text = "<br>".join([f"{t}: {wi:.1%}" for t, wi in zip(tickers, ms) if abs(wi) > 1e-3])
            fig_ef.add_trace(
                go.Scatter(
                    x=[ms_vol], y=[ms_ret], mode="markers",
                    marker=dict(symbol="star", size=18, color=color, line=dict(width=2, color="#FFFFFF")),
                    name="MaxSharpe",
                    text=[f"<b>MaxSharpe</b><br>Доходн. (год): {ms_ret:.2%}<br>Волат. (год): {ms_vol:.2%}<br><br>{ms_text}"],
                    hoverinfo="text",
                )
            )
        except Exception:
            pass

        is_current = (k_i == current_regime)
        title_prefix = "★ " if is_current else ""
        fig_ef.update_layout(
            title=dict(
                text=f"<span style='color:{color}'>{title_prefix}{labels[k_i]}</span>",
                x=0.0, xanchor="left",
            ),
            xaxis_title="Волатильность (годовая)",
            yaxis_title="Ожидаемая доходность (годовая)",
            height=380, xaxis_tickformat=".0%", yaxis_tickformat=".0%",
            template=PLOTLY_TEMPLATE,
            margin=dict(t=50, b=40, l=40, r=20),
            hoverlabel=dict(bgcolor="#222", font_size=12),
            legend=dict(orientation="h", yanchor="bottom", y=-0.30),
        )
        col_ef.plotly_chart(fig_ef, use_container_width=True)

with st.expander("Веса в табличном виде"):
    st.dataframe(all_w.style.format("{:.2%}"), use_container_width=True)
