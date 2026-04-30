"""Страница экспорта отчёта (Excel/CSV/PDF)."""
from __future__ import annotations

import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src.utils.io import dataframes_to_excel

st.title("📄 Отчёт и экспорт")
st.caption(
    "Экспорт результатов в .xlsx, .csv. PDF доступен при наличии reportlab."
)

result = st.session_state.get("backtest_result")
if result is None:
    st.warning("Сначала выполните бэктест на странице **📊 Бэктест**.")
    st.stop()

st.subheader("Сводка")
c1, c2 = st.columns(2)
c1.dataframe(result.metrics, use_container_width=True)
if not result.stats_significance.empty:
    c2.dataframe(result.stats_significance, use_container_width=True)

st.subheader("Скачать данные")
sheets = {
    "Summary Metrics": result.metrics,
    "Equity Curves": result.equity_curves,
    "Daily Returns": result.daily_returns,
    "Weights History": result.weights_history,
    "Regime History": result.regime_history,
    "Rebalance Log": result.rebalance_log,
}
if not result.stats_significance.empty:
    sheets["Stats Significance"] = result.stats_significance

xlsx_bytes = dataframes_to_excel(sheets)
st.download_button(
    "📥 Скачать Excel (.xlsx)", xlsx_bytes, "adaptive_portfolio_report.xlsx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

st.download_button(
    "📥 Скачать веса CSV", result.weights_history.to_csv().encode("utf-8"),
    "weights_history.csv", "text/csv",
)
st.download_button(
    "📥 Скачать метрики CSV", result.metrics.to_csv().encode("utf-8"),
    "metrics.csv", "text/csv",
)

# ----------------------------------------------------------------- PDF block
st.subheader("PDF-отчёт")
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import (
        Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
    )

    if st.button("Сформировать PDF"):
        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4, title="AdaptivePortfolio Report")
        styles = getSampleStyleSheet()
        story = [
            Paragraph("AdaptivePortfolio — отчёт по бэктесту", styles["Title"]),
            Spacer(1, 6),
            Paragraph("Автор: Маркелов Егор. НГУ, 3 курс, Бизнес-информатика, 2026.", styles["BodyText"]),
            Spacer(1, 12),
            Paragraph(
                "Адаптивная портфельная оптимизация на основе моделей переключения режимов "
                "волатильности (MS-GARCH(1,1) в спецификации Haas–Mittnik–Paolella, 2004). "
                "Российский рынок акций, индекс МосБиржи.",
                styles["BodyText"],
            ),
            Spacer(1, 18),
            Paragraph("Сводные метрики:", styles["Heading2"]),
        ]
        df = result.metrics.round(4)
        data = [["Стратегия"] + list(df.columns)] + [
            [str(idx)] + [str(v) for v in row] for idx, row in df.iterrows()
        ]
        table = Table(data, hAlign="LEFT")
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("FONTSIZE", (0, 0), (-1, -1), 7),
                ]
            )
        )
        story.append(table)
        doc.build(story)
        st.download_button(
            "📥 Скачать PDF", buf.getvalue(), "adaptive_portfolio_report.pdf",
            "application/pdf",
        )
except ImportError:
    st.info("Для PDF-экспорта установите `reportlab`.")
