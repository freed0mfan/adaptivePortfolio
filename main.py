"""Точка входа Streamlit-дашборда AdaptivePortfolio."""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Делаем доступным пакет src/ независимо от рабочего каталога
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="AdaptivePortfolio",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _build_navigation():
    pages = {
        "Система": [
            st.Page("pages/1_home.py", title="🏠 Главная"),
        ],
        "Анализ": [
            st.Page("pages/2_data.py", title="📥 Данные"),
            st.Page("pages/3_regimes.py", title="🔍 Режимы"),
            st.Page("pages/4_optimizer.py", title="⚖️ Оптимизация"),
        ],
        "Результаты": [
            st.Page("pages/5_backtest.py", title="📊 Бэктест"),
            st.Page("pages/6_report.py", title="📄 Отчёт"),
        ],
    }
    return st.navigation(pages)


nav = _build_navigation()
nav.run()
