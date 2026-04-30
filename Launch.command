#!/usr/bin/env bash
# AdaptivePortfolio — лаунчер для macOS / Linux.
# Двойной клик в Finder запускает дашборд.

set -e
cd "$(dirname "$0")"

echo "============================================================"
echo "  AdaptivePortfolio — адаптивная портфельная оптимизация"
echo "============================================================"
echo

# 1. Поиск Python
PY=""
for cand in python3.13 python3.12 python3.11 python3 python; do
    if command -v "$cand" >/dev/null 2>&1; then
        PY="$cand"
        break
    fi
done

if [ -z "$PY" ]; then
    echo "[ОШИБКА] Python 3.11+ не найден."
    echo "Откройте https://www.python.org/downloads/ и установите Python,"
    echo "после чего снова дважды кликните по этому файлу."
    open "https://www.python.org/downloads/" 2>/dev/null || true
    read -p "Нажмите Enter для выхода..."
    exit 1
fi

echo "[1/4] Найден Python: $($PY -c 'import sys; print(sys.executable)')"

# 2. venv
if [ ! -x ".venv/bin/python" ]; then
    echo "[2/4] Создаю окружение .venv..."
    "$PY" -m venv .venv
else
    echo "[2/4] Окружение .venv уже существует."
fi

VENV_PY="$(pwd)/.venv/bin/python"

# 3. Зависимости
if [ ! -f ".venv/.installed" ]; then
    echo "[3/4] Устанавливаю зависимости (5–10 минут при первом запуске)..."
    "$VENV_PY" -m pip install --upgrade pip wheel >/dev/null
    "$VENV_PY" -m pip install --prefer-binary -r requirements.txt
    touch ".venv/.installed"
else
    echo "[3/4] Зависимости уже установлены."
fi

# 4. Запуск
echo "[4/4] Запускаю дашборд..."
echo "Чтобы остановить — закройте это окно."
echo "============================================================"
echo
exec "$VENV_PY" -m streamlit run app/main.py --browser.gatherUsageStats false
