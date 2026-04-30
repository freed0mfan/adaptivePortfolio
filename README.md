# AdaptivePortfolio

Система адаптивной портфельной оптимизации на основе моделей переключения режимов
волатильности (Markov-Switching). Применение к российскому рынку акций (IMOEX).

> Программный артефакт курсовой работы Маркелова Егора | НГУ, 3 курс, Бизнес-информатика, 2026.

---

## Возможности

- 📥 Загрузка котировок IMOEX через MOEX ISS REST API (с локальным parquet-кэшем) и
fallback-генератор синтетических данных для оффлайн-демонстрации и тестов.
- 🔍 Идентификация скрытых режимов через **MS-AR** (`statsmodels`) или **Gaussian HMM**
(`hmmlearn`) с автоматическим выбором числа режимов K по AIC/BIC.
- ⚖️ Режимно-специфическая оптимизация: **MVO** или **CVaR** через `cvxpy`,
ковариация Ledoit–Wolf, мягкое смешивание портфелей по апостериорным вероятностям.
- 📊 Walk-forward бэктест с шестью бенчмарками: `IMOEX`, `EW`, `Static MVO`, `Risk Parity`, `Min Variance`, `Max Sharpe`. 13 метрик, t-тест α, блочный бутстрап Шарпа,
тест Дибольда–Мариано.
- 📄 Интерактивный Streamlit-дашборд (6 страниц) и экспорт результатов в Excel/CSV/PDF.

---

## Установка

Требуется Python ≥ 3.11.

```bash
git clone <adaptivePortfolio>.git adaptive_portfolio
cd adaptive_portfolio
python -m venv .venv && source .venv\Scripts\activate
pip install -r requirements.txt
```

## Запуск дашборда

```bash
streamlit run app/main.py
```

Откроется браузер по адресу `http://localhost:8501`. Если интернет недоступен — на
странице **📥 Данные** включите чекбокс **«Демо-режим (синтетические данные)»**.

## Запуск тестов

```bash
pytest tests/ --cov=src -v
```

## Структура

```
adaptive_portfolio/
├── src/
│   ├── data/         # MoexDataLoader, DataBundle, предобработка
│   ├── regime/       # MS-AR, GaussianHMM, RegimeSelector, RegimeInterpreter
│   ├── optimizer/    # MVO/CVaR через CVXPY, RegimeOptimizer, soft blending
│   ├── backtest/     # Backtester, стратегии, метрики, статистические тесты
│   └── utils/        # Конфиги, исключения, IO
├── app/              # Streamlit-дашборд (6 страниц)
├── tests/            # pytest, фикстуры с синтетикой
├── data/cache/       # parquet-кэш
└── requirements.txt
```

## Быстрый старт (Python API)

```python
from datetime import date

from src.data import MoexDataLoader
from src.regime import GaussianHMMRegimeModel
from src.optimizer import RegimeOptimizer
from src.backtest import Backtester, EqualWeightStrategy, ImoexStrategy

loader = MoexDataLoader()
bundle = loader.load(
    ["SBER", "LKOH", "GAZP", "GMKN", "ROSN"],
    start=date(2020, 1, 1), end=date(2024, 12, 31),
)

model = GaussianHMMRegimeModel(k_regimes=2)
model.fit(bundle.returns.mean(axis=1))

optimizer = RegimeOptimizer(lambda_risk=2.0, max_weight=0.30)
optimizer.fit(bundle, model)

result = Backtester(train_window=252, test_window=21).run(
    bundle, GaussianHMMRegimeModel, optimizer,
    strategies=[EqualWeightStrategy(), ImoexStrategy()],
    regime_model_kwargs={"k_regimes": 2},
)
print(result.metrics.round(3))
```

---

## Что важно знать

- **Воспроизводимость.** Все стохастические операции используют `random_state=42`.
- **Ограничения.** Только дневные/недельные данные, без коротких продаж, без
дивидендов.  Безрисковая ставка по умолчанию 16% годовых (ставка ЦБ РФ).
- **MOEX ISS.** Актуально для акций режима TQBR. При сетевой ошибке система
автоматически переходит на синтетические данные с логированием.
- **CVXPY-solvers.** По умолчанию используется CLARABEL; при отсутствии — SCS/ECOS.

