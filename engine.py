"""Walk-forward бэктестер."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional, Type

import numpy as np
import pandas as pd

from ..data.loader import DataBundle
from ..optimizer.regime_optimizer import RegimeOptimizer
from ..regime.base import RegimeModel
from ..utils.config import DEFAULT_RF_RATE_DAILY
from .metrics import PerformanceMetrics
from .stats import StatisticsModule
from .strategies import BaseStrategy

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    equity_curves: pd.DataFrame
    daily_returns: pd.DataFrame
    metrics: pd.DataFrame
    weights_history: pd.DataFrame
    regime_history: pd.DataFrame
    rebalance_log: pd.DataFrame
    stats_significance: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    metadata: Dict = field(default_factory=dict)


class Backtester:
    """Walk-forward оценка стратегий."""

    def __init__(
        self,
        train_window: int = 252,
        test_window: int = 21,
        step: int = 21,
        tc_bps: float = 10.0,
        rf_daily: float = DEFAULT_RF_RATE_DAILY,
    ):
        self.train_window = int(train_window)
        self.test_window = int(test_window)
        self.step = int(step)
        self.tc_bps = float(tc_bps)
        self.rf_daily = float(rf_daily)

    def run(
        self,
        data_bundle: DataBundle,
        regime_model_class: Type[RegimeModel],
        optimizer: RegimeOptimizer,
        strategies: Optional[List[BaseStrategy]] = None,
        regime_model_kwargs: Optional[dict] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
        index_weights: Optional[pd.Series] = None,
    ) -> BacktestResult:
        regime_model_kwargs = regime_model_kwargs or {}
        strategies = strategies or []
        returns = data_bundle.returns
        T = len(returns)
        if T < self.train_window + self.test_window:
            raise ValueError(
                f"Недостаточно данных: T={T}, требуется {self.train_window + self.test_window}"
            )

        adaptive_rets: Dict[pd.Timestamp, float] = {}
        adaptive_weights: Dict[pd.Timestamp, pd.Series] = {}
        regime_records: Dict[pd.Timestamp, dict] = {}
        rebalance_log: List[dict] = []

        strategy_rets: Dict[str, Dict[pd.Timestamp, float]] = {s.name: {} for s in strategies}

        windows_total = max((T - self.train_window) // self.step, 1)
        win_idx = 0

        t = self.train_window
        while t + self.test_window <= T:
            train_slice = returns.iloc[t - self.train_window : t]
            test_slice = returns.iloc[t : t + self.test_window]
            equal_weighted_train = train_slice.mean(axis=1)

            try:
                model = regime_model_class(**regime_model_kwargs)
                model.fit(equal_weighted_train)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Не удалось обучить модель режимов в окне [%s]: %s", t, exc)
                t += self.step
                continue

            try:
                local_optimizer = RegimeOptimizer(
                    lambda_risk=optimizer.lambda_risk,
                    use_cvar=optimizer.use_cvar,
                    cvar_confidence=optimizer.cvar_confidence,
                    n_cvar_scenarios=optimizer.n_cvar_scenarios,
                    max_weight=optimizer.max_weight,
                    min_weight=optimizer.min_weight,
                    estimation_window=optimizer.estimation_window,
                    rebalance_threshold=optimizer.rebalance_threshold,
                    min_rebalance_days=optimizer.min_rebalance_days,
                    tc_bps=optimizer.tc_bps,
                    random_state=optimizer.random_state,
                )
                local_train_bundle = DataBundle(
                    returns=train_slice,
                    prices=data_bundle.prices.loc[train_slice.index],
                    index_returns=data_bundle.index_returns.loc[
                        data_bundle.index_returns.index.intersection(train_slice.index)
                    ],
                    index_prices=data_bundle.index_prices.loc[
                        data_bundle.index_prices.index.intersection(train_slice.index)
                    ],
                    metadata={"window": "train"},
                )
                local_optimizer.fit(local_train_bundle, model)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Оптимизация упала в окне %s: %s", t, exc)
                t += self.step
                continue

            prev_w: Optional[pd.Series] = None
            days_since = 0
            for d in test_slice.index:
                last_ret = float(test_slice.loc[d].mean())
                pi_t = model.predict_next(last_ret)
                pw = local_optimizer.compute_portfolio_weights(
                    pi_t,
                    prev_weights=prev_w,
                    days_since_rebalance=days_since,
                    timestamp=d,
                )
                w = pw.weights
                ret_d = float((test_slice.loc[d] * w.reindex(test_slice.columns).fillna(0.0)).sum())
                if pw.rebalanced:
                    if prev_w is not None:
                        turnover = float((w - prev_w).abs().sum())
                    else:
                        turnover = 1.0
                    tc = turnover * self.tc_bps * 1e-4
                    ret_d -= tc
                    rebalance_log.append(
                        {
                            "date": d,
                            "turnover": turnover,
                            "tc_cost": tc,
                            "trigger_reason": "regime_shift" if prev_w is not None else "initial",
                        }
                    )
                    days_since = 0
                else:
                    days_since += 1
                adaptive_rets[d] = ret_d
                adaptive_weights[d] = w.copy()
                regime_records[d] = {
                    "dominant_regime": int(np.argmax(pi_t)),
                    **{f"pi_{k}": float(pi_t[k]) for k in range(len(pi_t))},
                }
                prev_w = w

            # бенчмарки: пересчёт весов раз в окно
            for strat in strategies:
                # IMOEX — взвешен по капитализации; берём фактические доходности индекса из MOEX API
                if strat.name == "IMOEX":
                    idx_ret = data_bundle.index_returns
                    for d in test_slice.index:
                        if d in idx_ret.index:
                            strategy_rets[strat.name][d] = float(idx_ret.loc[d])
                        else:
                            strategy_rets[strat.name][d] = float("nan")
                    continue
                try:
                    sw = strat.compute_weights(train_slice, as_of=test_slice.index[0], index_weights=None)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Стратегия %s упала на %s: %s", strat.name, t, exc)
                    sw = pd.Series(np.ones(returns.shape[1]) / returns.shape[1], index=returns.columns)
                sw = sw.reindex(returns.columns).fillna(0.0)
                sw = sw / sw.sum() if sw.sum() > 0 else sw
                for d in test_slice.index:
                    strategy_rets[strat.name][d] = float((test_slice.loc[d] * sw).sum())

            win_idx += 1
            if progress_callback:
                progress_callback(min(win_idx / max(windows_total, 1), 1.0))
            t += self.step

        if progress_callback:
            progress_callback(1.0)

        # Соберём итоговые данные
        adaptive_series = pd.Series(adaptive_rets).sort_index()
        adaptive_series.name = "Adaptive"
        weights_history = pd.DataFrame.from_dict(adaptive_weights, orient="index").sort_index()
        regime_history = pd.DataFrame.from_dict(regime_records, orient="index").sort_index()

        all_returns = {"Adaptive": adaptive_series}
        for s in strategies:
            ser = pd.Series(strategy_rets[s.name]).sort_index()
            ser.name = s.name
            all_returns[s.name] = ser
        daily_returns = pd.DataFrame(all_returns)
        equity_curves = np.exp(daily_returns.fillna(0.0).cumsum())

        # Метрики
        bench_for_metrics = None
        if "IMOEX" in daily_returns.columns:
            bench_for_metrics = daily_returns["IMOEX"]
        metrics_rows = {}
        for col in daily_returns.columns:
            wh = weights_history if col == "Adaptive" else None
            metrics_rows[col] = PerformanceMetrics.compute(
                daily_returns[col], benchmark_returns=bench_for_metrics, rf=self.rf_daily,
                weights_history=wh,
            )
        metrics_df = pd.DataFrame(metrics_rows).T

        # Статистика значимости
        stats_rows = []
        if bench_for_metrics is not None:
            for col in daily_returns.columns:
                if col == "IMOEX":
                    continue
                row = {"strategy": col}
                row.update(StatisticsModule.alpha_ttest(daily_returns[col], bench_for_metrics))
                point, lo, hi = StatisticsModule.sharpe_bootstrap_ci(daily_returns[col])
                row["sharpe_point_bs"] = point
                row["sharpe_ci_low"] = lo
                row["sharpe_ci_high"] = hi
                if "Adaptive" in daily_returns.columns and col != "Adaptive":
                    e_strat = daily_returns[col] - bench_for_metrics
                    e_adap = daily_returns["Adaptive"] - bench_for_metrics
                    dm = StatisticsModule.diebold_mariano_test(e_adap, e_strat)
                    row["dm_stat_vs_adaptive"] = dm["dm_stat"]
                    row["dm_pvalue_vs_adaptive"] = dm["p_value"]
                stats_rows.append(row)
        stats_df = pd.DataFrame(stats_rows).set_index("strategy") if stats_rows else pd.DataFrame()

        return BacktestResult(
            equity_curves=equity_curves,
            daily_returns=daily_returns,
            metrics=metrics_df,
            weights_history=weights_history,
            regime_history=regime_history,
            rebalance_log=pd.DataFrame(rebalance_log),
            stats_significance=stats_df,
            metadata={
                "train_window": self.train_window,
                "test_window": self.test_window,
                "step": self.step,
                "tickers": list(returns.columns),
                "completed_at": datetime.now().isoformat(),
            },
        )
