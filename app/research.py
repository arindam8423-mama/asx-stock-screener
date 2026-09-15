from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from itertools import product
from pathlib import Path

import pandas as pd

from app.backtest.engine import BacktestConfig, run_backtest
from app.backtest.models import BacktestResult
from app.config import settings
from app.data.market_data import download_history
from app.data.test_universe import TEST_UNIVERSE, all_test_tickers
from app.strategies import (
    BreakoutMomentumStrategy,
    BreakoutVariantStrategy,
    MeanReversionStrategy,
    TrendFollowingStrategy,
)


@dataclass(frozen=True)
class MeanReversionParameters:
    lookback_days: int
    stddevs: float
    rsi_period: int
    rsi_threshold: float
    max_holding_days: int


@dataclass(frozen=True)
class OptimizationCandidate:
    parameters: MeanReversionParameters
    result: BacktestResult


@dataclass(frozen=True)
class RobustnessSummary:
    parameter: str
    value: float | int
    candidates: int
    positive_return_count: int
    profitable_factor_count: int
    median_return_pct: float
    median_profit_factor: float


def download_test_universe(start: str, end: str, output_dir: Path | None = None) -> int:
    output_dir = output_dir or settings.prices_dir
    failures = 0
    for ticker in all_test_tickers():
        try:
            path = download_history(ticker, start, end, output_dir)
            print(f"{ticker}: {path}")
        except Exception as exc:  # noqa: BLE001 - report all ticker failures
            failures += 1
            print(f"{ticker}: ERROR: {exc}")
    return failures


def load_test_prices(prices_dir: Path | None = None) -> pd.DataFrame:
    prices_dir = prices_dir or settings.prices_dir
    frames = []
    for ticker in all_test_tickers():
        path = prices_dir / f"{ticker}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing test-universe price data: {path}")
        frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True)


def _group_metrics(result: BacktestResult) -> dict[str, dict[str, float | int]]:
    by_group: dict[str, dict[str, float | int]] = {}
    for group in ("large", "mid", "small"):
        group_trades = [trade for trade in result.trades if trade.cap_group == group]
        net_pnl = sum(trade.net_pnl for trade in group_trades)
        wins = sum(trade.net_pnl > 0 for trade in group_trades)
        by_group[group] = {
            "trades": len(group_trades),
            "net_pnl": net_pnl,
            "win_rate_pct": wins / len(group_trades) * 100 if group_trades else 0.0,
        }
    return by_group


def strategy_definitions() -> dict[str, object]:
    """Return the fixed, deliberately unoptimised strategies used for comparison."""
    return {
        "Momentum breakout": BreakoutMomentumStrategy(lookback_days=20, volume_lookback_days=20),
        "Mean reversion": MeanReversionStrategy(
            lookback_days=20, stddevs=2.0, rsi_period=14, rsi_threshold=30.0
        ),
        "Trend following": TrendFollowingStrategy(fast_days=20, slow_days=50),
        "Filtered breakout": BreakoutVariantStrategy(
            lookback_days=20,
            volume_lookback_days=20,
            volume_multiplier=1.25,
            atr_period=14,
            min_atr_pct=1.0,
            max_atr_pct=8.0,
        ),
    }


def _backtest_config(max_holding_days: int = 20, **kwargs) -> BacktestConfig:
    return BacktestConfig(
        initial_capital=10_000,
        position_size_pct=10,
        max_holding_days=max_holding_days,
        buy_brokerage=11,
        sell_brokerage=11,
        **kwargs,
    )


def run_strategy_comparison(prices_dir: Path | None = None) -> dict[str, dict[str, object]]:
    """Run all research strategies using identical backtest assumptions."""
    prices = load_test_prices(prices_dir)
    config = _backtest_config()
    comparison: dict[str, dict[str, object]] = {}
    for name, strategy in strategy_definitions().items():
        result = run_backtest(prices, strategy, TEST_UNIVERSE, config)
        comparison[name] = {"overall": result, "by_group": _group_metrics(result)}
    return comparison


def run_momentum_research(prices_dir: Path | None = None) -> dict[str, object]:
    prices = load_test_prices(prices_dir)
    result = run_backtest(
        prices,
        BreakoutMomentumStrategy(lookback_days=20, volume_lookback_days=20),
        TEST_UNIVERSE,
        _backtest_config(),
    )
    return {"overall": result, "by_group": _group_metrics(result)}


def _mean_reversion_grid() -> list[MeanReversionParameters]:
    """Return a deliberately bounded grid for the first optimisation pass."""
    return [
        MeanReversionParameters(*values)
        for values in product(
            (10, 15, 20, 30),  # lookback_days
            (1.5, 2.0, 2.5),  # Bollinger-style standard deviations
            (14,),  # RSI period; keep fixed initially to reduce degrees of freedom
            (25.0, 30.0, 35.0),  # RSI threshold
            (5, 10, 15, 20),  # max holding days
        )
    ]


def _mean_reversion_strategy(parameters: MeanReversionParameters) -> MeanReversionStrategy:
    return MeanReversionStrategy(
        lookback_days=parameters.lookback_days,
        stddevs=parameters.stddevs,
        rsi_period=parameters.rsi_period,
        rsi_threshold=parameters.rsi_threshold,
    )


def _candidate_sort_key(candidate: OptimizationCandidate) -> tuple[float, float, float, int]:
    result = candidate.result
    profit_factor = result.profit_factor
    # Infinite PF is allowed, but the minimum-trade filter below prevents a
    # one-trade/no-loss result from winning the search by itself.
    pf_rank = profit_factor if profit_factor != float("inf") else 1_000.0
    return (pf_rank, result.total_return_pct, -result.max_drawdown_pct, len(result.trades))


def optimize_mean_reversion(
    prices_dir: Path | None = None,
    train_start: date = date(2021, 1, 1),
    train_end: date = date(2024, 12, 1),
    test_start: date = date(2025, 1, 1),
    test_end: date = date(2025, 12, 31),
    top_n: int = 10,
    min_trades: int = 20,
) -> dict[str, object]:
    """Optimise mean reversion on train data and validate the finalists out of sample.

    The train end intentionally leaves a purge window before the test start so
    the 20-day maximum holding period cannot create train trades that remain
    open into the test period. The full price history is still supplied to the
    backtester so test indicators have legitimate pre-test warm-up data.
    """
    if train_start >= train_end or train_end >= test_start or test_start > test_end:
        raise ValueError("Invalid train/test date ranges")
    if top_n < 1 or min_trades < 1:
        raise ValueError("top_n and min_trades must be positive")

    prices = load_test_prices(prices_dir)
    grid = _mean_reversion_grid()
    train_candidates: list[OptimizationCandidate] = []

    for parameters in grid:
        strategy = _mean_reversion_strategy(parameters)
        result = run_backtest(
            prices,
            strategy,
            TEST_UNIVERSE,
            _backtest_config(
                max_holding_days=parameters.max_holding_days,
                entry_start_date=train_start,
                entry_end_date=train_end,
            ),
        )
        if len(result.trades) >= min_trades:
            train_candidates.append(OptimizationCandidate(parameters, result))

    train_candidates.sort(key=_candidate_sort_key, reverse=True)
    finalists = train_candidates[:top_n]

    validation: list[dict[str, object]] = []
    for candidate in finalists:
        test_result = run_backtest(
            prices,
            _mean_reversion_strategy(candidate.parameters),
            TEST_UNIVERSE,
            _backtest_config(
                max_holding_days=candidate.parameters.max_holding_days,
                entry_start_date=test_start,
                entry_end_date=test_end,
            ),
        )
        validation.append(
            {
                "parameters": candidate.parameters,
                "train": candidate.result,
                "test": test_result,
            }
        )

    return {
        "grid_size": len(grid),
        "eligible_candidates": len(train_candidates),
        "top_n": top_n,
        "min_trades": min_trades,
        "train_start": train_start,
        "train_end": train_end,
        "test_start": test_start,
        "test_end": test_end,
        "validation": validation,
    }


def _robustness_summary(
    candidates: list[tuple[MeanReversionParameters, BacktestResult]],
    parameter: str,
) -> list[RobustnessSummary]:
    grouped: dict[float | int, list[BacktestResult]] = {}
    for parameters, result in candidates:
        value = getattr(parameters, parameter)
        grouped.setdefault(value, []).append(result)

    summaries: list[RobustnessSummary] = []
    for value, results in sorted(grouped.items(), key=lambda item: item[0]):
        positive = sum(result.total_return_pct > 0 for result in results)
        profitable = sum(result.profit_factor > 1.0 for result in results)
        summaries.append(
            RobustnessSummary(
                parameter=parameter,
                value=value,
                candidates=len(results),
                positive_return_count=positive,
                profitable_factor_count=profitable,
                median_return_pct=float(pd.Series([r.total_return_pct for r in results]).median()),
                median_profit_factor=float(pd.Series([r.profit_factor for r in results]).median()),
            )
        )
    return summaries


def analyze_mean_reversion_robustness(
    prices_dir: Path | None = None,
    test_start: date = date(2025, 1, 1),
    test_end: date = date(2025, 12, 31),
    min_trades: int = 20,
) -> dict[str, object]:
    """Evaluate every eligible configuration on the untouched test period.

    This is diagnostic only: test results are never used to select a production
    strategy. The purpose is to determine whether performance is concentrated
    in one parameter combination or persists across a neighbourhood of values.
    """
    if test_start > test_end or min_trades < 1:
        raise ValueError("Invalid test range or minimum trade count")

    prices = load_test_prices(prices_dir)
    candidates: list[tuple[MeanReversionParameters, BacktestResult]] = []
    for parameters in _mean_reversion_grid():
        result = run_backtest(
            prices,
            _mean_reversion_strategy(parameters),
            TEST_UNIVERSE,
            _backtest_config(
                max_holding_days=parameters.max_holding_days,
                entry_start_date=test_start,
                entry_end_date=test_end,
            ),
        )
        if len(result.trades) >= min_trades:
            candidates.append((parameters, result))

    positive = [item for item in candidates if item[1].total_return_pct > 0]
    profitable = [item for item in candidates if item[1].profit_factor > 1.0]
    ranked_for_diagnostics = sorted(
        candidates,
        key=lambda item: (item[1].profit_factor, item[1].total_return_pct, -item[1].max_drawdown_pct),
        reverse=True,
    )

    return {
        "grid_size": len(_mean_reversion_grid()),
        "eligible_candidates": len(candidates),
        "positive_return_count": len(positive),
        "profitable_factor_count": len(profitable),
        "median_return_pct": float(pd.Series([r.total_return_pct for _, r in candidates]).median()),
        "median_profit_factor": float(pd.Series([r.profit_factor for _, r in candidates]).median()),
        "test_start": test_start,
        "test_end": test_end,
        "min_trades": min_trades,
        "top_diagnostic": [
            {"parameters": parameters, "result": result}
            for parameters, result in ranked_for_diagnostics[:10]
        ],
        "by_parameter": {
            parameter: _robustness_summary(candidates, parameter)
            for parameter in ("lookback_days", "stddevs", "rsi_threshold", "max_holding_days")
        },
    }


def run_train_test_benchmarks(
    prices_dir: Path | None = None,
    train_start: date = date(2021, 1, 1),
    train_end: date = date(2024, 12, 1),
    test_start: date = date(2025, 1, 1),
    test_end: date = date(2025, 12, 31),
) -> dict[str, dict[str, BacktestResult]]:
    """Evaluate the four fixed strategies on the same untouched train/test split."""
    prices = load_test_prices(prices_dir)
    results: dict[str, dict[str, BacktestResult]] = {}
    for name, strategy in strategy_definitions().items():
        results[name] = {
            "train": run_backtest(
                prices,
                strategy,
                TEST_UNIVERSE,
                _backtest_config(entry_start_date=train_start, entry_end_date=train_end),
            ),
            "test": run_backtest(
                prices,
                strategy,
                TEST_UNIVERSE,
                _backtest_config(entry_start_date=test_start, entry_end_date=test_end),
            ),
        }
    return results
