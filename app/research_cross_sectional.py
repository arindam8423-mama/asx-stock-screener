from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from itertools import product
from pathlib import Path

import pandas as pd

from app.backtest.models import BacktestResult, Trade
from app.config import settings
from app.data.test_universe import TEST_UNIVERSE, all_test_tickers
from app.research import load_test_prices


@dataclass(frozen=True)
class CrossSectionalMomentumParameters:
    lookback_days: int
    top_n: int
    max_holding_days: int


def _cap_group(ticker: str) -> str:
    for group, tickers in TEST_UNIVERSE.items():
        if ticker in tickers:
            return group
    return "unknown"


def _backtest_config_values() -> tuple[float, float, float, float]:
    return 10_000.0, 10.0, 11.0, 11.0


def _momentum_ranks(prices: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    frame = prices.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    closes = frame.pivot(index="date", columns="ticker", values="close").sort_index()
    returns = closes / closes.shift(lookback_days) - 1.0
    ranks = returns.rank(axis=1, method="first", ascending=False)
    return ranks


def _result_from_trades(trades: list[Trade]) -> BacktestResult:
    returns = pd.Series([trade.return_pct for trade in trades], dtype=float)
    profits = sum(max(trade.net_pnl, 0) for trade in trades)
    losses = sum(-min(trade.net_pnl, 0) for trade in trades)
    initial_capital = 10_000.0
    equity = initial_capital
    peak = equity
    max_drawdown = 0.0
    for trade in sorted(trades, key=lambda item: item.exit_date):
        equity += trade.net_pnl
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, (peak - equity) / peak * 100)
    final_capital = initial_capital + sum(trade.net_pnl for trade in trades)
    return BacktestResult(
        trades=trades,
        initial_capital=initial_capital,
        final_capital=final_capital,
        total_return_pct=(final_capital / initial_capital - 1) * 100,
        win_rate_pct=(returns.gt(0).mean() * 100) if len(returns) else 0.0,
        profit_factor=(profits / losses) if losses else float("inf") if profits else 0.0,
        max_drawdown_pct=max_drawdown,
        average_trade_pct=float(returns.mean()) if len(returns) else 0.0,
        median_trade_pct=float(returns.median()) if len(returns) else 0.0,
        average_holding_days=float(pd.Series([t.holding_days for t in trades]).mean()) if trades else 0.0,
    )


def run_cross_sectional_momentum(
    prices: pd.DataFrame,
    parameters: CrossSectionalMomentumParameters,
    entry_start_date: date | None = None,
    entry_end_date: date | None = None,
) -> BacktestResult:
    """Backtest long-only cross-sectional momentum.

    On each session, stocks are ranked by trailing return over the configured
    lookback. The top N stocks generate signals using only information known at
    that close. Entry occurs at the following session's open. Positions exit at
    the configured maximum holding period close. Each position uses 10% of the
    fixed $10,000 research capital and pays $11 buy + $11 sell brokerage.
    """
    if parameters.lookback_days < 1 or parameters.top_n < 1 or parameters.max_holding_days < 1:
        raise ValueError("Momentum parameters must be positive")

    frame = prices.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["ticker", "date"]).reset_index(drop=True)
    ranks = _momentum_ranks(frame, parameters.lookback_days)
    initial_capital, position_size_pct, buy_brokerage, sell_brokerage = _backtest_config_values()
    allocated = initial_capital * position_size_pct / 100
    trades: list[Trade] = []

    for ticker in sorted(frame["ticker"].unique()):
        history = frame[frame["ticker"] == ticker].reset_index(drop=True)
        entry_index: int | None = None
        entry_price = 0.0
        shares = 0.0
        entry_date = None

        for i in range(len(history) - 1):
            signal_date = history.loc[i, "date"].date()
            rank_value = ranks.loc[history.loc[i, "date"], ticker]
            eligible = pd.notna(rank_value) and float(rank_value) <= parameters.top_n
            in_window = (
                (entry_start_date is None or signal_date >= entry_start_date)
                and (entry_end_date is None or signal_date <= entry_end_date)
            )

            if entry_index is None:
                if eligible and in_window:
                    execution_index = i + 1
                    entry_price = float(history.loc[execution_index, "open"])
                    shares = max(allocated / entry_price, 0.0)
                    if shares > 0:
                        entry_index = execution_index
                        entry_date = history.loc[execution_index, "date"].date()
                continue

            holding_days = i - entry_index
            if holding_days < parameters.max_holding_days:
                continue

            exit_price = float(history.loc[i, "close"])
            gross_pnl = (exit_price - entry_price) * shares
            brokerage = buy_brokerage + sell_brokerage
            net_pnl = gross_pnl - brokerage
            return_pct = net_pnl / (entry_price * shares + buy_brokerage) * 100
            trades.append(
                Trade(
                    ticker=ticker,
                    cap_group=_cap_group(ticker),
                    entry_date=entry_date,
                    exit_date=history.loc[i, "date"].date(),
                    entry_price=entry_price,
                    exit_price=exit_price,
                    shares=shares,
                    gross_pnl=gross_pnl,
                    brokerage=brokerage,
                    net_pnl=net_pnl,
                    return_pct=return_pct,
                    holding_days=holding_days,
                    exit_reason="time",
                )
            )
            entry_index = None
            entry_date = None
            shares = 0.0

        if entry_index is not None:
            i = len(history) - 1
            exit_price = float(history.loc[i, "close"])
            gross_pnl = (exit_price - entry_price) * shares
            brokerage = buy_brokerage + sell_brokerage
            net_pnl = gross_pnl - brokerage
            return_pct = net_pnl / (entry_price * shares + buy_brokerage) * 100
            trades.append(
                Trade(
                    ticker=ticker,
                    cap_group=_cap_group(ticker),
                    entry_date=entry_date,
                    exit_date=history.loc[i, "date"].date(),
                    entry_price=entry_price,
                    exit_price=exit_price,
                    shares=shares,
                    gross_pnl=gross_pnl,
                    brokerage=brokerage,
                    net_pnl=net_pnl,
                    return_pct=return_pct,
                    holding_days=i - entry_index,
                    exit_reason="end_of_data",
                )
            )

    return _result_from_trades(trades)


def _grid() -> list[CrossSectionalMomentumParameters]:
    return [
        CrossSectionalMomentumParameters(*values)
        for values in product(
            (5, 10, 20),
            (1, 3, 5),
            (5, 10, 15, 20),
        )
    ]


def _sort_key(item: tuple[CrossSectionalMomentumParameters, BacktestResult]) -> tuple[float, float, float, int]:
    parameters, result = item
    pf = result.profit_factor if result.profit_factor != float("inf") else 1_000.0
    return pf, result.total_return_pct, -result.max_drawdown_pct, len(result.trades)


def _group_metrics(result: BacktestResult) -> dict[str, dict[str, float | int]]:
    metrics: dict[str, dict[str, float | int]] = {}
    for group in ("large", "mid", "small"):
        group_trades = [trade for trade in result.trades if trade.cap_group == group]
        wins = sum(trade.net_pnl > 0 for trade in group_trades)
        metrics[group] = {
            "trades": len(group_trades),
            "net_pnl": sum(trade.net_pnl for trade in group_trades),
            "win_rate_pct": wins / len(group_trades) * 100 if group_trades else 0.0,
        }
    return metrics


def optimize_cross_sectional_momentum(
    prices_dir: Path | None = None,
    train_start: date = date(2021, 1, 1),
    train_end: date = date(2024, 12, 1),
    test_start: date = date(2025, 1, 1),
    test_end: date = date(2025, 12, 31),
    top_n_results: int = 10,
    min_trades: int = 20,
) -> dict[str, object]:
    if train_start >= train_end or train_end >= test_start or test_start > test_end:
        raise ValueError("Invalid train/test date ranges")
    prices = load_test_prices(prices_dir)
    candidates: list[tuple[CrossSectionalMomentumParameters, BacktestResult]] = []
    for parameters in _grid():
        result = run_cross_sectional_momentum(
            prices, parameters, entry_start_date=train_start, entry_end_date=train_end
        )
        if len(result.trades) >= min_trades:
            candidates.append((parameters, result))
    candidates.sort(key=_sort_key, reverse=True)
    validation = []
    for parameters, train_result in candidates[:top_n_results]:
        test_result = run_cross_sectional_momentum(
            prices, parameters, entry_start_date=test_start, entry_end_date=test_end
        )
        validation.append({"parameters": parameters, "train": train_result, "test": test_result})
    return {
        "grid_size": len(_grid()),
        "eligible_candidates": len(candidates),
        "train_start": train_start,
        "train_end": train_end,
        "test_start": test_start,
        "test_end": test_end,
        "min_trades": min_trades,
        "validation": validation,
    }


def run_cross_sectional_benchmark(prices_dir: Path | None = None) -> dict[str, object]:
    prices = load_test_prices(prices_dir)
    parameters = CrossSectionalMomentumParameters(lookback_days=20, top_n=3, max_holding_days=20)
    result = run_cross_sectional_momentum(prices, parameters)
    return {"parameters": parameters, "overall": result, "by_group": _group_metrics(result)}
