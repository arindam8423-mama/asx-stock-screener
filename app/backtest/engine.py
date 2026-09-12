from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from app.backtest.models import BacktestResult, Trade
from app.config import settings


class Strategy(Protocol):
    def entry_signal(self, history: pd.DataFrame, index: int) -> bool: ...

    def exit_signal(self, history: pd.DataFrame, entry_index: int, index: int) -> bool: ...


@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: float = 10_000.0
    position_size_pct: float = 10.0
    max_holding_days: int = settings.max_holding_days
    buy_brokerage: float = settings.buy_brokerage_aud
    sell_brokerage: float = settings.sell_brokerage_aud
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None


def _cap_group(ticker: str, groups: dict[str, list[str]]) -> str:
    for group, tickers in groups.items():
        if ticker in tickers:
            return group
    return "unknown"


def run_backtest(
    prices: pd.DataFrame,
    strategy: Strategy,
    cap_groups: dict[str, list[str]],
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Run a simple long-only, one-position-per-ticker backtest.

    Signals are evaluated on the close of a session and trades execute at the
    following session's open, avoiding same-bar look-ahead. Positions are sized
    as a percentage of initial capital. Brokerage is charged on both sides.
    """
    config = config or BacktestConfig()
    required = {"date", "ticker", "open", "high", "low", "close"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"Missing price columns: {sorted(missing)}")

    frame = prices.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["ticker", "date"]).reset_index(drop=True)

    trades: list[Trade] = []
    for ticker, history in frame.groupby("ticker", sort=True):
        history = history.reset_index(drop=True)
        entry_index: int | None = None
        entry_price = 0.0
        shares = 0.0
        entry_date = None

        for i in range(len(history) - 1):
            if entry_index is None:
                if strategy.entry_signal(history, i):
                    execution_index = i + 1
                    entry_price = float(history.loc[execution_index, "open"])
                    allocated = config.initial_capital * config.position_size_pct / 100
                    shares = max((allocated - config.buy_brokerage) / entry_price, 0.0)
                    if shares > 0:
                        entry_index = execution_index
                        entry_date = history.loc[execution_index, "date"].date()
                continue

            holding_days = i - entry_index
            close_price = float(history.loc[i, "close"])
            high_price = float(history.loc[i, "high"])
            low_price = float(history.loc[i, "low"])
            change_pct = (close_price / entry_price - 1) * 100

            reason = None
            exit_price = None
            if config.stop_loss_pct is not None and low_price <= entry_price * (1 - config.stop_loss_pct / 100):
                reason, exit_price = "stop_loss", entry_price * (1 - config.stop_loss_pct / 100)
            elif config.take_profit_pct is not None and high_price >= entry_price * (1 + config.take_profit_pct / 100):
                reason, exit_price = "take_profit", entry_price * (1 + config.take_profit_pct / 100)
            elif strategy.exit_signal(history, entry_index, i):
                reason, exit_price = "signal", close_price
            elif holding_days >= config.max_holding_days:
                reason, exit_price = "time", close_price

            if reason is None:
                continue

            gross_pnl = (exit_price - entry_price) * shares
            brokerage = config.buy_brokerage + config.sell_brokerage
            net_pnl = gross_pnl - brokerage
            return_pct = net_pnl / (entry_price * shares + config.buy_brokerage) * 100
            trades.append(
                Trade(
                    ticker=ticker,
                    cap_group=_cap_group(ticker, cap_groups),
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
                    exit_reason=reason,
                )
            )
            entry_index = None
            entry_date = None
            shares = 0.0

        if entry_index is not None:
            i = len(history) - 1
            exit_price = float(history.loc[i, "close"])
            gross_pnl = (exit_price - entry_price) * shares
            brokerage = config.buy_brokerage + config.sell_brokerage
            net_pnl = gross_pnl - brokerage
            return_pct = net_pnl / (entry_price * shares + config.buy_brokerage) * 100
            trades.append(
                Trade(
                    ticker=ticker,
                    cap_group=_cap_group(ticker, cap_groups),
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

    returns = pd.Series([trade.return_pct for trade in trades], dtype=float)
    profits = sum(max(trade.net_pnl, 0) for trade in trades)
    losses = sum(-min(trade.net_pnl, 0) for trade in trades)
    equity = config.initial_capital
    peak = equity
    max_drawdown = 0.0
    for trade in sorted(trades, key=lambda item: item.exit_date):
        equity += trade.net_pnl
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, (peak - equity) / peak * 100)

    final_capital = config.initial_capital + sum(trade.net_pnl for trade in trades)
    return BacktestResult(
        trades=trades,
        initial_capital=config.initial_capital,
        final_capital=final_capital,
        total_return_pct=(final_capital / config.initial_capital - 1) * 100,
        win_rate_pct=(returns.gt(0).mean() * 100) if len(returns) else 0.0,
        profit_factor=(profits / losses) if losses else float("inf") if profits else 0.0,
        max_drawdown_pct=max_drawdown,
        average_trade_pct=float(returns.mean()) if len(returns) else 0.0,
        median_trade_pct=float(returns.median()) if len(returns) else 0.0,
        average_holding_days=float(pd.Series([t.holding_days for t in trades]).mean()) if trades else 0.0,
    )
