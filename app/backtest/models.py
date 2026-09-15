from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class Trade:
    ticker: str
    cap_group: str
    entry_date: date
    exit_date: date
    entry_price: float
    exit_price: float
    shares: float
    gross_pnl: float
    brokerage: float
    net_pnl: float
    return_pct: float
    holding_days: int
    exit_reason: str


@dataclass(frozen=True)
class BacktestResult:
    trades: list[Trade]
    initial_capital: float
    final_capital: float
    total_return_pct: float
    win_rate_pct: float
    profit_factor: float
    max_drawdown_pct: float
    average_trade_pct: float
    median_trade_pct: float
    average_holding_days: float
