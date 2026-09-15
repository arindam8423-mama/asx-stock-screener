from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from itertools import product
from pathlib import Path

import pandas as pd

from app.backtest.models import BacktestResult, Trade
from app.data.test_universe import TEST_UNIVERSE
from app.research import load_test_prices


TREND_FILTERS = ("none", "close_above_50dma", "close_above_100dma", "close_above_50dma_and_50dma_above_100dma")


@dataclass(frozen=True)
class CrossSectionalMomentumParameters:
    lookback_days: int
    top_n: int
    max_holding_days: int
    volatility_days: int | None = None
    trend_filter: str = "none"


def _cap_group(ticker: str) -> str:
    for group, tickers in TEST_UNIVERSE.items():
        if ticker in tickers:
            return group
    return "unknown"


def _backtest_config_values() -> tuple[float, float, float, float]:
    return 10_000.0, 10.0, 11.0, 11.0


def _close_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    frame = prices.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.pivot(index="date", columns="ticker", values="close").sort_index()


def _momentum_ranks(prices: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    closes = _close_matrix(prices)
    returns = closes / closes.shift(lookback_days) - 1.0
    return returns.rank(axis=1, method="first", ascending=False)


def _volatility_adjusted_ranks(prices: pd.DataFrame, lookback_days: int, volatility_days: int) -> pd.DataFrame:
    """Rank trailing return per unit of trailing daily volatility."""
    if volatility_days < 2:
        raise ValueError("volatility_days must be at least 2")
    closes = _close_matrix(prices)
    returns = closes / closes.shift(lookback_days) - 1.0
    daily_returns = closes.pct_change()
    volatility = daily_returns.rolling(volatility_days).std()
    score = returns / volatility.replace(0.0, pd.NA)
    return score.rank(axis=1, method="first", ascending=False)


def _trend_eligibility(prices: pd.DataFrame, trend_filter: str) -> pd.DataFrame:
    """Return a date/ticker eligibility mask using information known at the signal close."""
    if trend_filter not in TREND_FILTERS:
        raise ValueError(f"Unknown trend_filter: {trend_filter}")
    closes = _close_matrix(prices)
    if trend_filter == "none":
        return closes.notna()
    dma50 = closes.rolling(50).mean()
    dma100 = closes.rolling(100).mean()
    if trend_filter == "close_above_50dma":
        return closes.gt(dma50)
    if trend_filter == "close_above_100dma":
        return closes.gt(dma100)
    return closes.gt(dma50) & dma50.gt(dma100)


def _apply_trend_filter(ranks: pd.DataFrame, eligibility: pd.DataFrame) -> pd.DataFrame:
    filtered = ranks.copy()
    common_columns = filtered.columns.intersection(eligibility.columns)
    filtered.loc[:, common_columns] = filtered.loc[:, common_columns].where(eligibility.loc[filtered.index, common_columns])
    return filtered


def _mark_to_market_max_drawdown(prices: pd.DataFrame, trades: list[Trade]) -> float:
    """Calculate max drawdown from daily portfolio equity, including open-position mark-to-market."""
    if not trades:
        return 0.0
    initial_capital, _, buy_brokerage, sell_brokerage = _backtest_config_values()
    frame = prices.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    closes = frame.pivot(index="date", columns="ticker", values="close").sort_index()
    cash = initial_capital
    equity_curve: list[float] = []
    for current_date in pd.DatetimeIndex(closes.index):
        current_day = current_date.date()
        for trade in trades:
            if trade.entry_date == current_day:
                cash -= trade.shares * trade.entry_price + buy_brokerage
            if trade.exit_date == current_day:
                cash += trade.shares * trade.exit_price - sell_brokerage
        holdings_value = 0.0
        for trade in trades:
            if trade.entry_date <= current_day < trade.exit_date and trade.ticker in closes.columns:
                close = closes.at[current_date, trade.ticker]
                if pd.notna(close):
                    holdings_value += trade.shares * float(close)
        equity_curve.append(max(0.0, cash + holdings_value))
    equity = pd.Series(equity_curve, index=closes.index, dtype=float)
    peak = equity.cummax()
    drawdown = ((peak - equity) / peak.replace(0.0, pd.NA) * 100).fillna(0.0)
    return float(drawdown.max()) if len(drawdown) else 0.0


def _result_from_trades(trades: list[Trade], final_capital: float | None = None, prices: pd.DataFrame | None = None) -> BacktestResult:
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
        if peak > 0:
            max_drawdown = max(max_drawdown, (peak - equity) / peak * 100)
    if prices is not None:
        max_drawdown = _mark_to_market_max_drawdown(prices, trades)
    if final_capital is None:
        final_capital = max(0.0, initial_capital + sum(trade.net_pnl for trade in trades))
    return BacktestResult(
        trades=trades,
        initial_capital=initial_capital,
        final_capital=max(0.0, final_capital),
        total_return_pct=(max(0.0, final_capital) / initial_capital - 1) * 100,
        win_rate_pct=(returns.gt(0).mean() * 100) if len(returns) else 0.0,
        profit_factor=(profits / losses) if losses else float("inf") if profits else 0.0,
        max_drawdown_pct=max_drawdown,
        average_trade_pct=float(returns.mean()) if len(returns) else 0.0,
        median_trade_pct=float(returns.median()) if len(returns) else 0.0,
        average_holding_days=float(pd.Series([t.holding_days for t in trades]).mean()) if trades else 0.0,
    )


def _grid() -> list[CrossSectionalMomentumParameters]:
    return [CrossSectionalMomentumParameters(lookback, top_n, hold, trend_filter=trend_filter) for lookback, top_n, hold, trend_filter in product((5, 10, 20), (1, 3, 5), (5, 10, 15, 20), TREND_FILTERS)]


def _volatility_grid() -> list[CrossSectionalMomentumParameters]:
    return [CrossSectionalMomentumParameters(lookback, top_n, hold, volatility_days, trend_filter) for lookback, top_n, hold, volatility_days, trend_filter in product((5, 10, 20), (1, 3, 5), (5, 10, 15, 20), (10, 20), ("none",))]


def _sort_key(item: tuple[CrossSectionalMomentumParameters, BacktestResult]) -> tuple[float, float, float, int]:
    _, result = item
    pf = result.profit_factor if result.profit_factor != float("inf") else 1_000.0
    return pf, result.total_return_pct, -result.max_drawdown_pct, len(result.trades)


def _group_metrics(result: BacktestResult) -> dict[str, dict[str, float | int]]:
    metrics: dict[str, dict[str, float | int]] = {}
    for group in ("large", "mid", "small"):
        group_trades = [trade for trade in result.trades if trade.cap_group == group]
        wins = sum(trade.net_pnl > 0 for trade in group_trades)
        metrics[group] = {"trades": len(group_trades), "net_pnl": sum(trade.net_pnl for trade in group_trades), "win_rate_pct": wins / len(group_trades) * 100 if group_trades else 0.0}
    return metrics


def trend_filter_pass_rates(prices: pd.DataFrame) -> dict[str, float]:
    """Percentage of stock/date observations passing each trend filter."""
    eligibility = {name: _trend_eligibility(prices, name) for name in TREND_FILTERS}
    total = float(eligibility["none"].to_numpy(dtype=bool).sum())
    return {name: float(mask.to_numpy(dtype=bool).sum() / total * 100) if total else 0.0 for name, mask in eligibility.items()}


def _optimize_grid(grid: list[CrossSectionalMomentumParameters], prices_dir: Path | None, train_start: date, train_end: date, test_start: date, test_end: date, top_n_results: int, min_trades: int) -> dict[str, object]:
    prices = load_test_prices(prices_dir)
    candidates = []
    for parameters in grid:
        result = run_cross_sectional_momentum(prices, parameters, train_start, train_end)
        if len(result.trades) >= min_trades:
            candidates.append((parameters, result))
    candidates.sort(key=_sort_key, reverse=True)
    validation = [{"parameters": parameters, "train": train_result, "test": run_cross_sectional_momentum(prices, parameters, test_start, test_end)} for parameters, train_result in candidates[:top_n_results]]
    return {"candidate_count": len(candidates), "top_results": validation}


def run_cross_sectional_benchmark(prices_dir: Path | None = None) -> dict[str, object]:
    prices = load_test_prices(prices_dir)
    parameters = CrossSectionalMomentumParameters(lookback_days=20, top_n=3, max_holding_days=20)
    result = run_cross_sectional_momentum(prices, parameters)
    return {"parameters": parameters, "overall": result, "by_group": _group_metrics(result)}


def run_volatility_adjusted_benchmark(prices_dir: Path | None = None) -> dict[str, object]:
    prices = load_test_prices(prices_dir)
    parameters = CrossSectionalMomentumParameters(lookback_days=20, top_n=3, max_holding_days=20, volatility_days=20)
    result = run_cross_sectional_momentum(prices, parameters)
    return {"parameters": parameters, "overall": result, "by_group": _group_metrics(result)}


def run_trend_filtered_benchmark(prices_dir: Path | None = None) -> dict[str, object]:
    prices = load_test_prices(prices_dir)
    parameters = CrossSectionalMomentumParameters(20, 3, 20, trend_filter="close_above_50dma_and_50dma_above_100dma")
    result = run_cross_sectional_momentum(prices, parameters)
    return {"parameters": parameters, "overall": result, "by_group": _group_metrics(result), "trend_filter_pass_rates": trend_filter_pass_rates(prices)}


def optimize_cross_sectional_momentum(prices_dir: Path | None = None, train_start: date = date(2021, 1, 1), train_end: date = date(2024, 12, 1), test_start: date = date(2025, 1, 1), test_end: date = date(2025, 12, 31), top_n_results: int = 10, min_trades: int = 20) -> dict[str, object]:
    result = _optimize_grid(_grid(), prices_dir, train_start, train_end, test_start, test_end, top_n_results, min_trades)
    result.update({"grid_size": len(_grid()), "eligible_candidates": result["candidate_count"], "validation": result.pop("top_results"), "train_start": train_start, "train_end": train_end, "test_start": test_start, "test_end": test_end, "top_n_results": top_n_results, "min_trades": min_trades, "trend_filter_pass_rates": trend_filter_pass_rates(load_test_prices(prices_dir))})
    return result


def optimize_volatility_adjusted_momentum(prices_dir: Path | None = None, top_n_results: int = 10, min_trades: int = 20) -> dict[str, object]:
    result = _optimize_grid(_volatility_grid(), prices_dir, date(2021, 1, 1), date(2024, 12, 1), date(2025, 1, 1), date(2025, 12, 31), top_n_results, min_trades)
    result.update({"grid_size": len(_volatility_grid()), "eligible_candidates": result["candidate_count"], "validation": result.pop("top_results"), "train_start": date(2021, 1, 1), "train_end": date(2024, 12, 1), "test_start": date(2025, 1, 1), "test_end": date(2025, 12, 31), "top_n_results": top_n_results, "min_trades": min_trades})
    return result


def run_cross_sectional_momentum(prices: pd.DataFrame, parameters: CrossSectionalMomentumParameters, entry_start_date: date | None = None, entry_end_date: date | None = None) -> BacktestResult:
    initial_capital, position_size_pct, buy_brokerage, sell_brokerage = _backtest_config_values()
    rank_source = _volatility_adjusted_ranks(prices, parameters.lookback_days, parameters.volatility_days) if parameters.volatility_days is not None else _momentum_ranks(prices, parameters.lookback_days)
    if parameters.trend_filter != "none":
        rank_source = _apply_trend_filter(rank_source, _trend_eligibility(prices, parameters.trend_filter))
    rank_source = rank_source.sort_index()
    prices_frame = prices.copy()
    prices_frame["date"] = pd.to_datetime(prices_frame["date"])
    prices_frame = prices_frame.sort_values(["date", "ticker"])
    common_dates = pd.DatetimeIndex(sorted(prices_frame["date"].unique()))
    price_lookup = prices_frame.set_index(["date", "ticker"])
    date_to_index = {timestamp: index for index, timestamp in enumerate(common_dates)}
    positions: dict[str, dict[str, object]] = {}
    trades: list[Trade] = []
    cash = initial_capital

    def _row(ticker: str, timestamp: pd.Timestamp):
        try:
            return price_lookup.loc[(timestamp, ticker)]
        except KeyError:
            return None

    def _target_for(signal_timestamp: pd.Timestamp) -> list[str]:
        if signal_timestamp not in rank_source.index:
            return []
        row = rank_source.loc[signal_timestamp].dropna().sort_values()
        return [ticker for ticker in row.index[: parameters.top_n]]

    def _close_positions(exit_timestamp: pd.Timestamp, reason: str) -> None:
        nonlocal cash
        for ticker, position in list(positions.items()):
            exit_row = _row(ticker, exit_timestamp)
            if exit_row is None:
                continue
            entry_price = float(position["entry_price"])
            shares = float(position["shares"])
            exit_price = float(exit_row["open"])
            gross_pnl = (exit_price - entry_price) * shares
            net_pnl = gross_pnl - sell_brokerage - buy_brokerage
            return_pct = net_pnl / (entry_price * shares + buy_brokerage) * 100
            cash += shares * exit_price - sell_brokerage
            trades.append(Trade(ticker=ticker, cap_group=_cap_group(ticker), entry_date=position["entry_date"], exit_date=exit_timestamp.date(), entry_price=entry_price, exit_price=exit_price, shares=shares, gross_pnl=gross_pnl, brokerage=buy_brokerage + sell_brokerage, net_pnl=net_pnl, return_pct=return_pct, holding_days=date_to_index[exit_timestamp] - date_to_index[pd.Timestamp(position["entry_timestamp"])], exit_reason=reason))
            del positions[ticker]

    def _open_target(target: list[str], entry_timestamp: pd.Timestamp) -> None:
        nonlocal cash
        if not target or cash <= buy_brokerage:
            return
        allocation = cash * position_size_pct / 100
        for ticker in target:
            if ticker in positions or cash <= buy_brokerage:
                continue
            entry_row = _row(ticker, entry_timestamp)
            if entry_row is None:
                continue
            entry_price = float(entry_row["open"])
            spend = min(allocation, cash)
            shares = max((spend - buy_brokerage) / entry_price, 0.0)
            if shares <= 0:
                continue
            total_cost = shares * entry_price + buy_brokerage
            if total_cost > cash:
                continue
            cash -= total_cost
            positions[ticker] = {"entry_timestamp": entry_timestamp, "entry_date": entry_timestamp.date(), "entry_price": entry_price, "shares": shares}

    next_rebalance_signal_index: int | None = None
    for signal_index, signal_value in enumerate(common_dates[:-1]):
        signal_timestamp = pd.Timestamp(signal_value)
        signal_date = signal_timestamp.date()
        next_timestamp = pd.Timestamp(common_dates[signal_index + 1])
        in_window = ((entry_start_date is None or signal_date >= entry_start_date) and (entry_end_date is None or signal_date <= entry_end_date))
        if not positions:
            if in_window:
                _open_target(_target_for(signal_timestamp), next_timestamp)
                if positions:
                    next_rebalance_signal_index = signal_index + parameters.max_holding_days
            continue
        if not in_window or (next_rebalance_signal_index is not None and signal_index >= next_rebalance_signal_index):
            reason = "rebalance" if in_window else "end_of_window"
            _close_positions(next_timestamp, reason)
            if in_window:
                _open_target(_target_for(signal_timestamp), next_timestamp)
                if positions:
                    next_rebalance_signal_index = signal_index + parameters.max_holding_days
            else:
                next_rebalance_signal_index = None

    final_timestamp = pd.Timestamp(common_dates[-1])
    if positions:
        for ticker, position in list(positions.items()):
            exit_row = _row(ticker, final_timestamp)
            if exit_row is None:
                continue
            entry_price = float(position["entry_price"])
            shares = float(position["shares"])
            exit_price = float(exit_row["close"])
            gross_pnl = (exit_price - entry_price) * shares
            net_pnl = gross_pnl - sell_brokerage - buy_brokerage
            return_pct = net_pnl / (entry_price * shares + buy_brokerage) * 100
            cash += shares * exit_price - sell_brokerage
            trades.append(Trade(ticker=ticker, cap_group=_cap_group(ticker), entry_date=position["entry_date"], exit_date=final_timestamp.date(), entry_price=entry_price, exit_price=exit_price, shares=shares, gross_pnl=gross_pnl, brokerage=buy_brokerage + sell_brokerage, net_pnl=net_pnl, return_pct=return_pct, holding_days=date_to_index[final_timestamp] - date_to_index[pd.Timestamp(position["entry_timestamp"])], exit_reason="end_of_data"))
            del positions[ticker]

    return _result_from_trades(trades, final_capital=cash, prices=prices_frame)
