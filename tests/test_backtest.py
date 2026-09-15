from datetime import date

import pandas as pd

from app.backtest.engine import BacktestConfig, run_backtest
from app.data.test_universe import TEST_UNIVERSE


class AlwaysEnterOnce:
    def entry_signal(self, history: pd.DataFrame, index: int) -> bool:
        return index == 0

    def exit_signal(self, history: pd.DataFrame, entry_index: int, index: int) -> bool:
        return index == 2


class EnterOnlyOnce:
    def entry_signal(self, history: pd.DataFrame, index: int) -> bool:
        return index == 0

    def exit_signal(self, history: pd.DataFrame, entry_index: int, index: int) -> bool:
        return False


def test_backtest_executes_next_day_and_charges_both_brokerages():
    prices = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=4, freq="D"),
            "ticker": ["CBA"] * 4,
            "open": [10.0, 10.0, 12.0, 13.0],
            "high": [10.0, 10.0, 12.0, 13.0],
            "low": [10.0, 10.0, 12.0, 13.0],
            "close": [10.0, 10.0, 12.0, 13.0],
            "volume": [100] * 4,
        }
    )
    result = run_backtest(
        prices,
        AlwaysEnterOnce(),
        TEST_UNIVERSE,
        BacktestConfig(initial_capital=1_000, position_size_pct=100),
    )

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.entry_date == date(2026, 1, 2)
    assert trade.exit_date == date(2026, 1, 3)
    assert trade.entry_price == 10.0
    assert trade.exit_price == 12.0
    assert trade.brokerage == 22.0
    assert trade.net_pnl == 178.0


def test_backtest_time_exit():
    prices = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=5, freq="D"),
            "ticker": ["BHP"] * 5,
            "open": [10.0] * 5,
            "high": [10.0] * 5,
            "low": [10.0] * 5,
            "close": [10.0] * 5,
            "volume": [100] * 5,
        }
    )
    result = run_backtest(
        prices,
        EnterOnlyOnce(),
        TEST_UNIVERSE,
        BacktestConfig(initial_capital=1_000, position_size_pct=50, max_holding_days=2),
    )
    assert result.trades[0].exit_reason == "time"
