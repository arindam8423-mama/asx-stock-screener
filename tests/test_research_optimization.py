from datetime import date

import pandas as pd

from app.backtest.engine import BacktestConfig, run_backtest
from app.data.test_universe import TEST_UNIVERSE
from app.research import MeanReversionParameters, _mean_reversion_grid


class _AlwaysEnterStrategy:
    def entry_signal(self, history: pd.DataFrame, index: int) -> bool:
        return True

    def exit_signal(self, history: pd.DataFrame, entry_index: int, index: int) -> bool:
        return False


def _prices() -> pd.DataFrame:
    dates = pd.date_range("2024-12-01", periods=50, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "ticker": ["CBA"] * len(dates),
            "open": [10.0] * len(dates),
            "high": [10.1] * len(dates),
            "low": [9.9] * len(dates),
            "close": [10.0] * len(dates),
            "volume": [1_000] * len(dates),
        }
    )


def test_mean_reversion_grid_has_bounded_search_space():
    grid = _mean_reversion_grid()
    assert len(grid) == 144
    assert all(isinstance(item, MeanReversionParameters) for item in grid)


def test_entry_date_bounds_preserve_history_but_restrict_new_trades():
    result = run_backtest(
        _prices(),
        _AlwaysEnterStrategy(),
        TEST_UNIVERSE,
        BacktestConfig(
            initial_capital=10_000,
            position_size_pct=10,
            max_holding_days=5,
            buy_brokerage=11,
            sell_brokerage=11,
            entry_start_date=date(2025, 1, 1),
            entry_end_date=date(2025, 1, 10),
        ),
    )

    assert result.trades
    assert all(trade.entry_date >= date(2025, 1, 2) for trade in result.trades)
    assert all(trade.entry_date <= date(2025, 1, 11) for trade in result.trades)
