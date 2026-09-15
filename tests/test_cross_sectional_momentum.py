from datetime import date

import pandas as pd

from app.research_cross_sectional import (
    CrossSectionalMomentumParameters,
    _momentum_ranks,
    _volatility_adjusted_ranks,
    run_cross_sectional_momentum,
)


def _prices() -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=5, freq="D")
    rows = []
    closes = {
        "CBA": [10.0, 10.0, 11.0, 12.0, 12.0],
        "BHP": [10.0, 10.0, 10.0, 11.0, 11.0],
    }
    for ticker, values in closes.items():
        for day, close in zip(dates, values):
            rows.append({"date": day, "ticker": ticker, "open": close, "high": close, "low": close, "close": close, "volume": 1_000})
    return pd.DataFrame(rows)


def test_momentum_rank_uses_only_prior_closes():
    ranks = _momentum_ranks(_prices(), lookback_days=2)
    assert pd.isna(ranks.loc[pd.Timestamp("2025-01-02"), "CBA"])
    assert ranks.loc[pd.Timestamp("2025-01-04"), "CBA"] == 1.0
    assert ranks.loc[pd.Timestamp("2025-01-04"), "BHP"] == 2.0


def test_volatility_adjusted_rank_uses_only_prior_data():
    prices = _prices()
    ranks = _volatility_adjusted_ranks(prices, lookback_days=2, volatility_days=2)
    assert pd.isna(ranks.loc[pd.Timestamp("2025-01-02"), "CBA"])
    assert ranks.loc[pd.Timestamp("2025-01-04"), "CBA"] == 1.0


def test_volatility_adjusted_parameters_flow_through_backtest():
    result = run_cross_sectional_momentum(
        _prices(),
        CrossSectionalMomentumParameters(lookback_days=2, top_n=1, max_holding_days=1, volatility_days=2),
        entry_start_date=date(2025, 1, 3),
        entry_end_date=date(2025, 1, 3),
    )
    assert len(result.trades) == 1
    assert result.trades[0].ticker == "CBA"
    assert result.trades[0].entry_date == date(2025, 1, 4)


def test_cross_sectional_backtest_enters_after_rank_signal():
    result = run_cross_sectional_momentum(
        _prices(),
        CrossSectionalMomentumParameters(lookback_days=2, top_n=1, max_holding_days=1),
        entry_start_date=date(2025, 1, 3),
        entry_end_date=date(2025, 1, 3),
    )
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.ticker == "CBA"
    assert trade.entry_date == date(2025, 1, 4)
    assert trade.entry_price == 12.0


def test_cross_sectional_holds_positions_until_scheduled_rebalance():
    dates = pd.date_range("2025-01-01", periods=9, freq="D")
    rows = []
    closes = {
        "CBA": [10.0, 10.0, 12.0, 12.0, 12.0, 8.0, 8.0, 8.0, 8.0],
        "BHP": [10.0, 10.0, 9.0, 9.0, 9.0, 13.0, 13.0, 13.0, 13.0],
    }
    for ticker, values in closes.items():
        for day, close in zip(dates, values):
            rows.append({"date": day, "ticker": ticker, "open": close, "high": close, "low": close, "close": close, "volume": 1_000})
    result = run_cross_sectional_momentum(
        pd.DataFrame(rows),
        CrossSectionalMomentumParameters(lookback_days=2, top_n=1, max_holding_days=4),
        entry_start_date=date(2025, 1, 3),
        entry_end_date=date(2025, 1, 7),
    )
    assert result.trades
    first = result.trades[0]
    assert first.ticker == "CBA"
    assert first.entry_date == date(2025, 1, 4)
    assert first.exit_date == date(2025, 1, 8)
    assert first.holding_days == 4


def test_cross_sectional_portfolio_never_exceeds_top_n_open_positions():
    dates = pd.date_range("2025-01-01", periods=8, freq="D")
    rows = []
    closes = {"CBA": [10.0, 10.0, 12.0, 9.0, 9.0, 12.0, 9.0, 9.0], "BHP": [10.0, 10.0, 9.0, 12.0, 12.0, 9.0, 12.0, 12.0]}
    for ticker, values in closes.items():
        for day, close in zip(dates, values):
            rows.append({"date": day, "ticker": ticker, "open": close, "high": close, "low": close, "close": close, "volume": 1_000})
    result = run_cross_sectional_momentum(pd.DataFrame(rows), CrossSectionalMomentumParameters(2, 1, 5))
    for left in result.trades:
        for right in result.trades:
            if left is right:
                continue
            assert not (left.entry_date < right.exit_date and right.entry_date < left.exit_date)


def test_position_size_scales_down_after_losses():
    dates = pd.date_range("2025-01-01", periods=7, freq="D")
    prices = pd.DataFrame({"date": dates, "ticker": ["CBA"] * len(dates), "open": [10.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0], "high": [10.0] * 2 + [9.0, 8.0, 7.0, 6.0, 5.0], "low": [10.0] * 2 + [9.0, 8.0, 7.0, 6.0, 5.0], "close": [10.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0], "volume": [1_000] * len(dates)})
    result = run_cross_sectional_momentum(prices, CrossSectionalMomentumParameters(1, 1, 1), entry_start_date=date(2025, 1, 2), entry_end_date=date(2025, 1, 5))
    assert len(result.trades) >= 2
    assert result.trades[1].shares * result.trades[1].entry_price < result.trades[0].shares * result.trades[0].entry_price
    assert result.final_capital > 0


def test_cash_accounting_never_produces_negative_capital():
    dates = pd.date_range("2025-01-01", periods=12, freq="D")
    rows = []
    for ticker, start in {"CBA": 100.0, "BHP": 100.0, "CSL": 100.0}.items():
        values = [start] + [1.0] * (len(dates) - 1)
        for day, close in zip(dates, values):
            rows.append({"date": day, "ticker": ticker, "open": close, "high": close, "low": close, "close": close, "volume": 1_000})
    result = run_cross_sectional_momentum(pd.DataFrame(rows), CrossSectionalMomentumParameters(1, 3, 1))
    assert result.final_capital >= 0.0
    assert result.total_return_pct >= -100.0
