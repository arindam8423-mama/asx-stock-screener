from datetime import date

import pandas as pd

from app.research_cross_sectional import (
    CrossSectionalMomentumParameters,
    _momentum_ranks,
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
            rows.append(
                {
                    "date": day,
                    "ticker": ticker,
                    "open": close,
                    "high": close,
                    "low": close,
                    "close": close,
                    "volume": 1_000,
                }
            )
    return pd.DataFrame(rows)


def test_momentum_rank_uses_only_prior_closes():
    ranks = _momentum_ranks(_prices(), lookback_days=2)
    assert pd.isna(ranks.loc[pd.Timestamp("2025-01-02"), "CBA"])
    assert ranks.loc[pd.Timestamp("2025-01-04"), "CBA"] == 1.0
    assert ranks.loc[pd.Timestamp("2025-01-04"), "BHP"] == 2.0


def test_cross_sectional_backtest_enters_after_rank_signal():
    prices = _prices()
    result = run_cross_sectional_momentum(
        prices,
        CrossSectionalMomentumParameters(lookback_days=2, top_n=1, max_holding_days=1),
        entry_start_date=date(2025, 1, 3),
        entry_end_date=date(2025, 1, 3),
    )
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.ticker == "CBA"
    assert trade.entry_date == date(2025, 1, 4)
    assert trade.entry_price == 12.0
