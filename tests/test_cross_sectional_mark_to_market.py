from datetime import date

import pandas as pd

from app.backtest.models import Trade
from app.research_cross_sectional import _mark_to_market_max_drawdown


def test_mark_to_market_drawdown_captures_intratrade_drawdown():
    dates = pd.date_range("2025-01-01", periods=5, freq="D")
    closes = [100.0, 100.0, 70.0, 100.0, 100.0]
    prices = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["CBA"] * 5,
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1_000] * 5,
        }
    )
    trade = Trade(
        ticker="CBA",
        cap_group="large",
        entry_date=date(2025, 1, 2),
        exit_date=date(2025, 1, 5),
        entry_price=100.0,
        exit_price=100.0,
        shares=10.0,
        gross_pnl=0.0,
        brokerage=22.0,
        net_pnl=-22.0,
        return_pct=-2.18,
        holding_days=3,
        exit_reason="rebalance",
    )
    drawdown = _mark_to_market_max_drawdown(prices, [trade])
    assert drawdown > 2.0
