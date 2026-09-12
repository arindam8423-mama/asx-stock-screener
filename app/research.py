from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.backtest.engine import BacktestConfig, run_backtest
from app.config import settings
from app.data.market_data import download_history
from app.data.test_universe import TEST_UNIVERSE, all_test_tickers
from app.strategies import BreakoutMomentumStrategy


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


def run_momentum_research(prices_dir: Path | None = None) -> dict[str, object]:
    prices = load_test_prices(prices_dir)
    result = run_backtest(
        prices,
        BreakoutMomentumStrategy(lookback_days=20, volume_lookback_days=20),
        TEST_UNIVERSE,
        BacktestConfig(initial_capital=10_000, position_size_pct=10, max_holding_days=20),
    )
    by_group = {}
    for group in ("large", "mid", "small"):
        group_trades = [trade for trade in result.trades if trade.cap_group == group]
        net_pnl = sum(trade.net_pnl for trade in group_trades)
        wins = sum(trade.net_pnl > 0 for trade in group_trades)
        by_group[group] = {
            "trades": len(group_trades),
            "net_pnl": net_pnl,
            "win_rate_pct": wins / len(group_trades) * 100 if group_trades else 0.0,
        }

    return {"overall": result, "by_group": by_group}
