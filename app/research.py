from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.backtest.engine import BacktestConfig, run_backtest
from app.backtest.models import BacktestResult
from app.config import settings
from app.data.market_data import download_history
from app.data.test_universe import TEST_UNIVERSE, all_test_tickers
from app.strategies import (
    BreakoutMomentumStrategy,
    BreakoutVariantStrategy,
    MeanReversionStrategy,
    TrendFollowingStrategy,
)


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


def _group_metrics(result: BacktestResult) -> dict[str, dict[str, float | int]]:
    by_group: dict[str, dict[str, float | int]] = {}
    for group in ("large", "mid", "small"):
        group_trades = [trade for trade in result.trades if trade.cap_group == group]
        net_pnl = sum(trade.net_pnl for trade in group_trades)
        wins = sum(trade.net_pnl > 0 for trade in group_trades)
        by_group[group] = {
            "trades": len(group_trades),
            "net_pnl": net_pnl,
            "win_rate_pct": wins / len(group_trades) * 100 if group_trades else 0.0,
        }
    return by_group


def strategy_definitions() -> dict[str, object]:
    """Return the fixed, deliberately unoptimised strategies used for comparison."""
    return {
        "Momentum breakout": BreakoutMomentumStrategy(lookback_days=20, volume_lookback_days=20),
        "Mean reversion": MeanReversionStrategy(
            lookback_days=20, stddevs=2.0, rsi_period=14, rsi_threshold=30.0
        ),
        "Trend following": TrendFollowingStrategy(fast_days=20, slow_days=50),
        "Filtered breakout": BreakoutVariantStrategy(
            lookback_days=20,
            volume_lookback_days=20,
            volume_multiplier=1.25,
            atr_period=14,
            min_atr_pct=1.0,
            max_atr_pct=8.0,
        ),
    }


def run_strategy_comparison(prices_dir: Path | None = None) -> dict[str, dict[str, object]]:
    """Run all research strategies using identical backtest assumptions."""
    prices = load_test_prices(prices_dir)
    config = BacktestConfig(initial_capital=10_000, position_size_pct=10, max_holding_days=20)
    comparison: dict[str, dict[str, object]] = {}
    for name, strategy in strategy_definitions().items():
        result = run_backtest(prices, strategy, TEST_UNIVERSE, config)
        comparison[name] = {"overall": result, "by_group": _group_metrics(result)}
    return comparison


def run_momentum_research(prices_dir: Path | None = None) -> dict[str, object]:
    prices = load_test_prices(prices_dir)
    result = run_backtest(
        prices,
        BreakoutMomentumStrategy(lookback_days=20, volume_lookback_days=20),
        TEST_UNIVERSE,
        BacktestConfig(initial_capital=10_000, position_size_pct=10, max_holding_days=20),
    )
    return {"overall": result, "by_group": _group_metrics(result)}
