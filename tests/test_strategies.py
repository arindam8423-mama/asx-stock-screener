import pandas as pd

from app.strategies import BreakoutVariantStrategy, MeanReversionStrategy, TrendFollowingStrategy


def _prices(closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=len(closes), freq="D"),
            "open": closes,
            "high": [value * 1.01 for value in closes],
            "low": [value * 0.99 for value in closes],
            "close": closes,
            "volume": [1_000] * len(closes),
        }
    )


def test_trend_following_detects_fast_slow_crossover():
    closes = [10.0] * 55
    closes.extend([10.5, 11.0, 11.5])
    history = _prices(closes)
    strategy = TrendFollowingStrategy(fast_days=5, slow_days=20)
    signals = [strategy.entry_signal(history, i) for i in range(len(history))]
    assert any(signals)


def test_mean_reversion_detects_oversold_price():
    closes = [10.0] * 20 + [8.0]
    history = _prices(closes)
    history.loc[20, "low"] = 7.8
    strategy = MeanReversionStrategy(lookback_days=20, stddevs=2.0, rsi_period=14, rsi_threshold=30.0)
    assert strategy.entry_signal(history, 20)


def test_filtered_breakout_requires_volume_confirmation():
    closes = [10.0] * 20 + [11.0]
    history = _prices(closes)
    history.loc[20, "high"] = 11.0
    history.loc[20, "volume"] = 1_100
    strategy = BreakoutVariantStrategy(volume_multiplier=1.25)
    assert not strategy.entry_signal(history, 20)
    history.loc[20, "volume"] = 1_300
    assert strategy.entry_signal(history, 20)
