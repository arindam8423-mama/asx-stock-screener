from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class MeanReversionStrategy:
    """Buy an oversold close and exit when price reverts to its moving average."""

    lookback_days: int = 20
    stddevs: float = 2.0
    rsi_period: int = 14
    rsi_threshold: float = 30.0

    def entry_signal(self, history: pd.DataFrame, index: int) -> bool:
        minimum = max(self.lookback_days, self.rsi_period)
        if index < minimum:
            return False
        close = history["close"].astype(float)
        window = close.iloc[index - self.lookback_days : index]
        mean = float(window.mean())
        std = float(window.std(ddof=0))
        rsi = _rsi(close.iloc[: index + 1], self.rsi_period)
        if pd.isna(rsi.iloc[-1]):
            return False
        lower_band = mean - self.stddevs * std
        return float(close.iloc[index]) < lower_band and float(rsi.iloc[-1]) < self.rsi_threshold

    def exit_signal(self, history: pd.DataFrame, entry_index: int, index: int) -> bool:
        if index < self.lookback_days:
            return False
        close = float(history.loc[index, "close"])
        mean = float(history["close"].iloc[index - self.lookback_days : index].mean())
        return close >= mean


@dataclass(frozen=True)
class TrendFollowingStrategy:
    """Enter an established uptrend and exit when short-term trend breaks."""

    fast_days: int = 20
    slow_days: int = 50

    def entry_signal(self, history: pd.DataFrame, index: int) -> bool:
        if index < self.slow_days:
            return False
        close = history["close"].astype(float)
        fast = float(close.iloc[index - self.fast_days + 1 : index + 1].mean())
        slow = float(close.iloc[index - self.slow_days + 1 : index + 1].mean())
        previous_fast = float(close.iloc[index - self.fast_days : index].mean())
        previous_slow = float(close.iloc[index - self.slow_days : index].mean())
        return fast > slow and previous_fast <= previous_slow and float(close.iloc[index]) > fast

    def exit_signal(self, history: pd.DataFrame, entry_index: int, index: int) -> bool:
        if index < self.fast_days:
            return False
        close = history["close"].astype(float)
        fast = float(close.iloc[index - self.fast_days + 1 : index + 1].mean())
        return float(close.iloc[index]) < fast


@dataclass(frozen=True)
class BreakoutVariantStrategy:
    """Price breakout with an ATR volatility filter and volume confirmation."""

    lookback_days: int = 20
    volume_lookback_days: int = 20
    volume_multiplier: float = 1.25
    atr_period: int = 14
    min_atr_pct: float = 1.0
    max_atr_pct: float = 8.0

    def entry_signal(self, history: pd.DataFrame, index: int) -> bool:
        minimum = max(self.lookback_days, self.volume_lookback_days, self.atr_period)
        if index < minimum:
            return False
        prior = history.iloc[index - self.lookback_days : index]
        volume_history = history.iloc[index - self.volume_lookback_days : index]
        close = float(history.loc[index, "close"])
        volume = float(history.loc[index, "volume"])
        prior_high = float(prior["high"].max())
        average_volume = float(volume_history["volume"].mean())
        atr = _atr(history.iloc[: index + 1], self.atr_period).iloc[-1]
        if pd.isna(atr) or close <= 0:
            return False
        atr_pct = float(atr) / close * 100
        return (
            close > prior_high
            and volume >= average_volume * self.volume_multiplier
            and self.min_atr_pct <= atr_pct <= self.max_atr_pct
        )

    def exit_signal(self, history: pd.DataFrame, entry_index: int, index: int) -> bool:
        return False


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    average_gain = gains.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    average_loss = losses.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    # Handle the two degenerate RSI cases explicitly:
    # no losses => RSI 100; no gains => RSI 0. If both are zero, RSI is neutral (50).
    rsi = pd.Series(index=close.index, dtype=float)
    positive_loss = average_loss > 0
    positive_gain = average_gain > 0
    normal = positive_loss & positive_gain
    rsi.loc[normal] = 100 - (100 / (1 + average_gain.loc[normal] / average_loss.loc[normal]))
    rsi.loc[positive_gain & ~positive_loss] = 100.0
    rsi.loc[~positive_gain & positive_loss] = 0.0
    rsi.loc[~positive_gain & ~positive_loss] = 50.0
    return rsi


def _atr(history: pd.DataFrame, period: int) -> pd.Series:
    high = history["high"].astype(float)
    low = history["low"].astype(float)
    close = history["close"].astype(float)
    previous_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()], axis=1
    ).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
