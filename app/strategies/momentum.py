from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class BreakoutMomentumStrategy:
    lookback_days: int = 20
    volume_lookback_days: int = 20
    volume_multiplier: float = 1.0

    def entry_signal(self, history: pd.DataFrame, index: int) -> bool:
        if index < max(self.lookback_days, self.volume_lookback_days):
            return False
        prior = history.iloc[index - self.lookback_days : index]
        volume_history = history.iloc[index - self.volume_lookback_days : index]
        close = float(history.loc[index, "close"])
        volume = float(history.loc[index, "volume"])
        prior_high = float(prior["high"].max())
        average_volume = float(volume_history["volume"].mean())
        return close > prior_high and volume >= average_volume * self.volume_multiplier

    def exit_signal(self, history: pd.DataFrame, entry_index: int, index: int) -> bool:
        return False
