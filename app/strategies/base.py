from __future__ import annotations

from typing import Protocol

import pandas as pd


class Strategy(Protocol):
    def entry_signal(self, history: pd.DataFrame, index: int) -> bool: ...

    def exit_signal(self, history: pd.DataFrame, entry_index: int, index: int) -> bool: ...
