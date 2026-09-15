from __future__ import annotations

from datetime import date
from itertools import product
from pathlib import Path

from app.research import load_test_prices
from app.research_cross_sectional import CrossSectionalMomentumParameters, _sort_key, run_cross_sectional_momentum


# Walk-forward uses only the raw cross-sectional momentum family.
# Trend filters and volatility adjustment have already been rejected as separate research branches.
RAW_MOMENTUM_GRID = tuple(
    CrossSectionalMomentumParameters(lookback, top_n, hold)
    for lookback, top_n, hold in product((5, 10, 20), (1, 3, 5), (5, 10, 15, 20))
)

WALK_FORWARD_WINDOWS = (
    (date(2021, 1, 1), date(2022, 12, 31), date(2023, 1, 1), date(2023, 12, 31)),
    (date(2021, 1, 1), date(2023, 12, 31), date(2024, 1, 1), date(2024, 12, 31)),
    (date(2021, 1, 1), date(2024, 12, 1), date(2025, 1, 1), date(2025, 12, 31)),
)


def run_walk_forward(prices_dir: Path | None = None, min_trades: int = 20) -> dict[str, object]:
    """Expanding-window walk-forward test using only raw cross-sectional momentum."""
    prices = load_test_prices(prices_dir)
    rows: list[dict[str, object]] = []

    for train_start, train_end, test_start, test_end in WALK_FORWARD_WINDOWS:
        candidates: list[tuple[CrossSectionalMomentumParameters, object]] = []
        for parameters in RAW_MOMENTUM_GRID:
            result = run_cross_sectional_momentum(prices, parameters, train_start, train_end)
            if len(result.trades) >= min_trades:
                candidates.append((parameters, result))
        if not candidates:
            rows.append({"train_start": train_start, "train_end": train_end, "test_start": test_start, "test_end": test_end, "candidate_count": 0})
            continue
        candidates.sort(key=_sort_key, reverse=True)
        parameters, train_result = candidates[0]
        test_result = run_cross_sectional_momentum(prices, parameters, test_start, test_end)
        rows.append({
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "candidate_count": len(candidates),
            "parameters": parameters,
            "train": train_result,
            "test": test_result,
        })

    return {"windows": rows, "grid_size": len(RAW_MOMENTUM_GRID), "min_trades": min_trades}
