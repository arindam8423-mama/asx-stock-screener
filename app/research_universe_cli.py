from __future__ import annotations

import argparse
from pathlib import Path

from app.config import settings
from app.data.research_universe import save_current_research_universe


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the broad ASX research candidate universe")
    parser.add_argument(
        "--output",
        type=Path,
        default=settings.data_dir / "research_universe.csv",
        help="Output CSV path (default: data/research_universe.csv)",
    )
    args = parser.parse_args()

    universe = save_current_research_universe(args.output)
    print("Broad ASX research candidate universe")
    print(f"Candidates: {len(universe)}")
    print(f"Unique tickers: {universe['ticker'].nunique()}")
    print(f"Output: {args.output}")
    print()
    print("Important: this is a current candidate snapshot, not yet a survivorship-safe historical universe.")
    print("Do not use it for the final 5-year backtest until listing/delisting dates and historical membership are populated.")


if __name__ == "__main__":
    main()
