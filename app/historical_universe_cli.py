from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from app.data.historical_universe import (
    HISTORICAL_METADATA_COLUMNS,
    build_historical_security_master,
    point_in_time_research_universe,
)
from app.data.research_universe import load_research_universe

DEFAULT_RESEARCH_UNIVERSE = Path("data/research_universe.csv")
DEFAULT_METADATA = Path("data/historical_security_metadata.csv")
AUDIT_DATES = ("2021-01-01", "2022-01-01", "2023-01-01", "2024-01-01", "2025-01-01")


def audit_historical_universe(
    research_path: Path,
    metadata_path: Path,
) -> None:
    """Print coverage and point-in-time eligibility without downloading prices."""
    research = load_research_universe(research_path)

    print("Historical ASX universe audit")
    print("=" * 30)
    print(f"Current candidates:       {len(research):,}")
    print(f"Unique securities/ISINs:  {research['isin'].nunique():,}")

    if not metadata_path.exists():
        print(f"\nHistorical metadata:      NOT FOUND ({metadata_path})")
        print("Listing/delisting dates are not populated yet.")
        print("No point-in-time universe will be inferred from the current snapshot.")
        return

    metadata = pd.read_csv(metadata_path)
    missing = [column for column in HISTORICAL_METADATA_COLUMNS[:3] if column not in metadata.columns]
    if missing:
        raise ValueError(f"Historical metadata is missing columns: {missing}")

    master = build_historical_security_master(research, metadata)
    listing_known = master["listing_date"].notna()
    delisting_known = master["delisting_date"].notna()

    print(f"\nListing dates known:      {listing_known.sum():,}")
    print(f"Listing dates unknown:    {(~listing_known).sum():,}")
    print(f"Delisting dates known:    {delisting_known.sum():,}")
    print(f"Delisting dates unknown:  {(~delisting_known).sum():,}")

    print("\nPoint-in-time eligibility")
    for as_of in AUDIT_DATES:
        eligible = point_in_time_research_universe(master, as_of)
        print(f"  {as_of}: {len(eligible):,}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit historical ASX research-universe coverage")
    parser.add_argument("--research-universe", type=Path, default=DEFAULT_RESEARCH_UNIVERSE)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    args = parser.parse_args()

    if not args.research_universe.exists():
        raise FileNotFoundError(
            f"Research universe not found: {args.research_universe}. "
            "Run python -m app.research_universe_cli first."
        )

    audit_historical_universe(args.research_universe, args.metadata)


if __name__ == "__main__":
    main()
