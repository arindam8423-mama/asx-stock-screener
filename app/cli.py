from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from app.config import settings
from app.data.historical_reference import (
    download_asx_code_changes,
    load_delistings,
    save_code_changes,
    save_delistings,
)
from app.data.market_data import download_history, load_universe
from app.data.security_master import build_security_master, save_security_master
from app.data.storage import MarketDataStore
from app.data.universe import download_asx_isin_directory, equity_universe, save_universe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ASX stock screener tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download-prices", help="Download historical ASX OHLCV data")
    download.add_argument("--universe", type=Path, required=True)
    download.add_argument("--start", required=True, help="Inclusive start date, e.g. 2021-01-01")
    download.add_argument("--end", required=True, help="Exclusive end date, e.g. 2026-01-01")
    download.add_argument("--continue-on-error", action="store_true")

    universe = subparsers.add_parser(
        "download-universe", help="Download the current ASX equity candidate universe"
    )
    universe.add_argument("--output", type=Path, default=None)
    universe.add_argument(
        "--raw-output",
        type=Path,
        default=None,
        help="Optional path for the complete unfiltered ASX instrument directory",
    )

    master = subparsers.add_parser(
        "build-security-master",
        help="Build the security-master foundation from a normalized universe CSV",
    )
    master.add_argument("--universe", type=Path, default=None)
    master.add_argument("--output", type=Path, default=None)

    code_changes = subparsers.add_parser(
        "download-code-changes", help="Download ASX historical code/name changes"
    )
    code_changes.add_argument("--output", type=Path, default=None)

    delistings = subparsers.add_parser(
        "import-delistings", help="Import an externally obtained historical delisting CSV"
    )
    delistings.add_argument("--input", type=Path, required=True)
    delistings.add_argument("--output", type=Path, default=None)

    ingest = subparsers.add_parser("ingest", help="Create/refresh the local DuckDB prices view")
    ingest.add_argument("--prices-dir", type=Path, default=None)

    return parser


def download_command(args: argparse.Namespace) -> int:
    settings.ensure_data_dirs()
    tickers = load_universe(args.universe)
    print(f"Downloading {len(tickers)} ASX tickers from {args.start} to {args.end}")

    failures = 0
    for index, ticker in enumerate(tickers, start=1):
        try:
            path = download_history(ticker, args.start, args.end, settings.prices_dir)
            print(f"[{index}/{len(tickers)}] {ticker}: {path}")
        except Exception as exc:  # noqa: BLE001 - one bad ticker should not stop a batch
            failures += 1
            print(f"[{index}/{len(tickers)}] {ticker}: ERROR: {exc}")
            if not args.continue_on_error:
                raise

    print(f"Completed with {failures} failure(s)")
    return 1 if failures else 0


def download_universe_command(args: argparse.Namespace) -> int:
    settings.ensure_data_dirs()
    raw_output = args.raw_output
    output = args.output or (settings.data_dir / "universe.csv")

    universe = download_asx_isin_directory()
    candidates = equity_universe(universe)

    if raw_output is not None:
        save_universe(universe, raw_output)
        print(f"Saved {len(universe):,} raw ASX instruments to {raw_output}")

    save_universe(candidates, output)
    print(f"Saved {len(candidates):,} equity candidates to {output}")
    print(f"Excluded {len(universe) - len(candidates):,} non-equity/other instruments")
    return 0


def build_security_master_command(args: argparse.Namespace) -> int:
    settings.ensure_data_dirs()
    universe_path = args.universe or (settings.data_dir / "universe.csv")
    output = args.output or (settings.data_dir / "security_master.csv")
    universe = pd.read_csv(universe_path)
    master = build_security_master(universe)
    save_security_master(master, output)
    print(f"Saved {len(master):,} securities to {output}")
    print("Listing/delisting dates remain unknown until historical reference data is added")
    return 0


def download_code_changes_command(args: argparse.Namespace) -> int:
    settings.ensure_data_dirs()
    output = args.output or (settings.data_dir / "code_changes.csv")
    changes = download_asx_code_changes()
    save_code_changes(changes, output)
    print(f"Saved {len(changes):,} ASX code/name changes to {output}")
    return 0


def import_delistings_command(args: argparse.Namespace) -> int:
    settings.ensure_data_dirs()
    output = args.output or (settings.data_dir / "delistings.csv")
    delisted = load_delistings(args.input)
    save_delistings(delisted, output)
    print(f"Imported {len(delisted):,} historical delistings to {output}")
    return 0


def ingest_command(args: argparse.Namespace) -> int:
    prices_dir = args.prices_dir or settings.prices_dir
    store = MarketDataStore(settings.duckdb_path)
    count = store.ingest_parquet_directory(prices_dir)
    print(f"Loaded {count:,} price rows into the local DuckDB view")
    return 0


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "download-prices":
        return download_command(args)
    if args.command == "download-universe":
        return download_universe_command(args)
    if args.command == "build-security-master":
        return build_security_master_command(args)
    if args.command == "download-code-changes":
        return download_code_changes_command(args)
    if args.command == "import-delistings":
        return import_delistings_command(args)
    if args.command == "ingest":
        return ingest_command(args)
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
