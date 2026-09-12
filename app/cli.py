from __future__ import annotations

import argparse
from pathlib import Path

from app.config import settings
from app.data.market_data import download_history, load_universe
from app.data.storage import MarketDataStore
from app.data.universe import download_asx_isin_directory, save_universe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ASX stock screener tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    universe = subparsers.add_parser(
        "download-universe", help="Download the current ASX ISIN directory"
    )
    universe.add_argument(
        "--output",
        type=Path,
        default=Path("data/universe.csv"),
        help="Output CSV path (default: data/universe.csv)",
    )

    download = subparsers.add_parser("download-prices", help="Download historical ASX OHLCV data")
    download.add_argument("--universe", type=Path, required=True)
    download.add_argument("--start", required=True, help="Inclusive start date, e.g. 2021-01-01")
    download.add_argument("--end", required=True, help="Exclusive end date, e.g. 2026-01-01")
    download.add_argument("--continue-on-error", action="store_true")

    ingest = subparsers.add_parser("ingest", help="Create/refresh the local DuckDB prices view")
    ingest.add_argument("--prices-dir", type=Path, default=None)

    return parser


def universe_command(args: argparse.Namespace) -> int:
    universe = download_asx_isin_directory()
    path = save_universe(universe, args.output)
    print(f"Saved {len(universe):,} ASX instruments to {path}")
    print("instrument_type is intentionally left as 'unknown' until security-type metadata is added")
    return 0


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


def ingest_command(args: argparse.Namespace) -> int:
    prices_dir = args.prices_dir or settings.prices_dir
    store = MarketDataStore(settings.duckdb_path)
    count = store.ingest_parquet_directory(prices_dir)
    print(f"Loaded {count:,} price rows into the local DuckDB view")
    return 0


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "download-universe":
        return universe_command(args)
    if args.command == "download-prices":
        return download_command(args)
    if args.command == "ingest":
        return ingest_command(args)
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
